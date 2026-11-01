"""Event-log files: native event rows plus what the rows cannot say.

The native log (``BatchedEnv.enable_log``) records int32 rows, and with
``LOG_GAMES | LOG_ACTIONS`` those rows replay any game exactly. The rows do not
record which package version produced them, whose policy sat in each seat, or
any run metadata. This module writes all of that into one append-only file:

    with EventWriter("run.fcev", agents=["ppo-v3"] * 4, meta={"run": "a"}) as out:
        batch.enable_log(fastcatan.log.LOG_ALL, capacity_per_env=4096)
        batch.reset()
        while training:
            ...
            batch.step_autoreset(actions, dones)
            out.drain(batch)                       # every step or every few
            for i in np.flatnonzero(dones):        # seats changed for the new game
                out.assign(i, batch.game_id(i), agents=[...])

    log = read_events("run.fcev")
    for game in log.games():
        env = fastcatan.Env()
        for action in replay(game, env):            # verified against the rows
            ...

File layout: an 8-byte magic, then frames of ``tag (4 bytes) | length
(uint64 LE) | payload``. ``HEAD`` (JSON, first frame) stamps the format and
package versions, ``ROWS`` holds raw little-endian int32 event rows, ``GAME``
(JSON) assigns agents and metadata to one ``(env_idx, game_id)``. A run killed
mid-write leaves a file that reads up to its last complete frame.
"""
from __future__ import annotations

import json
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Iterator

import numpy as np

from . import _fastcatan as _fc

MAGIC = b"FCEVLOG\x01"
FORMAT = "fastcatan-event-log"
VERSION = 1

_FRAME = struct.Struct("<4sQ")
_EV = {name: offset for name, (offset, _) in _fc.EVENT_FIELDS.items()}
_START, _END, _ACTION = _fc.log.GAME_START, _fc.log.GAME_END, _fc.log.ACTION
_PLAYERS = _fc.NUM_PLAYERS


def _package_version() -> str:
    from importlib.metadata import version
    return version("fastcatan")


def _jsonable(x: Any) -> Any:
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _check_agents(agents: Any) -> list[str] | None:
    if agents is None:
        return None
    agents = [str(a) for a in agents]
    if len(agents) != _PLAYERS:
        raise ValueError(f"agents needs {_PLAYERS} names, got {len(agents)}")
    return agents


# --- writing -------------------------------------------------------------------

class EventWriter:
    """Append-only event-log file. Use as a context manager.

    `agents` (one name per seat) and `meta` are the run-wide defaults; `assign`
    overrides them for a single game.
    """

    def __init__(self, path: str | Path, *, agents: list[str] | None = None,
                 meta: dict | None = None):
        self.path = Path(path)
        self._agents = _check_agents(agents)
        self._meta = _jsonable(meta or {})
        self._fh: IO[bytes] | None = None
        self.rows_written = 0
        self._overflow_warned = False

    def __enter__(self) -> "EventWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "wb")
        self._fh.write(MAGIC)
        self._frame(b"HEAD", json.dumps({
            "format": FORMAT,
            "version": VERSION,
            "fastcatan": _package_version(),
            "event_version": int(_fc.EVENT_VERSION),
            "event_ints": int(_fc.EVENT_INTS),
            "agents": self._agents,
            "meta": self._meta,
        }).encode())
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def _frame(self, tag: bytes, payload: bytes) -> None:
        if self._fh is None:
            raise RuntimeError("EventWriter is not open; use it as a context manager")
        self._fh.write(_FRAME.pack(tag, len(payload)))
        self._fh.write(payload)

    def write(self, rows: np.ndarray) -> None:
        """Append event rows shaped ``(k, EVENT_INTS)``."""
        rows = np.asarray(rows)
        if rows.ndim != 2 or rows.shape[1] != _fc.EVENT_INTS:
            raise ValueError(f"rows must be shaped (k, {_fc.EVENT_INTS}), got {rows.shape}")
        if len(rows) == 0:
            return
        self._frame(b"ROWS", np.ascontiguousarray(rows, dtype="<i4").tobytes())
        self.rows_written += len(rows)

    def drain(self, batch: Any) -> int:
        """Move every pending row of `batch` into the file; returns the count.

        Warns once if the native buffers overflowed: games that lost rows
        cannot be replayed (the replay reports them as missing rows).
        """
        pending = batch.pending_events()
        if pending:
            buffer = np.empty((pending, _fc.EVENT_INTS), dtype=np.int32)
            count = batch.drain_events(buffer)
            self.write(buffer[:count])
        else:
            count = 0
        if batch.log_overflow and not self._overflow_warned:
            warnings.warn(f"event log overflowed ({batch.log_overflow} rows refused); "
                          "drain more often or raise capacity_per_env")
            self._overflow_warned = True
        return count

    def assign(self, env_idx: int, game_id: int, *, agents: list[str] | None = None,
               meta: dict | None = None) -> None:
        """Name the agents (one per seat) and metadata of one game."""
        self._frame(b"GAME", json.dumps({
            "env_idx": int(env_idx), "game_id": int(game_id),
            "agents": _check_agents(agents), "meta": _jsonable(meta or {}),
        }).encode())


# --- reading -------------------------------------------------------------------

@dataclass
class Game:
    """One game's rows, in the order they were recorded."""
    env_idx: int
    game_id: int
    rows: np.ndarray
    agents: list[str] | None = None
    meta: dict = field(default_factory=dict)

    @property
    def start(self) -> np.ndarray | None:
        hit = self.rows[self.rows[:, _EV["event_type"]] == _START]
        return hit[0] if len(hit) else None

    @property
    def end(self) -> np.ndarray | None:
        hit = self.rows[self.rows[:, _EV["event_type"]] == _END]
        return hit[0] if len(hit) else None

    @property
    def actions(self) -> np.ndarray:
        return self.rows[self.rows[:, _EV["event_type"]] == _ACTION]

    @property
    def seed(self) -> int | None:
        row = self.start
        if row is None:
            return None
        return ((int(row[_EV["seed_hi"]]) & 0xFFFFFFFF) << 32) \
            | (int(row[_EV["seed_lo"]]) & 0xFFFFFFFF)

    @property
    def finished(self) -> bool:
        return self.end is not None

    @property
    def winner(self) -> int | None:
        row = self.end
        return None if row is None else int(row[_EV["winner"]])

    @property
    def truncated(self) -> bool:
        """Ended by MAX_TURNS with no winner."""
        return self.finished and self.winner == _fc.NO_PLAYER


@dataclass
class EventLog:
    header: dict
    rows: np.ndarray
    assignments: dict[tuple[int, int], dict]
    cut: bool = False

    def games(self) -> list[Game]:
        """Every game in the file, ordered by (env_idx, game_id)."""
        if len(self.rows) == 0:
            return []
        env = self.rows[:, _EV["env_idx"]]
        gid = self.rows[:, _EV["game_id"]]
        order = np.lexsort((gid, env))        # stable: keeps append order within a game
        rows = self.rows[order]
        keys = np.stack([rows[:, _EV["env_idx"]], rows[:, _EV["game_id"]]], axis=1)
        cuts = np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1
        out = []
        for chunk in np.split(rows, cuts):
            e, g = int(chunk[0, _EV["env_idx"]]), int(chunk[0, _EV["game_id"]])
            a = self.assignments.get((e, g), {})
            out.append(Game(e, g, chunk,
                            agents=a.get("agents") or self.header.get("agents"),
                            meta={**self.header.get("meta", {}), **a.get("meta", {})}))
        return out


def read_events(path: str | Path) -> EventLog:
    """Load an `EventWriter` file. A file cut off mid-write loads up to its
    last complete frame, with a warning, and reports ``cut=True``."""
    path = Path(path)
    data = path.read_bytes()
    if not data.startswith(MAGIC):
        raise ValueError(f"{path}: not a {FORMAT} file")
    pos, header, chunks, assignments, cut = len(MAGIC), None, [], {}, False
    while pos < len(data):
        if pos + _FRAME.size > len(data):
            cut = True
            break
        tag, length = _FRAME.unpack_from(data, pos)
        start, pos = pos + _FRAME.size, pos + _FRAME.size + length
        if pos > len(data):
            cut = True
            break
        payload = data[start:pos]
        if tag == b"HEAD":
            header = json.loads(payload)
            if header.get("format") != FORMAT or header.get("version") != VERSION:
                raise ValueError(f"{path}: unsupported {FORMAT} version {header.get('version')}")
            if header["event_ints"] != _fc.EVENT_INTS or header["event_version"] != _fc.EVENT_VERSION:
                raise ValueError(
                    f"{path}: event rows are EVENT_VERSION {header['event_version']}, "
                    f"this fastcatan reads {_fc.EVENT_VERSION}")
        elif header is None:
            raise ValueError(f"{path}: frame before the header")
        elif tag == b"ROWS":
            chunks.append(np.frombuffer(payload, dtype="<i4").reshape(-1, _fc.EVENT_INTS))
        elif tag == b"GAME":
            a = json.loads(payload)
            assignments[(a["env_idx"], a["game_id"])] = a
    if header is None:
        raise ValueError(f"{path}: no header")
    if cut:
        warnings.warn(f"{path}: file is cut off; reading the complete frames only")
    rows = (np.concatenate(chunks) if chunks
            else np.zeros((0, _fc.EVENT_INTS), dtype=np.int32)).astype(np.int32)
    return EventLog(header, rows, assignments, cut)


# --- replay --------------------------------------------------------------------

class ReplayError(RuntimeError):
    """The rows cannot be replayed: missing rows, or the simulator diverged."""


def check_version(log: EventLog) -> None:
    """Raise unless the installed package is the one that recorded `log`.

    Replays depend on the exact rules and random-number streams, which can
    change between versions; `replay` also detects a divergence step by step.
    """
    recorded, installed = log.header["fastcatan"], _package_version()
    if recorded != installed:
        raise ReplayError(f"log was recorded with fastcatan {recorded}, "
                          f"installed is {installed}; install {recorded} to replay it")


def replay(game: Game, env: Any) -> Iterator[int]:
    """Reset `env` (an ``Env``) to `game` and step through its actions,
    yielding each action right after it is applied.

    Every action row's pre-step context (turn, acting seat, VP, hand sizes) is
    compared with `env` before stepping, and the game end with the final state,
    so a version or rules mismatch fails at the first step that differs.
    Actions stepped after the game ended are no-ops and are not yielded.
    """
    where = f"env {game.env_idx} game {game.game_id}"
    seed = game.seed
    if seed is None:
        raise ReplayError(f"{where}: no GAME_START row (recorded without LOG_GAMES, "
                          "or logging was enabled after the game began)")
    env.reset(seed)
    expected = 0
    for row in game.actions:
        step = int(row[_EV["step_in_game"]])
        if step != expected:
            raise ReplayError(f"{where}: rows missing before step {step} "
                              f"(expected {expected}; log overflow or LOG_ACTIONS off)")
        expected += 1
        if env.done:
            continue
        got = (env.turn_count, env.actor_to_act,
               [env.player_vp(p) for p in range(_PLAYERS)],
               [env.player_vp_public(p) for p in range(_PLAYERS)],
               [env.player_handsize(p) for p in range(_PLAYERS)])
        want = (int(row[_EV["turn"]]), int(row[_EV["actor"]]),
                row[_EV["true_vp"]:_EV["true_vp"] + _PLAYERS].tolist(),
                row[_EV["public_vp"]:_EV["public_vp"] + _PLAYERS].tolist(),
                row[_EV["handsize"]:_EV["handsize"] + _PLAYERS].tolist())
        if got != want:
            raise ReplayError(f"{where}: diverged before step {step}: "
                              f"(turn, actor, vp, public vp, hands) is {got}, log says {want}")
        action = int(row[_EV["action"]])
        env.step(action)
        yield action

    end = game.end
    if end is not None:
        if not env.done:
            raise ReplayError(f"{where}: log ends the game but the replay is still running "
                              f"after {expected} actions (rows missing?)")
        if env.winner != int(end[_EV["winner"]]) or env.turn_count != int(end[_EV["turns"]]):
            raise ReplayError(f"{where}: replay ended with winner {env.winner} on turn "
                              f"{env.turn_count}, log says {int(end[_EV['winner']])} on "
                              f"turn {int(end[_EV['turns']])}")
