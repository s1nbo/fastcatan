"""Detailed game logs for evaluation games.

Every record carries the full state after it, so a log is self-contained: the
viewer draws it without the simulator, the agents, or their checkpoints.

A file is JSON Lines (gzipped when the path ends in ``.gz``) holding one or
more games, each written as

    {"kind": "game", ...}   header: seed, agents, meta, board, topology,
                            action names, initial state
    {"kind": "step", ...}   one per applied action: actor, action, text, the
                            legal actions the actor had, optional agent `info`,
                            and the full `state` after the action
    {"kind": "end",  ...}   winner, turns, steps, VP (only if the game ended)

Recording an evaluation game on an ``Env``::

    with GameLogWriter("viz/logs/eval.jsonl.gz") as log:
        env.reset(seed)
        rec = log.start_game(env, seed=seed, agents=["ppo-v3", "random", "random", "random"])
        while not env.done:
            action = agents[env.actor_to_act].act(env)
            env.step(action)
            rec.record(action)            # after the step; info={...} for agent output

For a ``BatchedEnv`` slot pass ``env_idx=i`` to `start_game` and call
`record` after each batched step (use ``step``, not ``step_autoreset``).
"""
from __future__ import annotations

import gzip
import json
import math
import re
import warnings
import zlib
from pathlib import Path
from typing import IO, Any

import fastcatan

FORMAT = "fastcatan-game-log"
VERSION = 1

ACT = fastcatan.action
PLAYERS = fastcatan.NUM_PLAYERS
NO_PLAYER = fastcatan.NO_PLAYER
RES = ["brick", "lumber", "wool", "grain", "ore"]
DEV = ["knight", "victory point", "road building", "year of plenty", "monopoly"]
FLAG_PLACE_ROAD = 6
FLAG_TRADE_PENDING = 7
RESPONSE_DECLINE = 2

TOPOLOGY_HPP = Path(__file__).resolve().parent.parent / "include" / "topology.hpp"


def _jsonable(x: Any) -> Any:
    """Agent output as plain JSON: numpy / torch values become numbers or
    lists, and non-finite floats become strings, since NaN is not valid JSON
    and would make the whole log unreadable in the browser."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, float):
        return x if math.isfinite(x) else str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _topology() -> dict:
    """Board adjacency the viewer needs, parsed from include/topology.hpp."""
    text = TOPOLOGY_HPP.read_text()

    def table(name: str) -> list[list[int]]:
        m = re.search(rf"\b{name}\s*=\s*\{{\{{(.*?)\}}\}}\s*;", text, re.DOTALL)
        if not m:
            raise KeyError(f"{name} not found in {TOPOLOGY_HPP}")
        return [[int(x, 16) for x in re.findall(r"0x[0-9A-Fa-f]+", body) if int(x, 16) != 0xFF]
                for body in re.findall(r"\{\{([^{}]*)\}\}", m.group(1))]

    return {"hex_nodes": table("hex_to_node"), "edge_nodes": table("edge_to_node"),
            "port_nodes": table("port_to_node")}


# --- state capture -------------------------------------------------------------

def _views(env: Any, env_idx: int | None) -> list[dict]:
    if env_idx is None:
        return [env.state_view(p) for p in range(PLAYERS)]
    return [env.state_view(env_idx, p) for p in range(PLAYERS)]


def capture(views: list[dict]) -> dict:
    """Full (omniscient) state from the four per-seat views."""
    v = views[0]
    players = []
    for p in range(PLAYERS):
        q = views[p]["players"][p]
        players.append({
            "vp": q["victory_points"],
            "public_vp": q["public_victory_points"],
            "hand": q["hand_size"],
            "res": q["resources"],
            "dev": q["development_cards"],
            "dev_new": q["development_cards_bought_this_turn"],
            "knights": q["knights_played"],
            "road": q["road_length"],
            "settlements_left": q["settlements_remaining"],
            "cities_left": q["cities_remaining"],
            "roads_left": q["roads_remaining"],
            "discard": q["discard_remaining"],
            "ports": q["ports"],
        })

    trade = None
    proposer = v["trade"]["proposer"]
    if proposer != NO_PLAYER:
        t = views[proposer]["trade"]  # the proposer always sees its own draft
        trade = {"proposer": proposer, "open": v["flag"] == FLAG_TRADE_PENDING,
                 "give": t["give"], "want": t["want"], "responses": t["responses"]}

    return {
        "turn": v["turn_count"],
        "phase": v["phase"],
        "flag": v["flag"],
        "current": v["current_player"],
        "actor": v["actor_to_act"],
        "dice": v["dice_roll"],
        "robber": v["robber_hex"],
        "longest_road": v["longest_road_owner"],
        "largest_army": v["largest_army_owner"],
        "dev_deck": v["development_deck_size"],
        "free_roads": v["free_roads_remaining"],
        "done": v["done"],
        "winner": v["winner"],
        "bank": v["bank"],
        # One char per node ('.', 'a'-'d' settlement, 'A'-'D' city, by seat)
        # and per edge ('.' or the owning seat).
        "nodes": "".join("." if lvl == 0 else chr((97 if lvl == 1 else 65) + owner)
                         for lvl, owner in zip(v["node_levels"], v["node_owners"])),
        "edges": "".join("." if o == NO_PLAYER else str(o) for o in v["edge_owners"]),
        "players": players,
        "trade": trade,
    }


# --- action text ---------------------------------------------------------------

_ACTION_NAMES = [
    (ACT.SETTLE_BASE, "SETTLE"), (ACT.CITY_BASE, "CITY"), (ACT.ROAD_BASE, "ROAD"),
    (ACT.ROLL_DICE, "ROLL_DICE"), (ACT.END_TURN, "END_TURN"),
    (ACT.DISCARD_BASE, "DISCARD"), (ACT.MOVE_ROBBER_BASE, "MOVE_ROBBER"),
    (ACT.STEAL_BASE, "STEAL"), (ACT.TRADE_BASE, "BANK_TRADE"),
    (ACT.BUY_DEV, "BUY_DEV"), (ACT.PLAY_KNIGHT, "PLAY_KNIGHT"),
    (ACT.PLAY_ROAD_BUILDING, "PLAY_ROAD_BUILDING"),
    (ACT.PLAY_YEAR_OF_PLENTY, "PLAY_YEAR_OF_PLENTY"),
    (ACT.PLAY_MONOPOLY, "PLAY_MONOPOLY"), (ACT.TRADE_ADD_GIVE_BASE, "TRADE_ADD_GIVE"),
    (ACT.TRADE_ADD_WANT_BASE, "TRADE_ADD_WANT"), (ACT.TRADE_OPEN, "TRADE_OPEN"),
    (ACT.TRADE_ACCEPT, "TRADE_ACCEPT"), (ACT.TRADE_DECLINE, "TRADE_DECLINE"),
    (ACT.TRADE_CONFIRM_BASE, "TRADE_CONFIRM"), (ACT.TRADE_CANCEL, "TRADE_CANCEL"),
    (fastcatan.NUM_ACTIONS, ""),
]


def action_label(action: int) -> str:
    """Short name of an action ID: ROAD(17), END_TURN, ..."""
    for (base, name), (nxt, _) in zip(_ACTION_NAMES, _ACTION_NAMES[1:]):
        if base <= action < nxt:
            return name if nxt - base == 1 else f"{name}({action - base})"
    return f"ACTION({action})"


def _bundle(counts: list[int]) -> str:
    parts = [f"{n} {RES[r]}" for r, n in enumerate(counts) if n]
    return " + ".join(parts) if parts else "nothing"


def describe(action: int, before: dict, after: dict,
             hexinfo: list[str]) -> tuple[str, str, dict]:
    """(type, sentence, board highlight) for a legal action, from the states
    on either side of it."""
    actor = before["actor"]
    mine = [a - b for a, b in zip(after["players"][actor]["res"],
                                  before["players"][actor]["res"])]

    if action < ACT.CITY_BASE:
        return "build", f"built a settlement on node {action}", {"node": action}
    if action < ACT.ROAD_BASE:
        n = action - ACT.CITY_BASE
        return "build", f"upgraded node {n} to a city", {"node": n}
    if action < ACT.ROLL_DICE:
        e = action - ACT.ROAD_BASE
        free = " (free)" if before["phase"] < 2 or before["flag"] == FLAG_PLACE_ROAD else ""
        return "build", f"built a road on edge {e}{free}", {"edge": e}
    if action == ACT.ROLL_DICE:
        d = after["dice"]
        return "roll", f"rolled {d}" + (" — robber!" if d == 7 else ""), {}
    if action == ACT.END_TURN:
        return "turn", "ended the turn", {}
    if action < ACT.MOVE_ROBBER_BASE:
        return "robber", f"discarded 1 {RES[action - ACT.DISCARD_BASE]}", {}
    if action < ACT.STEAL_BASE:
        h = action - ACT.MOVE_ROBBER_BASE
        return "robber", f"moved the robber to hex {h} ({hexinfo[h]})", {"hex": h}
    if action < ACT.TRADE_BASE:
        took = [r for r in range(5) if mine[r] > 0]
        what = f"1 {RES[took[0]]}" if took else "nothing"
        return "robber", f"stole {what} from P{action - ACT.STEAL_BASE}", {}
    if action < ACT.BUY_DEV:
        give, get = divmod(action - ACT.TRADE_BASE, 5)
        return "bank", f"traded {-mine[give]} {RES[give]} for 1 {RES[get]} with the bank", {}
    if action == ACT.BUY_DEV:
        # Purchases land in dev_new, except victory points, which count at once.
        def held(state: dict, d: int) -> int:
            q = state["players"][actor]
            return q["dev"][d] + q["dev_new"][d]
        got = [d for d in range(5) if held(after, d) > held(before, d)]
        return "dev", f"bought a development card ({DEV[got[0]] if got else '?'})", {}
    if action == ACT.PLAY_KNIGHT:
        return "dev", "played a knight", {}
    if action == ACT.PLAY_ROAD_BUILDING:
        return "dev", "played road building", {}
    if action < ACT.PLAY_MONOPOLY:
        a, b = divmod(action - ACT.PLAY_YEAR_OF_PLENTY, 5)
        return "dev", f"played year of plenty: {RES[a]} + {RES[b]}", {}
    if action < ACT.TRADE_ADD_GIVE_BASE:
        r = action - ACT.PLAY_MONOPOLY
        return "dev", f"played monopoly on {RES[r]} (took {mine[r]})", {}
    if action < ACT.TRADE_ADD_WANT_BASE:
        return "draft", f"offers +1 {RES[action - ACT.TRADE_ADD_GIVE_BASE]}", {}
    if action < ACT.TRADE_OPEN:
        return "draft", f"asks for +1 {RES[action - ACT.TRADE_ADD_WANT_BASE]}", {}

    trade = before["trade"] or {"give": [0] * 5, "want": [0] * 5}
    if action == ACT.TRADE_OPEN:
        return "trade", (f"opened a trade: gives {_bundle(trade['give'])} "
                         f"for {_bundle(trade['want'])}"), {}
    if action == ACT.TRADE_ACCEPT:
        responses = (after["trade"] or {}).get("responses") or []
        if actor < len(responses) and responses[actor] == RESPONSE_DECLINE:
            return "trade", "tried to accept but cannot pay — declined", {}
        return "trade", "accepted the trade", {}
    if action == ACT.TRADE_DECLINE:
        return "trade", "declined the trade", {}
    if action < ACT.TRADE_CANCEL:
        return "trade", (f"traded with P{action - ACT.TRADE_CONFIRM_BASE}: gave "
                         f"{_bundle(trade['give'])}, got {_bundle(trade['want'])}"), {}
    return "trade", "cancelled the trade", {}


# --- writing -------------------------------------------------------------------

class GameRecorder:
    """Records one game. Call `record` after every applied action, in order."""

    def __init__(self, writer: "GameLogWriter", env: Any, env_idx: int | None,
                 seed: int | None, agents: list[str] | None, meta: dict | None):
        if agents is not None and len(agents) != PLAYERS:
            raise ValueError(f"agents needs {PLAYERS} names, got {len(agents)}")
        self._writer = writer
        self._env = env
        self._idx = env_idx
        self.steps = 0
        self.finished = False

        self._state, self._legal = self._capture()
        first = _views(env, env_idx)[0]
        self._hexinfo = ["desert" if r == 5 else f"{RES[r]} {n}"
                         for r, n in zip(first["hex_resources"], first["hex_numbers"])]
        if seed is None and env_idx is not None:
            seed = env.game_seed(env_idx)
        writer._emit({
            "kind": "game",
            "format": FORMAT,
            "version": VERSION,
            "fastcatan": fastcatan.__version__,
            "seed": None if seed is None else str(seed),  # 64-bit: unsafe as a JS number
            "agents": None if agents is None else [str(a) for a in agents],
            "meta": _jsonable(meta or {}),
            "board": {"hex_resources": first["hex_resources"],
                      "hex_numbers": first["hex_numbers"],
                      "port_types": first["port_types"]},
            "topology": writer.topology,
            # [first id, name] per action family, closed by [NUM_ACTIONS, ""].
            "action_names": [[base, name] for base, name in _ACTION_NAMES],
            "state": self._state,
        })

    def _capture(self) -> tuple[dict, list[int]]:
        views = _views(self._env, self._idx)
        state = capture(views)
        legal = [] if state["done"] else views[state["actor"]]["legal_actions"] or []
        return state, legal

    def record(self, action: int, info: dict | None = None,
               warnings: list[str] | None = None) -> None:
        """Log `action`, which has just been applied to the env.

        `info` is free-form agent output (policy, value, ...) shown with the
        step; `warnings` are flagged in red by the viewer.
        """
        if self.finished:
            raise RuntimeError("game already ended; call start_game for the next one")
        action = int(action)
        before, legal = self._state, self._legal
        after, self._legal = self._capture()

        if action in legal:
            kind, text, hl = describe(action, before, after, self._hexinfo)
        else:
            kind, text, hl = "noop", f"{action_label(action)} — illegal, no-op", {}
        rec: dict[str, Any] = {
            "kind": "step", "i": self.steps, "actor": before["actor"], "action": action,
            "name": action_label(action), "type": kind, "text": text, "legal": legal,
        }
        if hl:
            rec["hl"] = hl
        if info:
            rec["info"] = _jsonable(info)
        if warnings:
            rec["warnings"] = [str(w) for w in warnings]
        rec["state"] = after
        self._writer._emit(rec)
        self.steps += 1
        self._state = after

        if after["done"]:
            self._writer._emit({
                "kind": "end", "winner": after["winner"], "turns": after["turn"],
                "steps": self.steps, "vp": [q["vp"] for q in after["players"]],
            })
            self._writer.flush()
            self.finished = True


class GameLogWriter:
    """A log file holding any number of games. Use as a context manager."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.topology = _topology()
        self._fh: IO[str] | None = None

    def __enter__(self) -> "GameLogWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.suffix == ".gz":
            self._fh = gzip.open(self.path, "wt", encoding="utf-8")
        else:
            self._fh = open(self.path, "w", encoding="utf-8")
        return self

    def __exit__(self, *_exc) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _emit(self, obj: dict) -> None:
        assert self._fh is not None, "writer not open"
        self._fh.write(json.dumps(obj, separators=(",", ":"), default=str))
        self._fh.write("\n")

    def flush(self) -> None:
        """Push everything written so far to disk. Called after each finished
        game, so a run killed later still leaves those games readable (gzip
        flushes with Z_SYNC_FLUSH, making the data decodable up to here)."""
        if self._fh is not None:
            self._fh.flush()

    def start_game(self, env: Any, *, env_idx: int | None = None, seed: int | None = None,
                   agents: list[str] | None = None, meta: dict | None = None) -> GameRecorder:
        """Begin a game on a freshly reset `env` (or BatchedEnv slot `env_idx`)."""
        return GameRecorder(self, env, env_idx, seed, agents, meta)


# --- reading -------------------------------------------------------------------

def _lines(path: Path) -> tuple[list[str], bool]:
    """The file's lines, plus whether it was cut off (a killed run leaves a
    gzip stream without its end marker and possibly a half-written line)."""
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            data = d.decompress(raw) + d.flush()
        except zlib.error:
            data, cut = b"", True  # corrupt rather than truncated
        else:
            cut = not d.eof
    else:
        data, cut = raw, False
    lines = data.decode("utf-8", errors="replace").split("\n")
    if lines and lines[-1].strip():
        cut = True  # every record ends with a newline; this one never finished
        lines.pop()
    return lines, cut


def read_game_log(path: str | Path) -> list[dict]:
    """Load every game in a log: ``[{"header", "steps", "end"}, ...]``.

    A log cut off mid-write (killed run) loads up to the last complete record,
    with a warning; its last game then has no "end".
    """
    path = Path(path)
    lines, cut = _lines(path)
    if cut:
        warnings.warn(f"{path}: log is cut off; reading the complete records only")
    games: list[dict] = []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["kind"] == "game":
            if rec.get("format") != FORMAT or rec.get("version") != VERSION:
                raise ValueError(f"{path}:{n}: not a {FORMAT} v{VERSION} record")
            games.append({"header": rec, "steps": [], "end": None})
        elif not games:
            raise ValueError(f"{path}:{n}: {rec['kind']} record before any game header")
        elif rec["kind"] == "step":
            games[-1]["steps"].append(rec)
        elif rec["kind"] == "end":
            games[-1]["end"] = rec
    return games
