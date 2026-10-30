"""Native event log: exact agreement with a pure-Python reference logger."""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import pytest

import fastcatan
from fastcatan import log as ev
from tests.conftest import legal_actions


NO_PLAYER = fastcatan.NO_PLAYER
NO_ACTION = fastcatan.NO_ACTION
PLAYERS = fastcatan.NUM_PLAYERS
RESOURCES = fastcatan.NUM_RESOURCES
PHASE_ENDED = 3

ACT = fastcatan.action
TRADE_FIRST = ACT.TRADE_OPEN      # 278
TRADE_LAST = ACT.TRADE_CANCEL     # 285
RESPONSE_ACCEPT = 1


# --- reference logger ------------------------------------------------------
#
# Builds the expected rows from per-env getters alone, following the recording
# rules documented in README: pre-step context for everything except a game
# end, which reports the finished game; a trade action counts as effective
# exactly when its pre-step mask bit is set; and an accept the simulator
# coerced into a decline is recorded as the decline that happened.


# Hot-path offsets, resolved once from the published layout.
F = {name: offset for name, (offset, _) in fastcatan.EVENT_FIELDS.items()}

# A row with every optional field already at its "unused" sentinel.
BLANK_ROW = [0] * fastcatan.EVENT_INTS
for _name in ("action", "proposer", "partner", "winner", "turns", "steps",
              "seed_lo", "seed_hi"):
    BLANK_ROW[F[_name]] = -1



def put(row: list[int], name: str, value) -> None:
    offset, length = fastcatan.EVENT_FIELDS[name]
    if length == 1:
        row[offset] = int(value)
    else:
        assert len(value) == length
        row[offset:offset + length] = [int(v) for v in value]


def get(row, name: str):
    offset, length = fastcatan.EVENT_FIELDS[name]
    return int(row[offset]) if length == 1 else [int(v) for v in row[offset:offset + length]]


def context_row(batch, i: int, game_id: int, step_in_game: int,
                event_type: int, actor: int, action: int) -> list[int]:
    """A row whose player and turn context comes from the live state."""
    row = list(BLANK_ROW)
    row[F["env_idx"]] = i
    row[F["game_id"]] = game_id
    row[F["step_in_game"]] = step_in_game
    row[F["turn"]] = batch.turn_count(i)
    row[F["event_type"]] = event_type
    row[F["actor"]] = actor
    row[F["action"]] = action
    at = F["public_vp"]
    row[at:at + PLAYERS] = [batch.player_vp_public(i, p) for p in range(PLAYERS)]
    at = F["true_vp"]
    row[at:at + PLAYERS] = [batch.player_vp(i, p) for p in range(PLAYERS)]
    at = F["handsize"]
    row[at:at + PLAYERS] = [batch.player_handsize(i, p) for p in range(PLAYERS)]
    return row


class ReferenceLogger:
    """Expected event rows, per env, in append order."""

    def __init__(self, batch, mask: int, num_envs: int):
        self.batch = batch
        self.mask = mask
        self.n = num_envs
        self.rows: dict[int, list[list[int]]] = defaultdict(list)
        self.game_id = [0] * num_envs
        self.step_in_game = [0] * num_envs
        self.pre: dict[int, dict] = {}

    # -- game start, for batch.reset() and after an autoreset --
    def game_start(self, i: int) -> None:
        if not self.mask & ev.LOG_GAMES:
            return
        row = context_row(self.batch, i, self.game_id[i], 0, ev.GAME_START,
                          self.batch.actor_to_act(i), -1)
        seed = self.batch.game_seed(i)
        put(row, "seed_lo", np.int32(np.uint32(seed & 0xFFFFFFFF)))
        put(row, "seed_hi", np.int32(np.uint32(seed >> 32)))
        self.rows[i].append(row)

    def reset(self) -> None:
        for i in range(self.n):
            self.game_id[i] = 0
            self.step_in_game[i] = 0
            self.game_start(i)

    # -- before the batched step --
    def capture(self, i: int, action: int, legal: bool) -> None:
        """Read the pre-step context this env may need after the step.

        Only an action that will actually produce a pre-step row needs the
        full player context, so a plain build or roll costs two getters.
        """
        if action == NO_ACTION:
            self.pre.pop(i, None)
            return
        actor = self.batch.actor_to_act(i)
        pre = {
            "actor": actor,
            "step_in_game": self.step_in_game[i],
            "was_ended": self.batch.phase(i) == PHASE_ENDED,
            "legal": legal,
            "action": action,
            "row": None,
        }
        is_trade = legal and TRADE_FIRST <= action <= TRADE_LAST
        if self.mask & ev.LOG_ACTIONS or (self.mask & ev.LOG_TRADES and is_trade):
            pre["row"] = context_row(self.batch, i, self.game_id[i],
                                     self.step_in_game[i], ev.ACTION,
                                     actor, action)
        if self.mask & ev.LOG_TRADES and is_trade:
            pre["current_player"] = self.batch.current_player(i)
            pre["proposer"] = self.batch.trade_proposer(i)
            pre["give"] = [self.batch.trade_give(i, r) for r in range(RESOURCES)]
            pre["want"] = [self.batch.trade_want(i, r) for r in range(RESOURCES)]
        self.pre[i] = pre

    # -- after the batched step, before any autoreset is observed --
    def settle(self, i: int, done: bool, autoreset: bool) -> None:
        pre = self.pre.pop(i, None)
        if pre is not None:
            self.step_in_game[i] += 1
            if self.mask & ev.LOG_ACTIONS:
                self.rows[i].append(list(pre["row"]))
            if self.mask & ev.LOG_TRADES:
                self._trade(i, pre)
        if done:
            if (self.mask & ev.LOG_GAMES and pre is not None
                    and not pre["was_ended"]):
                self._game_end(i, pre)
            if autoreset:
                self.game_id[i] += 1
                self.step_in_game[i] = 0

    def _trade(self, i: int, pre: dict) -> None:
        action = pre["action"]
        if not pre["legal"] or not TRADE_FIRST <= action <= TRADE_LAST:
            return
        partner = -1
        if action == ACT.TRADE_OPEN:
            event_type = ev.TRADE_OPEN
        elif action == ACT.TRADE_CANCEL:
            event_type = ev.TRADE_CANCEL
        elif action >= ACT.TRADE_CONFIRM_BASE:
            event_type = ev.TRADE_CONFIRM
            partner = action - ACT.TRADE_CONFIRM_BASE
        else:
            registered = self.batch.trade_response(i, pre["actor"])
            event_type = (ev.TRADE_ACCEPT if registered == RESPONSE_ACCEPT
                          else ev.TRADE_DECLINE)
        row = list(pre["row"])
        put(row, "event_type", event_type)
        put(row, "proposer", pre["proposer"] if pre["proposer"] != NO_PLAYER
            else pre["current_player"])
        put(row, "partner", partner)
        put(row, "give", pre["give"])
        put(row, "want", pre["want"])
        self.rows[i].append(row)

    def _game_end(self, i: int, pre: dict) -> None:
        row = context_row(self.batch, i, self.game_id[i], pre["step_in_game"],
                          ev.GAME_END, pre["actor"], pre["action"])
        put(row, "winner", self.batch.last_winner(i))
        put(row, "turns", self.batch.turn_count(i))
        put(row, "steps", pre["step_in_game"] + 1)
        self.rows[i].append(row)


# --- harness ---------------------------------------------------------------


def drain_into(batch, sink: dict[int, list[list[int]]], capacity: int) -> int:
    """Drain every pending row, grouping by env in arrival order."""
    buffer = np.zeros((capacity, fastcatan.EVENT_INTS), dtype=np.int32)
    total = 0
    while True:
        count = batch.drain_events(buffer)
        if count == 0:
            return total
        rows = buffer[:count].tolist()
        envs = [row[F["env_idx"]] for row in rows]
        assert envs == sorted(envs), "a drain must be ordered by env_idx"
        for row in rows:
            sink[row[F["env_idx"]]].append(row)
        total += count


def compare(actual: dict[int, list[list[int]]],
            expected: dict[int, list[list[int]]]) -> int:
    assert set(actual) == set(expected), "env sets differ"
    compared = 0
    for i in sorted(expected):
        got, want = actual[i], expected[i]
        assert len(got) == len(want), (
            f"env {i}: {len(got)} rows recorded, {len(want)} expected\n"
            f"first divergence: {next((a, b) for a, b in zip(got, want) if a != b)}"
            if got != want else f"env {i}: row count mismatch")
        for index, (a, b) in enumerate(zip(got, want)):
            assert a == b, (
                f"env {i} row {index} differs\n"
                f"recorded: { {k: get(a, k) for k in fastcatan.EVENT_FIELDS} }\n"
                f"expected: { {k: get(b, k) for k in fastcatan.EVENT_FIELDS} }")
        compared += len(got)
    return compared


def unpack_masks(masks: np.ndarray) -> np.ndarray:
    """(n, MASK_WORDS) uint64 bitsets -> (n, 64*MASK_WORDS) uint8 flags."""
    return np.unpackbits(masks.view(np.uint8), axis=1, bitorder="little")


def trade_biased_action(legal: list[int], rng: random.Random) -> int:
    """Random legal action, biased so player trades actually happen."""
    for preferred in (ACT.TRADE_OPEN, ACT.TRADE_ACCEPT):
        if preferred in legal and rng.random() < 0.8:
            return preferred
    if rng.random() < 0.8:
        confirms = [a for a in legal
                    if ACT.TRADE_CONFIRM_BASE <= a
                    < ACT.TRADE_CONFIRM_BASE + PLAYERS]
        if confirms:
            return confirms[rng.randrange(len(confirms))]
    if rng.random() < 0.6:
        adds = [a for a in legal
                if ACT.TRADE_ADD_GIVE_BASE <= a < ACT.TRADE_OPEN]
        if adds:
            return adds[rng.randrange(len(adds))]
    return legal[rng.randrange(len(legal))]


def run_reference_comparison(mask: int, games: int, num_envs: int,
                             seed: int, drain_every: int,
                             capacity: int) -> dict:
    """Play `games` trade-heavy games and compare log against the reference.

    Uses `step` rather than `step_autoreset` so the reference logger can read
    the finished game's own final tally, which an autoreset would overwrite.
    Each wave of `num_envs` games is drained and compared before the next.
    """
    batch = fastcatan.BatchedEnv(num_envs, seed=seed)
    batch.enable_log(mask, capacity)
    rng = random.Random(seed)

    masks = np.zeros((num_envs, fastcatan.MASK_WORDS), dtype=np.uint64)
    actions = np.zeros(num_envs, dtype=np.uint32)
    dones = np.zeros(num_envs, dtype=np.uint8)
    stats = {"rows": 0, "games": 0, "illegal": 0, "types": defaultdict(int)}

    played = 0
    while played < games:
        batch.reset()
        reference = ReferenceLogger(batch, mask, num_envs)
        reference.reset()
        recorded: dict[int, list[list[int]]] = defaultdict(list)
        live = list(range(num_envs))
        step = 0

        while live:
            batch.write_masks(masks)
            flags = unpack_masks(masks)
            legal_sets = np.nonzero(flags)
            per_env = {}
            for env_idx, action_id in zip(legal_sets[0].tolist(),
                                          legal_sets[1].tolist()):
                per_env.setdefault(env_idx, []).append(action_id)
            actions[:] = NO_ACTION
            for i in live:
                legal = per_env[i]
                action = trade_biased_action(legal, rng)
                if rng.random() < 0.02:
                    # Mask-illegal actions are strict no-ops, but LOG_ACTIONS
                    # still records them; feed a few in so that rule is under
                    # test rather than assumed.
                    candidate = rng.randrange(fastcatan.NUM_ACTIONS)
                    if candidate not in legal:
                        action = candidate
                actions[i] = action
                is_legal = action in legal
                stats["illegal"] += not is_legal
                reference.capture(i, action, is_legal)
            batch.step(actions, dones)
            for i in live:
                reference.settle(i, bool(dones[i]), autoreset=False)
            live = [i for i in live if not dones[i]]

            step += 1
            if step % drain_every == 0:
                drain_into(batch, recorded, capacity * 4)

        drain_into(batch, recorded, capacity * 4)
        for i in range(num_envs):
            recorded.setdefault(i, [])
        stats["rows"] += compare(recorded, dict(reference.rows))
        for rows in recorded.values():
            for row in rows:
                stats["types"][get(row, "event_type")] += 1
        played += num_envs
        stats["games"] = played

    assert batch.log_overflow == 0, (
        f"{batch.log_overflow} rows dropped; drain interval or capacity too small")
    batch.disable_log()
    return stats


# --- tests -----------------------------------------------------------------


def test_event_layout_constants() -> None:
    assert fastcatan.EVENT_INTS == 36
    assert fastcatan.EVENT_VERSION == 1
    covered = sorted(fastcatan.EVENT_FIELDS.values())
    assert sum(length for _, length in covered) == fastcatan.EVENT_INTS
    cursor = 0
    for offset, length in covered:
        assert offset == cursor, "EVENT_FIELDS must tile the row with no gaps"
        cursor += length
    assert ev.LOG_ALL == ev.LOG_TRADES | ev.LOG_GAMES | ev.LOG_ACTIONS
    assert len({ev.GAME_START, ev.GAME_END, ev.ACTION, ev.TRADE_OPEN,
                ev.TRADE_ACCEPT, ev.TRADE_DECLINE, ev.TRADE_CONFIRM,
                ev.TRADE_CANCEL}) == 8


def test_log_is_disabled_by_default() -> None:
    batch = fastcatan.BatchedEnv(4)
    assert batch.log_mask == 0
    assert batch.pending_events() == 0
    assert batch.log_overflow == 0
    batch.reset()
    masks = np.zeros((4, fastcatan.MASK_WORDS), dtype=np.uint64)
    actions = np.zeros(4, dtype=np.uint32)
    dones = np.zeros(4, dtype=np.uint8)
    batch.write_masks(masks)
    for i in range(4):
        actions[i] = legal_actions(masks[i])[0]
    batch.step(actions, dones)
    assert batch.pending_events() == 0
    with pytest.raises(RuntimeError, match="not enabled"):
        batch.drain_events(
            np.zeros((1, fastcatan.EVENT_INTS), dtype=np.int32))


def test_enable_log_validates_arguments() -> None:
    batch = fastcatan.BatchedEnv(4)
    with pytest.raises(RuntimeError, match="log mask"):
        batch.enable_log(0, 16)
    with pytest.raises(RuntimeError, match="log mask"):
        batch.enable_log(ev.LOG_ALL + 1, 16)
    with pytest.raises(RuntimeError, match="capacity_per_env"):
        batch.enable_log(ev.LOG_GAMES, 0)
    batch.enable_log(ev.LOG_GAMES, 16)
    assert batch.log_mask == ev.LOG_GAMES
    assert batch.log_capacity_per_env == 16
    with pytest.raises(RuntimeError, match="event buffer shape"):
        batch.drain_events(
            np.zeros((4, fastcatan.EVENT_INTS - 1), dtype=np.int32))
    batch.disable_log()
    assert batch.log_mask == 0


def test_run_random_games_refuses_to_lose_events() -> None:
    batch = fastcatan.BatchedEnv(4, seed=1)
    batch.enable_log(ev.LOG_GAMES, 8)
    batch.reset()
    with pytest.raises(RuntimeError, match="disable_log"):
        batch.run_random_games(4)
    batch.disable_log()
    completed, _ = batch.run_random_games(4)
    assert completed == 4


def test_game_start_records_the_replay_seed() -> None:
    n = 8
    batch = fastcatan.BatchedEnv(n, seed=99)
    batch.enable_log(ev.LOG_GAMES, 4)
    batch.reset()
    assert batch.pending_events() == n

    buffer = np.zeros((n, fastcatan.EVENT_INTS), dtype=np.int32)
    assert batch.drain_events(buffer) == n
    assert batch.pending_events() == 0

    single = fastcatan.Env()
    for i in range(n):
        row = buffer[i]
        assert get(row, "event_type") == ev.GAME_START
        assert get(row, "env_idx") == i
        assert get(row, "game_id") == 0
        assert get(row, "step_in_game") == 0
        assert get(row, "turn") == 0
        assert get(row, "actor") == batch.actor_to_act(i)
        assert get(row, "action") == -1
        seed = ((get(row, "seed_hi") & 0xFFFFFFFF) << 32) \
            | (get(row, "seed_lo") & 0xFFFFFFFF)
        assert seed == batch.game_seed(i)
        # The recorded seed reproduces the episode exactly.
        single.reset(seed)
        assert single.snapshot() == batch.snapshot(i)


def test_autoreset_emits_game_end_then_start() -> None:
    n = 6
    batch = fastcatan.BatchedEnv(n, seed=4)
    batch.enable_log(ev.LOG_GAMES, 64)
    batch.reset()

    masks = np.zeros((n, fastcatan.MASK_WORDS), dtype=np.uint64)
    actions = np.zeros(n, dtype=np.uint32)
    dones = np.zeros(n, dtype=np.uint8)
    rng = random.Random(4)
    rows: dict[int, list[list[int]]] = defaultdict(list)

    completed = 0
    for _ in range(60_000):
        if completed >= 3 * n:
            break
        batch.write_masks(masks)
        for i in range(n):
            actions[i] = rng.choice(legal_actions(masks[i]))
        batch.step_autoreset(actions, dones)
        completed += int(dones.sum())
        drain_into(batch, rows, 4 * n * 64)
    drain_into(batch, rows, 4 * n * 64)
    assert batch.log_overflow == 0

    for i in range(n):
        stream = rows[i]
        assert get(stream[0], "event_type") == ev.GAME_START
        assert get(stream[0], "game_id") == 0
        # Every end is immediately followed by the next game's start, with the
        # episode counter advanced exactly once.
        for index in range(1, len(stream) - 1, 2):
            end, start = stream[index], stream[index + 1]
            assert get(end, "event_type") == ev.GAME_END
            assert get(start, "event_type") == ev.GAME_START
            assert get(start, "game_id") == get(end, "game_id") + 1
            assert get(end, "steps") == get(end, "step_in_game") + 1
            assert get(end, "turns") == get(end, "turn")
            winner = get(end, "winner")
            assert winner == NO_PLAYER or get(end, "true_vp")[winner] >= 10
            assert get(start, "step_in_game") == 0
            assert get(start, "turn") == 0
            assert get(start, "true_vp") == [0, 0, 0, 0]
        ends = sum(1 for row in stream if get(row, "event_type") == ev.GAME_END)
        assert ends >= 2, f"env {i} finished only {ends} games"


def test_drain_leaves_rows_that_do_not_fit() -> None:
    n = 4
    batch = fastcatan.BatchedEnv(n, seed=11)
    batch.enable_log(ev.LOG_GAMES, 4)
    batch.reset()
    assert batch.pending_events() == n

    small = np.zeros((1, fastcatan.EVENT_INTS), dtype=np.int32)
    seen = []
    for expected_left in range(n - 1, -1, -1):
        assert batch.drain_events(small) == 1
        seen.append(get(small[0], "env_idx"))
        assert batch.pending_events() == expected_left
    assert seen == list(range(n))
    assert batch.drain_events(small) == 0
    assert batch.log_overflow == 0


def test_overflow_is_counted_not_silent() -> None:
    n = 2
    batch = fastcatan.BatchedEnv(n, seed=3)
    batch.enable_log(ev.LOG_ACTIONS, 4)
    batch.reset()

    masks = np.zeros((n, fastcatan.MASK_WORDS), dtype=np.uint64)
    actions = np.zeros(n, dtype=np.uint32)
    dones = np.zeros(n, dtype=np.uint8)
    for _ in range(10):
        batch.write_masks(masks)
        for i in range(n):
            actions[i] = legal_actions(masks[i])[0]
        batch.step(actions, dones)

    assert batch.pending_events() == n * 4
    assert batch.log_overflow == n * 6
    # Draining frees the buffers again, and the counter stays monotonic.
    buffer = np.zeros((n * 4, fastcatan.EVENT_INTS), dtype=np.int32)
    assert batch.drain_events(buffer) == n * 4
    assert batch.log_overflow == n * 6
    batch.enable_log(ev.LOG_ACTIONS, 4)
    assert batch.log_overflow == 0


def test_log_ignores_snapshots() -> None:
    n = 4
    batch = fastcatan.BatchedEnv(n, seed=8)
    batch.enable_log(ev.LOG_ALL, 32)
    batch.reset()
    data = np.zeros((n, fastcatan.SNAPSHOT_BYTES), dtype=np.uint8)
    batch.save_snapshots(data)
    before = batch.pending_events()
    batch.load_snapshots(data)
    assert batch.pending_events() == before
    assert batch.log_mask == ev.LOG_ALL


def test_illegal_action_is_recorded_without_a_trade_event() -> None:
    """A mask-illegal action is a no-op, but LOG_ACTIONS still records it."""
    batch = fastcatan.BatchedEnv(1, seed=21)
    batch.enable_log(ev.LOG_ALL, 8)
    batch.reset()
    assert batch.drain_events(
        np.zeros((1, fastcatan.EVENT_INTS), dtype=np.int32)) == 1  # game start

    masks = np.zeros((1, fastcatan.MASK_WORDS), dtype=np.uint64)
    batch.write_masks(masks)
    legal = set(legal_actions(masks[0]))
    # Responding to a trade nobody opened is illegal in the opening phase.
    illegal = ACT.TRADE_ACCEPT
    assert illegal not in legal

    before = batch.snapshot(0)
    actor, step_before = batch.actor_to_act(0), batch.step_in_game(0)
    batch.step(np.array([illegal], dtype=np.uint32),
               np.zeros(1, dtype=np.uint8))

    assert batch.snapshot(0) == before, "an illegal action must not mutate state"
    assert batch.step_in_game(0) == step_before + 1

    buffer = np.zeros((4, fastcatan.EVENT_INTS), dtype=np.int32)
    assert batch.drain_events(buffer) == 1, "expected exactly one row"
    row = buffer[0]
    assert get(row, "event_type") == ev.ACTION
    assert get(row, "action") == illegal
    assert get(row, "actor") == actor
    assert get(row, "step_in_game") == step_before
    assert get(row, "proposer") == -1
    assert get(row, "give") == [0] * RESOURCES


@pytest.mark.slow
def test_trade_events_match_reference_over_1000_games() -> None:
    stats = run_reference_comparison(
        mask=ev.LOG_TRADES | ev.LOG_GAMES,
        games=1_024, num_envs=64, seed=17,
        drain_every=97, capacity=256)
    assert stats["games"] >= 1_000
    for event_type in (ev.TRADE_OPEN, ev.TRADE_ACCEPT, ev.TRADE_DECLINE,
                       ev.TRADE_CONFIRM, ev.TRADE_CANCEL, ev.GAME_START,
                       ev.GAME_END):
        assert stats["types"][event_type] > 0, f"event type {event_type} never fired"
    assert stats["rows"] > 10_000


def test_all_events_match_reference() -> None:
    stats = run_reference_comparison(
        mask=ev.LOG_ALL, games=32, num_envs=32, seed=23,
        drain_every=23, capacity=128)
    assert stats["types"][ev.ACTION] > 10_000
    assert stats["types"][ev.GAME_END] == 32
    assert stats["types"][ev.TRADE_CONFIRM] > 0
    # Every stepped action produced exactly one EVENT_ACTION row, no-ops
    # included.
    assert stats["illegal"] > 100
    assert stats["types"][ev.ACTION] == sum(
        count for event_type, count in stats["types"].items()
        if event_type == ev.ACTION)
