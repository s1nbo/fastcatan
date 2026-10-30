"""Observation encoder: field-by-field parity with state_view, and no leaks."""
from __future__ import annotations

import random

import numpy as np
import pytest

import fastcatan
from tests.conftest import legal_actions


NO_PLAYER = fastcatan.NO_PLAYER
RESOURCES = fastcatan.NUM_RESOURCES
PLAYERS = fastcatan.NUM_PLAYERS
DEV_KINDS = 5

FLAG_TRADE_PENDING = 7  # Flag::TRADE_PENDING
RESPONSE_ACCEPT = 1     # trade_response ACCEPT

# Normalization divisors, re-derived here from the documented structural
# maxima rather than imported, so a change to either side fails the test.
VP = 10.0
HAND = 25.0
DEV = 10.0
KNIGHTS = 10.0
ROADLEN = 15.0
SETTLE = 5.0
CITY = 4.0
ROAD = 15.0
DISCARD = 10.0
RES = 19.0
BANK = 19.0
DEVDECK = 25.0
HEXNUM = 12.0
TURN = 400.0
FREEROADS = 2.0
TRADE = 19.0
COMPOSE = 50.0


def obs_of(env: fastcatan.Env, pov: int) -> np.ndarray:
    out = np.zeros(fastcatan.OBS_SIZE, dtype=np.float32)
    env.write_obs(pov, out)
    return out


def field(obs: np.ndarray, name: str) -> np.ndarray:
    offset, length = fastcatan.OBS_FIELDS[name]
    return obs[offset:offset + length]


def one_hot(index: int | None, size: int) -> list[float]:
    """Expected one-hot; `None` (or out-of-range) means every slot stays 0."""
    return [1.0 if index is not None and i == index else 0.0 for i in range(size)]


def relseat(pov: int, seat: int) -> int:
    return (seat - pov) % PLAYERS


def seat_hot(pov: int, seat: int) -> list[float]:
    """5-slot [self, +1, +2, +3, none] seat encoding."""
    return one_hot(4 if seat == NO_PLAYER else relseat(pov, seat), 5)


def check_parity(env: fastcatan.Env, pov: int) -> None:
    """Assert every observation field matches state_view(pov)."""
    obs = obs_of(env, pov)
    view = env.state_view(pov)
    per = fastcatan.OBS_PER_PLAYER

    # --- per-player blocks, relative-seat order ---
    players = field(obs, "players")
    for rel in range(PLAYERS):
        seat = (pov + rel) % PLAYERS
        who = view["players"][seat]
        block = players[rel * per:(rel + 1) * per]
        expected = [
            (who["victory_points"] if rel == 0
             else who["public_victory_points"]) / VP,
            who["hand_size"] / HAND,
            who["development_card_count"] / DEV,
            who["knights_played"] / KNIGHTS,
            who["road_length"] / ROADLEN,
            who["settlements_remaining"] / SETTLE,
            who["cities_remaining"] / CITY,
            who["roads_remaining"] / ROAD,
            *[float((who["ports"] >> bit) & 1) for bit in range(6)],
            who["discard_remaining"] / DISCARD,
            float(view["current_player"] == seat),
            float(view["actor_to_act"] == seat),
        ]
        assert block == pytest.approx(expected), f"player block rel={rel}"

    # --- the pov's own hand ---
    own = view["players"][pov]
    assert field(obs, "self_private") == pytest.approx(
        [r / RES for r in own["resources"]]
        + [d / DEV for d in own["development_cards"]]
        + [d / DEV for d in own["development_cards_bought_this_turn"]]
    )

    # --- board ---
    nodes = field(obs, "nodes")
    for node in range(fastcatan.NUM_NODES):
        level = view["node_levels"][node]
        owner = view["node_owners"][node]
        expected = [0.0] * (2 * PLAYERS)
        if level != 0:
            expected[2 * relseat(pov, owner) + (level - 1)] = 1.0
        assert list(nodes[node * 2 * PLAYERS:(node + 1) * 2 * PLAYERS]) == expected

    edges = field(obs, "edges")
    for edge in range(fastcatan.NUM_EDGES):
        owner = view["edge_owners"][edge]
        assert list(edges[edge * PLAYERS:(edge + 1) * PLAYERS]) == one_hot(
            None if owner == NO_PLAYER else relseat(pov, owner), PLAYERS)

    hex_res = field(obs, "hex_resource")
    for hex_id in range(fastcatan.NUM_HEXES):
        assert list(hex_res[hex_id * 6:(hex_id + 1) * 6]) == one_hot(
            view["hex_resources"][hex_id], 6)
    assert field(obs, "hex_number") == pytest.approx(
        [n / HEXNUM for n in view["hex_numbers"]])

    ports = field(obs, "port_type")
    for port in range(fastcatan.NUM_PORTS):
        assert list(ports[port * 6:(port + 1) * 6]) == one_hot(
            view["port_types"][port], 6)
    assert list(field(obs, "robber")) == one_hot(
        view["robber_hex"], fastcatan.NUM_HEXES)

    # --- game state ---
    assert list(field(obs, "phase")) == one_hot(view["phase"], 4)
    assert list(field(obs, "flag")) == one_hot(view["flag"], 8)
    assert list(field(obs, "dice_roll")) == one_hot(view["dice_roll"], 13)
    assert field(obs, "turn_count") == pytest.approx([view["turn_count"] / TURN])
    assert field(obs, "bank") == pytest.approx([b / BANK for b in view["bank"]])
    assert field(obs, "dev_deck_size") == pytest.approx(
        [view["development_deck_size"] / DEVDECK])
    assert list(field(obs, "longest_road_owner")) == seat_hot(
        pov, view["longest_road_owner"])
    assert list(field(obs, "largest_army_owner")) == seat_hot(
        pov, view["largest_army_owner"])
    assert list(field(obs, "start_player")) == one_hot(
        relseat(pov, view["start_player"]), 4)
    assert field(obs, "free_roads") == pytest.approx(
        [view["free_roads_remaining"] / FREEROADS])
    assert field(obs, "dev_card_played") == pytest.approx(
        [float(view["dev_card_played"])])

    # The compose budget belongs to the turn owner alone.
    expected_compose = (env.trade_compose_count / COMPOSE
                        if pov == view["current_player"] else 0.0)
    assert field(obs, "trade_compose") == pytest.approx([expected_compose])

    # --- player-to-player trade ---
    trade = view["trade"]
    assert list(field(obs, "trade_proposer")) == seat_hot(pov, trade["proposer"])

    # state_view hides an unopened draft from everyone but its owner, and the
    # observation must hide exactly the same thing.
    if trade["give"] is None:
        assert not field(obs, "trade_give").any()
        assert not field(obs, "trade_want").any()
    else:
        assert field(obs, "trade_give") == pytest.approx(
            [g / TRADE for g in trade["give"]])
        assert field(obs, "trade_want") == pytest.approx(
            [w / TRADE for w in trade["want"]])

    responses = field(obs, "trade_response")
    if trade["responses"] is None:
        assert not responses.any()
    else:
        for rel in range(PLAYERS):
            seat = (pov + rel) % PLAYERS
            assert list(responses[rel * 4:(rel + 1) * 4]) == one_hot(
                trade["responses"][seat], 4)


def test_obs_shape_constants() -> None:
    assert fastcatan.OBS_SIZE == 1089
    assert fastcatan.OBS_VERSION == 1
    covered = sorted(
        (offset, length) for offset, length in fastcatan.OBS_FIELDS.values())
    assert sum(length for _, length in covered) == fastcatan.OBS_SIZE
    cursor = 0
    for offset, length in covered:
        assert offset == cursor, "OBS_FIELDS must tile the vector with no gaps"
        cursor += length


def test_obs_buffer_shapes_are_checked() -> None:
    env = fastcatan.Env()
    with pytest.raises(RuntimeError, match="not reset"):
        env.write_obs(0, np.zeros(fastcatan.OBS_SIZE, dtype=np.float32))
    env.reset(0)
    with pytest.raises(RuntimeError, match="pov"):
        env.write_obs(PLAYERS, np.zeros(fastcatan.OBS_SIZE, dtype=np.float32))
    with pytest.raises(RuntimeError, match="obs buffer"):
        env.write_obs(0, np.zeros(fastcatan.OBS_SIZE - 1, dtype=np.float32))

    batch = fastcatan.BatchedEnv(2)
    batch.reset()
    with pytest.raises(RuntimeError, match="obs buffer"):
        batch.write_obs(np.zeros((1, fastcatan.OBS_SIZE), dtype=np.float32))
    with pytest.raises(RuntimeError, match="seats length"):
        batch.write_obs_pov(
            np.zeros(1, dtype=np.uint8),
            np.zeros((2, fastcatan.OBS_SIZE), dtype=np.float32))
    with pytest.raises(RuntimeError, match="seat"):
        batch.write_obs_pov(
            np.full(2, PLAYERS, dtype=np.uint8),
            np.zeros((2, fastcatan.OBS_SIZE), dtype=np.float32))


@pytest.mark.slow
def test_obs_matches_state_view_over_random_games() -> None:
    """Every field of every pov matches state_view across >=1000 games."""
    games = 1000
    checked = 0
    for seed in range(games):
        rng = random.Random(seed)
        env = fastcatan.Env()
        env.reset(seed)
        # One randomly chosen step per game, so the sample spans openings,
        # midgames, trades and terminal states across the 1000 games.
        for _ in range(rng.randrange(1, 2_000)):
            if env.done:
                break
            env.step(rng.choice(env.legal_actions()))
        for pov in range(PLAYERS):
            check_parity(env, pov)
            checked += 1
    assert checked == games * PLAYERS


def test_obs_matches_state_view_during_open_trade() -> None:
    """Cover the trade branches explicitly, not just by chance."""
    probe_budget = 150
    seen_open = seen_accept = seen_draft = 0
    for seed in range(60):
        rng = random.Random(seed)
        env = fastcatan.Env()
        env.reset(seed)
        for _ in range(3_000):
            if env.done:
                break
            legal = env.legal_actions()
            # Bias the walk towards composing, opening and accepting trades.
            for preferred in (fastcatan.action.TRADE_OPEN,
                              fastcatan.action.TRADE_ACCEPT):
                if preferred in legal:
                    legal = [preferred]
                    break
            else:
                adds = [a for a in legal
                        if fastcatan.action.TRADE_ADD_GIVE_BASE <= a
                        < fastcatan.action.TRADE_OPEN]
                if adds and rng.random() < 0.7:
                    legal = adds
            env.step(rng.choice(legal))

            drafting = env.trade_proposer == NO_PLAYER and any(
                env.trade_give(r) or env.trade_want(r) for r in range(RESOURCES))
            if env.flag == FLAG_TRADE_PENDING:
                seen_open += 1
                if any(env.trade_response(p) == RESPONSE_ACCEPT
                       for p in range(PLAYERS)):
                    seen_accept += 1
            elif drafting:
                seen_draft += 1
            else:
                continue
            for pov in range(PLAYERS):
                check_parity(env, pov)
            probe_budget -= 1
            if probe_budget <= 0 and seen_open and seen_accept and seen_draft:
                break
        if probe_budget <= 0 and seen_open and seen_accept and seen_draft:
            break
    assert seen_open, "never reached an open proposal"
    assert seen_accept, "never reached an accepted proposal"
    assert seen_draft, "never reached an unopened draft"


def test_batched_obs_matches_single_env() -> None:
    n = 16
    batch = fastcatan.BatchedEnv(n, seed=5)
    batch.reset()
    masks = np.zeros((n, fastcatan.MASK_WORDS), dtype=np.uint64)
    actions = np.zeros(n, dtype=np.uint32)
    dones = np.zeros(n, dtype=np.uint8)
    rng = random.Random(5)

    batched = np.zeros((n, fastcatan.OBS_SIZE), dtype=np.float32)
    seats = np.zeros(n, dtype=np.uint8)
    actors = np.zeros(n, dtype=np.uint8)
    env = fastcatan.Env()

    for _ in range(80):
        batch.write_masks(masks)
        for i in range(n):
            actions[i] = rng.choice(legal_actions(masks[i]))
        batch.step(actions, dones)

        batch.write_actors(actors)
        batch.write_obs(batched)
        for i in range(n):
            env.load_snapshot(batch.snapshot(i))
            assert actors[i] == env.actor_to_act
            np.testing.assert_array_equal(batched[i], obs_of(env, actors[i]))

        for i in range(n):
            seats[i] = rng.randrange(PLAYERS)
        batch.write_obs_pov(seats, batched)
        for i in range(n):
            env.load_snapshot(batch.snapshot(i))
            np.testing.assert_array_equal(batched[i], obs_of(env, int(seats[i])))


# --- leakage ---------------------------------------------------------------
#
# Hidden opponent state is reached through snapshot surgery, which means the
# test needs to know where two private blocks sit inside the snapshot. Rather
# than hard-code struct offsets, locate each block by its contents and then
# prove the patch landed by reading it back through the public getters.


def _unique_offset(blob: bytes, pattern: bytes) -> int | None:
    """Offset of `pattern` in `blob`, or None unless it occurs exactly once."""
    first = blob.find(pattern)
    if first < 0 or blob.find(pattern, first + 1) >= 0:
        return None
    return first


def _locate_private_blocks(env: fastcatan.Env, blob: bytes) -> tuple[int | None, int | None]:
    """Byte offsets of player_resources and player_dev inside a snapshot."""
    resources = bytes(
        env.player_resource(seat, r)
        for seat in range(PLAYERS) for r in range(RESOURCES))
    cards = bytes(
        env.player_dev(seat, c)
        for seat in range(PLAYERS) for c in range(DEV_KINDS))
    # An all-zero block is not locatable; those games are simply skipped.
    res_at = _unique_offset(blob, resources) if any(resources) else None
    dev_at = _unique_offset(blob, cards) if any(cards) else None
    return res_at, dev_at


def _exchange(state: bytearray, base: int, width: int, first: int, second: int,
              rng: random.Random) -> bool:
    """Swap one unit of two different kinds between two seats' blocks.

    Both seats keep their total, so every public count derived from these
    blocks (hand size, development-card count, the bank, public victory
    points) is untouched. What moves is the composition, which is exactly the
    information the observation must not expose.
    """
    held_first = [k for k in range(width) if state[base + first * width + k]]
    held_second = [k for k in range(width) if state[base + second * width + k]]
    pairs = [(x, y) for x in held_first for y in held_second if x != y]
    if not pairs:
        return False
    give, take = rng.choice(pairs)
    state[base + first * width + give] -= 1
    state[base + second * width + give] += 1
    state[base + second * width + take] -= 1
    state[base + first * width + take] += 1
    return True


@pytest.mark.slow
def test_obs_hides_opponent_private_state() -> None:
    """Randomizing hidden opponent holdings must not move a single float."""
    probed = 0
    for seed in range(400):
        rng = random.Random(seed)
        env = fastcatan.Env()
        env.reset(seed)
        for _ in range(rng.randrange(40, 600)):
            if env.done:
                break
            env.step(rng.choice(env.legal_actions()))

        baseline = env.snapshot()
        res_at, dev_at = _locate_private_blocks(env, baseline)
        if res_at is None and dev_at is None:
            continue

        for pov in range(PLAYERS):
            opponents = [seat for seat in range(PLAYERS) if seat != pov]
            # One pair per probe. Chaining exchanges across all three pairs
            # can cycle back to the original holdings, which would quietly
            # turn the probe into a no-op.
            first, second = rng.sample(opponents, 2)
            mutated = bytearray(baseline)
            changed = False
            if res_at is not None:
                changed |= _exchange(
                    mutated, res_at, RESOURCES, first, second, rng)
            if dev_at is not None:
                changed |= _exchange(
                    mutated, dev_at, DEV_KINDS, first, second, rng)
            if not changed:
                continue

            other = fastcatan.Env()
            other.load_snapshot(bytes(mutated))

            # Prove the surgery hit the intended blocks: hidden holdings moved
            # while every public counter stayed exactly where it was.
            moved = any(
                other.player_resource(seat, r) != env.player_resource(seat, r)
                for seat in opponents for r in range(RESOURCES)
            ) or any(
                other.player_dev(seat, c) != env.player_dev(seat, c)
                for seat in opponents for c in range(DEV_KINDS)
            )
            assert moved, "snapshot patch did not change any hidden holding"
            for seat in range(PLAYERS):
                assert other.player_handsize(seat) == env.player_handsize(seat)
                assert other.player_total_dev(seat) == env.player_total_dev(seat)
                assert other.player_vp_public(seat) == env.player_vp_public(seat)
                assert other.player_resource(pov, 0) == env.player_resource(pov, 0)
            for r in range(RESOURCES):
                assert other.bank(r) == env.bank(r)
                assert other.player_resource(pov, r) == env.player_resource(pov, r)
            for c in range(DEV_KINDS):
                assert other.player_dev(pov, c) == env.player_dev(pov, c)

            np.testing.assert_array_equal(
                obs_of(other, pov), obs_of(env, pov),
                err_msg=f"observation leaked hidden state (seed={seed}, pov={pov})")
            probed += 1
    assert probed > 500, f"only {probed} leakage probes ran"


def test_obs_hides_hidden_victory_points() -> None:
    """A victory-point card swapped between opponents must stay invisible."""
    env = fastcatan.Env()
    found = None
    for seed in range(2_000):
        rng = random.Random(seed)
        env.reset(seed)
        while not env.done:
            legal = env.legal_actions()
            action = (fastcatan.action.BUY_DEV
                      if fastcatan.action.BUY_DEV in legal
                      else rng.choice(legal))
            env.step(action)
            holders = [
                seat for seat in range(PLAYERS)
                if env.player_dev(seat, 1) > 0  # unplayed victory-point card
            ]
            others = [
                seat for seat in range(PLAYERS)
                if seat not in holders
                and any(env.player_dev(seat, c) > 0 for c in (0, 2, 3, 4))
            ]
            if holders and others:
                found = (holders[0], others[0])
                break
        if found:
            break
    assert found, "no state with a hidden victory-point card to move"
    holder, other_seat = found
    pov = next(seat for seat in range(PLAYERS)
               if seat not in (holder, other_seat))

    blob = env.snapshot()
    _, dev_at = _locate_private_blocks(env, blob)
    assert dev_at is not None
    swap_kind = next(c for c in (0, 2, 3, 4) if env.player_dev(other_seat, c) > 0)

    mutated = bytearray(blob)
    mutated[dev_at + holder * DEV_KINDS + 1] -= 1
    mutated[dev_at + other_seat * DEV_KINDS + 1] += 1
    mutated[dev_at + other_seat * DEV_KINDS + swap_kind] -= 1
    mutated[dev_at + holder * DEV_KINDS + swap_kind] += 1

    moved = fastcatan.Env()
    moved.load_snapshot(bytes(mutated))
    assert moved.player_dev(other_seat, 1) == env.player_dev(other_seat, 1) + 1
    assert moved.player_total_dev(holder) == env.player_total_dev(holder)
    assert moved.player_vp_public(holder) == env.player_vp_public(holder)
    np.testing.assert_array_equal(obs_of(moved, pov), obs_of(env, pov))
