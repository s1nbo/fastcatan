"""Env.determinize: resample what one seat cannot see, keep everything it can.

Checks, over states reached by random play:
  * conservation: 19 of each resource (95 total) and every development type's
    count across deck + hands; hand sizes, held and bought-this-turn counts;
  * the pov's observation and semantic view are bit-identical afterwards (so
    the observation does not already leak what was resampled);
  * determinism in the seed, and fresh future chance events;
  * uniform marginals against the exact hypergeometric laws;
  * open-trade bundles stay with the seats that publicly committed them;
  * no opponent is dealt into a win.
"""
from __future__ import annotations

import itertools
import math
import random
from collections import Counter

import numpy as np
import pytest

import fastcatan
from tests.conftest import find_state

PLAYERS = fastcatan.NUM_PLAYERS
RES = fastcatan.NUM_RESOURCES
DEV = 5
VP_CARD = 1
WIN_VP = 10
FLAG_NONE = 0
FLAG_TRADE_PENDING = 7
PHASE_MAIN = 2
TR_ACCEPT = 1
ACT = fastcatan.action


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def scratch(snap: bytes) -> fastcatan.Env:
    env = fastcatan.Env()
    env.load_snapshot(snap)
    return env


def determinized(snap: bytes, pov: int, seed: int) -> fastcatan.Env:
    env = scratch(snap)
    env.determinize(pov, seed)
    return env


def opponents(pov: int) -> list[int]:
    return [seat for seat in range(PLAYERS) if seat != pov]


def held(env, seat: int) -> int:
    return sum(env.player_dev(seat, d) for d in range(DEV))


def pending(env, seat: int) -> int:
    return sum(env.player_dev_bought(seat, d) for d in range(DEV))


def hidden(env, pov: int) -> tuple:
    opp = opponents(pov)
    return (
        tuple(env.player_resource(o, r) for o in opp for r in range(RES)),
        tuple(env.player_dev(o, d) for o in opp for d in range(DEV)),
        tuple(env.player_dev_bought(o, d) for o in opp for d in range(DEV)),
        tuple(env.dev_deck(d) for d in range(DEV)),
    )


def dev_type_totals(env) -> list[int]:
    return [
        env.dev_deck(d) + sum(
            env.player_dev(p, d) + env.player_dev_bought(p, d)
            for p in range(PLAYERS))
        for d in range(DEV)
    ]


def resource_pool(env, pov: int) -> list[int]:
    return [sum(env.player_resource(o, r) for o in opponents(pov))
            for r in range(RES)]


def dev_pool(env, pov: int) -> list[int]:
    return [env.dev_deck(d) + sum(
                env.player_dev(o, d) + env.player_dev_bought(o, d)
                for o in opponents(pov))
            for d in range(DEV)]


def obs_bytes(env, pov: int) -> bytes:
    out = np.zeros(fastcatan.OBS_SIZE, dtype=np.float32)
    env.write_obs(pov, out)
    return out.tobytes()


_STATES: list[bytes] | None = None


def sample_states() -> list[bytes]:
    """Snapshots spread over random games, dev-card heavy, ended ones too."""
    global _STATES
    if _STATES is None:
        states = []
        for game in range(30):
            seed = 9_000 + game
            rng = random.Random(seed)
            env = fastcatan.Env()
            env.reset(seed)
            step = 0
            while not env.done:
                legal = env.legal_actions()
                if ACT.BUY_DEV in legal and rng.random() < 0.7:
                    env.step(ACT.BUY_DEV)
                else:
                    env.step(rng.choice(legal))
                step += 1
                if step % 89 == 0:
                    states.append(env.snapshot())
            states.append(env.snapshot())
        _STATES = states
    return _STATES


def binomial(n: int, k: int) -> int:
    return math.comb(n, k) if 0 <= k <= n else 0


def chi2_z(observed: Counter, expected: dict, samples: int) -> float:
    """Wilson-Hilferty z-score of a chi-square goodness-of-fit statistic.

    Cells with an expected count below 5 are pooled into one.
    """
    stat, df = 0.0, -1
    rest_obs = rest_exp = 0.0
    for cell, p in expected.items():
        e = p * samples
        o = observed.get(cell, 0)
        if e < 5:
            rest_obs += o
            rest_exp += e
            continue
        stat += (o - e) ** 2 / e
        df += 1
    if rest_exp > 0:
        stat += (rest_obs - rest_exp) ** 2 / rest_exp
        df += 1
    assert df >= 1, "not enough cells for a goodness-of-fit test"
    a = 2.0 / (9.0 * df)
    return ((stat / df) ** (1.0 / 3.0) - (1.0 - a)) / math.sqrt(a)


def assert_hypergeometric_means(samples, n, pool, label):
    """samples[i][k]: count of kind k among n cards drawn from `pool`."""
    total = sum(pool)
    count = len(samples)
    for k, kind in enumerate(pool):
        mean = n * kind / total
        var = n * (kind / total) * (1 - kind / total) * (total - n) / max(total - 1, 1)
        got = sum(s[k] for s in samples) / count
        tol = 5.0 * math.sqrt(var / count) + 1e-9
        assert abs(got - mean) <= tol, (
            f"{label}: kind {k} mean {got:.4f}, expected {mean:.4f} +- {tol:.4f}")


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

def test_refuses_the_live_game() -> None:
    live = fastcatan.Env()
    live.reset(3)
    with pytest.raises(RuntimeError, match="scratch"):
        live.determinize(0, 1)

    copy = scratch(live.snapshot())
    with pytest.raises(RuntimeError, match="pov"):
        copy.determinize(PLAYERS, 1)
    copy.determinize(0, 1)

    copy.reset(4)  # a fresh game is a live game again
    with pytest.raises(RuntimeError, match="scratch"):
        copy.determinize(0, 1)

    with pytest.raises(RuntimeError):
        fastcatan.Env().determinize(0, 1)


# --------------------------------------------------------------------------
# conservation and pov information
# --------------------------------------------------------------------------

def test_sample_covers_the_interesting_phases() -> None:
    flags = Counter(scratch(snap).flag for snap in sample_states())
    phases = Counter(scratch(snap).phase for snap in sample_states())
    assert flags[FLAG_TRADE_PENDING] > 0 and flags[1] > 0  # trade, discard
    assert phases[PHASE_MAIN] > 100 and phases[3] >= 30


def test_conserves_counts_and_pov_information() -> None:
    checked = changed = 0
    for index, snap in enumerate(sample_states()):
        before = scratch(snap)
        for pov in range(PLAYERS):
            obs = obs_bytes(before, pov)
            view = before.state_view(pov)
            for seed in (index, 10_000 + index):
                after = determinized(snap, pov, seed)

                for r in range(RES):
                    assert after.bank(r) + sum(
                        after.player_resource(p, r) for p in range(PLAYERS)) == 19
                assert sum(after.bank(r) for r in range(RES)) + sum(
                    after.player_handsize(p) for p in range(PLAYERS)) == 95
                assert dev_type_totals(after) == dev_type_totals(before)
                assert sum(dev_type_totals(after)) + sum(
                    after.player_knights_played(p) for p in range(PLAYERS)) <= 25

                for p in range(PLAYERS):
                    assert after.player_handsize(p) == before.player_handsize(p)
                    assert sum(after.player_resource(p, r) for r in range(RES)) \
                        == after.player_handsize(p)
                    assert after.player_total_dev(p) == before.player_total_dev(p)
                    assert held(after, p) == held(before, p)
                    assert pending(after, p) == pending(before, p)
                    assert after.player_dev_bought(p, VP_CARD) == 0
                    assert after.player_vp_public(p) == before.player_vp_public(p)
                    assert after.player_vp(p) == (
                        after.player_vp_public(p) + after.player_dev(p, VP_CARD))
                for r in range(RES):
                    assert after.player_resource(pov, r) == before.player_resource(pov, r)
                for d in range(DEV):
                    assert after.player_dev(pov, d) == before.player_dev(pov, d)
                    assert after.player_dev_bought(pov, d) == before.player_dev_bought(pov, d)

                assert obs_bytes(after, pov) == obs, (
                    f"observation moved (state {index}, pov {pov}, seed {seed})")
                assert after.state_view(pov) == view, (
                    f"state_view moved (state {index}, pov {pov}, seed {seed})")
                scratch(after.snapshot())  # still a valid, loadable state

                checked += 1
                changed += hidden(after, pov) != hidden(before, pov)
    assert checked > 3_000
    assert changed > checked // 2, "determinize rarely changed anything"


def test_determinized_games_play_to_the_end() -> None:
    rng = random.Random(5)
    played = 0
    for index, snap in enumerate(sample_states()[::7]):
        env = determinized(snap, rng.randrange(PLAYERS), index)
        if env.done:
            continue
        for _ in range(200_000):
            legal = env.legal_actions()
            assert legal
            if env.step(rng.choice(legal)):
                break
        assert env.done
        scratch(env.snapshot())
        played += 1
    assert played > 50


def test_finished_game_only_reseeds() -> None:
    for snap in sample_states():
        before = scratch(snap)
        if not before.done:
            continue
        for pov in range(PLAYERS):
            after = determinized(snap, pov, 77)
            assert hidden(after, pov) == hidden(before, pov)
            assert after.winner == before.winner
            assert after.truncated == before.truncated
            assert after.snapshot() != snap  # the RNG moved


# --------------------------------------------------------------------------
# seeds and the RNG
# --------------------------------------------------------------------------

def test_same_seed_same_state_different_seed_different_state() -> None:
    varied = 0
    for index, snap in enumerate(sample_states()[::4]):
        env = scratch(snap)
        if env.done:
            continue
        pov = index % PLAYERS
        assert determinized(snap, pov, 42).snapshot() == \
            determinized(snap, pov, 42).snapshot()

        runs = [determinized(snap, pov, seed) for seed in range(20)]
        assert len({run.snapshot() for run in runs}) == 20  # RNG differs
        if len({hidden(run, pov) for run in runs}) > 1:
            varied += 1
    assert varied > 60


def test_future_dice_are_not_the_real_ones() -> None:
    env = find_state(lambda e, legal: e.phase == PHASE_MAIN
                     and e.flag == FLAG_NONE and ACT.ROLL_DICE in legal
                     and e.turn_count > 4)
    snap = env.snapshot()
    pov = env.current_player

    # A plain snapshot copy knows the real roll: that is the leak.
    copies = []
    for _ in range(5):
        copy = scratch(snap)
        copy.step(ACT.ROLL_DICE)
        copies.append(copy.dice_roll)
    assert len(set(copies)) == 1

    samples = 7_200
    rolls = Counter()
    for seed in range(samples):
        copy = determinized(snap, pov, seed)
        copy.step(ACT.ROLL_DICE)
        rolls[copy.dice_roll] += 1
    expected = {total: (6 - abs(7 - total)) / 36 for total in range(2, 13)}
    assert set(rolls) == set(expected)
    assert chi2_z(rolls, expected, samples) < 4.5
    assert rolls[copies[0]] / samples < 0.25


# --------------------------------------------------------------------------
# uniformity
# --------------------------------------------------------------------------

SAMPLES = 10_000


def _resource_state():
    for snap in sample_states():
        env = scratch(snap)
        if env.done or env.flag == FLAG_TRADE_PENDING:
            continue
        for pov in range(PLAYERS):
            sizes = [env.player_handsize(o) for o in opponents(pov)]
            pool = resource_pool(env, pov)
            if min(sizes) >= 2 and sum(sizes) >= 9 and sum(k > 0 for k in pool) >= 4:
                return snap, pov
    raise AssertionError("no state with three loaded opponent hands")


def test_resource_marginals_are_hypergeometric() -> None:
    snap, pov = _resource_state()
    env = scratch(snap)
    pool = resource_pool(env, pov)
    total = sum(pool)
    hands = {o: [] for o in opponents(pov)}
    for seed in range(SAMPLES):
        after = determinized(snap, pov, seed)
        for o in hands:
            hands[o].append(tuple(after.player_resource(o, r) for r in range(RES)))

    for o, draws in hands.items():
        n = env.player_handsize(o)
        assert_hypergeometric_means(draws, n, pool, f"seat {o} resources")

        # Full hand composition against the multivariate hypergeometric pmf.
        expected = {}
        for hand in itertools.product(*(range(k + 1) for k in pool)):
            if sum(hand) != n:
                continue
            expected[hand] = math.prod(
                binomial(k, h) for k, h in zip(pool, hand)) / binomial(total, n)
        assert abs(sum(expected.values()) - 1) < 1e-9
        z = chi2_z(Counter(draws), expected, SAMPLES)
        assert z < 4.5, f"seat {o} hand composition is not uniform (z={z:.2f})"


def _dev_state():
    for snap in sample_states():
        env = scratch(snap)
        if env.done:
            continue
        for pov in range(PLAYERS):
            opp = opponents(pov)
            pool = dev_pool(env, pov)
            if any(pending(env, o) for o in opp):
                continue
            if sum(held(env, o) for o in opp) < 4 or sum(k > 0 for k in pool) < 4:
                continue
            # No deal may reach WIN_VP, so rejection never bites.
            if all(env.player_vp_public(o) + min(held(env, o), pool[VP_CARD]) < WIN_VP
                   for o in opp):
                return snap, pov
    raise AssertionError("no state with spread development cards")


def test_dev_marginals_are_hypergeometric() -> None:
    snap, pov = _dev_state()
    env = scratch(snap)
    pool = dev_pool(env, pov)
    hands = {o: [] for o in opponents(pov)}
    decks = []
    for seed in range(SAMPLES):
        after = determinized(snap, pov, seed)
        for o in hands:
            hands[o].append(tuple(after.player_dev(o, d) for d in range(DEV)))
        decks.append(tuple(after.dev_deck(d) for d in range(DEV)))

    for o, draws in hands.items():
        assert_hypergeometric_means(draws, held(env, o), pool, f"seat {o} dev")
    deck_size = sum(env.dev_deck(d) for d in range(DEV))
    assert_hypergeometric_means(decks, deck_size, pool, "deck")


# --------------------------------------------------------------------------
# public commitments and plausibility
# --------------------------------------------------------------------------

def test_open_trade_bundles_stay_with_their_owners() -> None:
    env = find_state(
        lambda e, legal: e.flag == FLAG_TRADE_PENDING
        and e.actor_to_act == e.trade_proposer
        and any(e.trade_response(p) == TR_ACCEPT for p in range(PLAYERS)),
        max_seeds=200)
    snap = env.snapshot()
    proposer = env.trade_proposer
    give = [env.trade_give(r) for r in range(RES)]
    want = [env.trade_want(r) for r in range(RES)]
    accepters = [p for p in range(PLAYERS) if env.trade_response(p) == TR_ACCEPT]

    for pov in range(PLAYERS):
        for seed in range(300):
            after = determinized(snap, pov, seed)
            assert all(after.player_resource(proposer, r) >= give[r] for r in range(RES))
            for p in accepters:
                assert all(after.player_resource(p, r) >= want[r] for r in range(RES))
            # Every offered CONFIRM still resolves the trade instead of
            # silently doing nothing.
            for p in accepters:
                confirm = ACT.TRADE_CONFIRM_BASE + p
                assert confirm in after.legal_actions()
                done = scratch(after.snapshot())
                done.step(confirm)
                assert done.flag == FLAG_NONE


def test_unopened_draft_is_private_to_its_owner() -> None:
    env = find_state(
        lambda e, legal: e.trade_proposer == fastcatan.NO_PLAYER
        and any(e.trade_give(r) or e.trade_want(r) for r in range(RES)),
        max_seeds=50)
    snap = env.snapshot()
    owner = env.current_player
    draft = [(env.trade_give(r), env.trade_want(r)) for r in range(RES)]
    for pov in range(PLAYERS):
        after = determinized(snap, pov, 1)
        got = [(after.trade_give(r), after.trade_want(r)) for r in range(RES)]
        assert got == (draft if pov == owner else [(0, 0)] * RES)


def test_no_opponent_is_dealt_into_a_win() -> None:
    def risky(e, legal):
        if e.phase != PHASE_MAIN:
            return False
        for pov in range(PLAYERS):
            pool_vp = dev_pool(e, pov)[VP_CARD]
            for o in opponents(pov):
                short = WIN_VP - e.player_vp_public(o)
                if 0 < short <= min(held(e, o), pool_vp):
                    return True
        return False

    env = find_state(risky, max_seeds=200)
    snap = env.snapshot()
    hits = 0
    for pov in range(PLAYERS):
        pool_vp = dev_pool(env, pov)[VP_CARD]
        exposed = [o for o in opponents(pov)
                   if 0 < WIN_VP - env.player_vp_public(o) <= min(held(env, o), pool_vp)]
        if not exposed:
            continue
        hits += 1
        for seed in range(2_000):
            after = determinized(snap, pov, seed)
            for o in opponents(pov):
                if after.player_vp_public(o) < WIN_VP:
                    assert after.player_vp(o) < WIN_VP, (pov, o, seed)
            assert not after.done
    assert hits
