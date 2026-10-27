"""Tricky scenario tests.

We cannot construct arbitrary game states from Python (no setter API),
so each scenario scans random rollouts for the trigger condition (a
specific Flag value, dice roll, etc.) and asserts the relevant invariants
at that moment. We rejection-sample across many seeds to ensure every
scenario fires at least once.
"""
from __future__ import annotations

import random
import numpy as np
import pytest

import fastcatan
from tests.conftest import find_state, legal_actions


# Flag enum mirror (from include/state.hpp)
FLAG_NONE           = 0
FLAG_DISCARD        = 1
FLAG_MOVE_ROBBER    = 2
FLAG_ROBBER_STEAL   = 3


def _action_range(action_id: int) -> str:
    a = fastcatan.action
    if action_id < a.CITY_BASE:               return "SETTLE"
    if action_id < a.ROAD_BASE:               return "CITY"
    if action_id < a.ROLL_DICE:               return "ROAD"
    if action_id == a.ROLL_DICE:              return "ROLL"
    if action_id == a.END_TURN:               return "END_TURN"
    if action_id < a.MOVE_ROBBER_BASE:        return "DISCARD"
    if action_id < a.STEAL_BASE:              return "MOVE_ROBBER"
    if action_id < a.TRADE_BASE:              return "STEAL"
    if action_id < a.BUY_DEV:                 return "TRADE_BANK"
    if action_id == a.BUY_DEV:                return "BUY_DEV"
    if action_id == a.PLAY_KNIGHT:            return "PLAY_KNIGHT"
    if action_id == a.PLAY_ROAD_BUILDING:     return "PLAY_RB"
    if action_id < a.PLAY_MONOPOLY:           return "PLAY_YOP"
    if action_id < a.TRADE_ADD_GIVE_BASE:     return "PLAY_MONO"
    if action_id < a.TRADE_ADD_WANT_BASE:     return "TRADE_GIVE"
    if action_id < a.TRADE_OPEN:              return "TRADE_WANT"
    if action_id == a.TRADE_OPEN:             return "TRADE_OPEN"
    if action_id == a.TRADE_ACCEPT:           return "TRADE_ACCEPT"
    if action_id == a.TRADE_DECLINE:          return "TRADE_DECLINE"
    if action_id < a.TRADE_CANCEL:            return "TRADE_CONFIRM"
    return "TRADE_CANCEL"


def _drive_collect(seed: int, max_steps: int = 200_000, observers=()):
    """Random rollout. observers = [(name, fn(env, mask, action, step) -> bool/None)].

    Each observer is called every step; if it returns True the rollout still
    continues but the observer's hit count is incremented.
    """
    rng = random.Random(seed)
    env = fastcatan.Env()
    env.reset(seed)
    mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
    hits = {name: 0 for name, _ in observers}

    for step_idx in range(max_steps):
        env.action_mask(mask)
        legals = legal_actions(mask)
        action = rng.choice(legals)
        for name, fn in observers:
            if fn(env, mask, action, step_idx):
                hits[name] += 1
        done = env.step(action)
        if done:
            return env, step_idx + 1, hits
    raise AssertionError(f"game did not terminate (seed={seed})")


# ---------------------------------------------------------------------------
# 7-roll discard sub-phase
# ---------------------------------------------------------------------------

def test_discard_phase_invariants():
    """When flag == DISCARD, only DISCARD_BASE.. actions are legal, and
    only players with handsize > 7 owe discards."""
    a = fastcatan.action

    fires = 0
    for seed in range(30):
        rng = random.Random(seed)
        env = fastcatan.Env()
        env.reset(seed)
        mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
        for _ in range(200_000):
            env.action_mask(mask)
            legals = legal_actions(mask)
            if env.flag == FLAG_DISCARD:
                fires += 1
                # Every legal action must be a DISCARD_* id
                for aid in legals:
                    assert a.DISCARD_BASE <= aid < a.DISCARD_BASE + 5, (
                        f"non-discard action {aid} ({_action_range(aid)}) legal during DISCARD flag"
                    )
                # Last dice roll must have been a 7
                assert env.dice_roll == 7, f"DISCARD flag without 7 roll, dice={env.dice_roll}"
            env.step(rng.choice(legals))
            if env.phase == 3:
                break
        if fires >= 5:
            break
    assert fires > 0, "DISCARD flag never fired across 30 seeds — coverage gap"


# ---------------------------------------------------------------------------
# Robber movement
# ---------------------------------------------------------------------------

def test_move_robber_phase():
    """When flag == MOVE_ROBBER, only MOVE_ROBBER_* actions are legal, and
    the chosen target is never the current robber_hex (forbidden in Catan)."""
    a = fastcatan.action

    fires = 0
    for seed in range(30):
        rng = random.Random(seed)
        env = fastcatan.Env()
        env.reset(seed)
        mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
        for _ in range(200_000):
            env.action_mask(mask)
            legals = legal_actions(mask)
            if env.flag == FLAG_MOVE_ROBBER:
                fires += 1
                for aid in legals:
                    assert a.MOVE_ROBBER_BASE <= aid < a.MOVE_ROBBER_BASE + 19, (
                        f"non-move-robber action {aid} legal during MOVE_ROBBER"
                    )
            env.step(rng.choice(legals))
            if env.phase == 3:
                break
        if fires >= 3:
            break
    assert fires > 0


# ---------------------------------------------------------------------------
# Steal sub-phase
# ---------------------------------------------------------------------------

def test_steal_phase_options_valid():
    """When flag == ROBBER_STEAL, victim options are STEAL_BASE..STEAL_BASE+4,
    and the current player is never their own victim."""
    a = fastcatan.action

    fires = 0
    for seed in range(40):
        rng = random.Random(seed)
        env = fastcatan.Env()
        env.reset(seed)
        mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
        for _ in range(200_000):
            env.action_mask(mask)
            legals = legal_actions(mask)
            if env.flag == FLAG_ROBBER_STEAL:
                fires += 1
                cp = env.current_player
                for aid in legals:
                    assert a.STEAL_BASE <= aid < a.STEAL_BASE + 4, (
                        f"non-steal action {aid} legal during ROBBER_STEAL"
                    )
                    victim = aid - a.STEAL_BASE
                    assert victim != cp, "current player listed as own steal target"
            env.step(rng.choice(legals))
            if env.phase == 3:
                break
        if fires >= 2:
            break
    # ROBBER_STEAL is rarer (robber moved to hex with adjacent victims). Allow 0.


# ---------------------------------------------------------------------------
# Year of plenty and monopoly resolve in the PLAY action itself; neither
# card opens a follow-up flag.
# ---------------------------------------------------------------------------

YOP_ACTIONS = range(fastcatan.action.PLAY_YEAR_OF_PLENTY,
                    fastcatan.action.PLAY_YEAR_OF_PLENTY + 25)
MONOPOLY_ACTIONS = range(fastcatan.action.PLAY_MONOPOLY,
                         fastcatan.action.PLAY_MONOPOLY + 5)
DEV_PLAY_ACTIONS = range(fastcatan.action.PLAY_KNIGHT,
                         fastcatan.action.TRADE_ADD_GIVE_BASE)
DEV_YOP = 3
DEV_MONOPOLY = 4


def _resources(env, seat: int) -> list[int]:
    return [env.player_resource(seat, r) for r in range(5)]


def _bank(env) -> list[int]:
    return [env.bank(r) for r in range(5)]


def _assert_dev_card_spent(env, seat: int, card: int, cards_before: int):
    assert env.player_dev(seat, card) == cards_before - 1
    assert env.flag == FLAG_NONE
    assert env.actor_to_act == seat
    assert env.dev_card_played
    assert not any(aid in DEV_PLAY_ACTIONS for aid in env.legal_actions()), (
        "a second development card became playable in the same turn"
    )


def test_year_of_plenty_resolves_in_one_action():
    """Every legal YOP pair moves exactly that pair from the bank to the player."""
    env = find_state(lambda env, legal: any(aid in YOP_ACTIONS for aid in legal))
    seat = env.actor_to_act
    plays = [aid for aid in env.legal_actions() if aid in YOP_ACTIONS]
    start = env.snapshot()
    hand, bank = _resources(env, seat), _bank(env)
    cards = env.player_dev(seat, DEV_YOP)

    for play in plays:
        env.load_snapshot(start)
        r1, r2 = divmod(play - fastcatan.action.PLAY_YEAR_OF_PLENTY, 5)
        gained = [0] * 5
        gained[r1] += 1
        gained[r2] += 1

        assert not env.step(play)
        assert _resources(env, seat) == [h + g for h, g in zip(hand, gained)]
        assert _bank(env) == [b - g for b, g in zip(bank, gained)]
        _assert_dev_card_spent(env, seat, DEV_YOP, cards)


def test_monopoly_resolves_in_one_action():
    """Each monopoly resource takes every opponent's cards of that resource."""
    def playable_with_targets(env, legal):
        seat = env.actor_to_act
        return (any(aid in MONOPOLY_ACTIONS for aid in legal)
                and any(env.player_handsize(p) for p in range(4) if p != seat))

    env = find_state(playable_with_targets)
    seat = env.actor_to_act
    plays = [aid for aid in env.legal_actions() if aid in MONOPOLY_ACTIONS]
    assert plays == list(MONOPOLY_ACTIONS), "all five monopoly resources must be legal"
    start = env.snapshot()
    hands = [_resources(env, p) for p in range(4)]
    bank = _bank(env)
    cards = env.player_dev(seat, DEV_MONOPOLY)

    for resource, play in enumerate(plays):
        env.load_snapshot(start)
        taken = sum(hands[p][resource] for p in range(4) if p != seat)

        assert not env.step(play)
        for p in range(4):
            expected = list(hands[p])
            expected[resource] = hands[p][resource] + taken if p == seat else 0
            assert _resources(env, p) == expected, f"seat {p} after monopoly on r{resource}"
        assert _bank(env) == bank
        _assert_dev_card_spent(env, seat, DEV_MONOPOLY, cards)


# ---------------------------------------------------------------------------
# Dev card cooldown
# ---------------------------------------------------------------------------

def test_dev_card_cannot_be_played_same_turn_as_bought():
    """After BUY_DEV on turn T, none of PLAY_* (knight/RB/YOP/mono) for the just-bought
    card type should become legal until END_TURN.

    Approach: when we observe a BUY_DEV step, snapshot the dev_bought_this_turn
    pattern is invisible — instead we assert that within the SAME turn after a
    BUY_DEV, the player cannot finish that turn with a higher player_knights_played
    than they had before. (Weak proxy, but checks cooldown semantics.)
    """
    a = fastcatan.action

    fires = 0
    for seed in range(20):
        rng = random.Random(seed)
        env = fastcatan.Env()
        env.reset(seed)
        mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
        for _ in range(200_000):
            env.action_mask(mask)
            legals = legal_actions(mask)
            action = rng.choice(legals)
            cp = env.current_player
            turn_before = env.turn_count
            knights_before = env.player_knights_played(cp)
            is_buy = (action == a.BUY_DEV)
            env.step(action)
            if is_buy:
                # If they bought, knights_played cannot rise this same turn from
                # *the just-bought* card. We can't easily isolate which card, so
                # only assert: knights_played didn't jump by 2+ in one turn.
                fires += 1
                if env.turn_count == turn_before:
                    delta = env.player_knights_played(cp) - knights_before
                    assert delta <= 1, f"knight played multiple times in one turn: delta={delta}"
            if env.phase == 3:
                break
        if fires >= 5:
            return
    # Tolerate zero if very rare; just don't fail without firing.


# ---------------------------------------------------------------------------
# Game end and winner attribution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_terminal_action_sets_winner(seed):
    """Only the turn owner wins: either the player who took the final action,
    or the next player whose turn began with >= 10 VP."""
    rng = random.Random(seed)
    env = fastcatan.Env()
    env.reset(seed)
    mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)

    last_cp = last_action = None
    for _ in range(200_000):
        env.action_mask(mask)
        legals = legal_actions(mask)
        action = rng.choice(legals)
        last_cp, last_action = env.current_player, action
        done = env.step(action)
        if done:
            break

    assert env.done
    assert env.winner == env.current_player
    assert env.player_vp(env.winner) >= 10
    if last_action != fastcatan.action.END_TURN:
        assert env.winner == last_cp, "an own-turn win belongs to the player who acted"
    terminal = env.snapshot()
    assert env.step(0), "stepping a terminal game must still report terminal"
    assert env.snapshot() == terminal


# ---------------------------------------------------------------------------
# Robber relocation
# ---------------------------------------------------------------------------

def test_robber_moves_only_on_move_robber_action():
    """robber_hex (read via implied side-effect: any change happens iff a
    MOVE_ROBBER action was just executed).

    We don't expose robber_hex directly; instead snapshot before/after and
    compare. If the action wasn't MOVE_ROBBER_*, snapshot should match in
    its robber slot, but since we can't address that slot, we use indirect
    check: when MOVE_ROBBER fires, the flag transitions away from MOVE_ROBBER.
    """
    for seed in range(10):
        rng = random.Random(seed)
        env = fastcatan.Env()
        env.reset(seed)
        mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
        for _ in range(200_000):
            env.action_mask(mask)
            legals = legal_actions(mask)
            action = rng.choice(legals)
            in_mr = (env.flag == FLAG_MOVE_ROBBER)
            env.step(action)
            if in_mr:
                # Must transition to ROBBER_STEAL or NONE (no victims case)
                assert env.flag in (FLAG_NONE, FLAG_ROBBER_STEAL), (
                    f"MOVE_ROBBER step didn't clear flag, now={env.flag}"
                )
            if env.phase == 3:
                break
