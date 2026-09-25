"""Exact physical transition enumeration."""
from __future__ import annotations

import math
import random

import fastcatan


def _pre_roll_state(seed: int = 0) -> fastcatan.Env:
    env = fastcatan.Env()
    env.reset(seed)
    while env.phase != 2:
        env.step(env.legal_actions()[0])
    assert env.dice_roll == 0
    return env


def _buy_dev_state(seed: int = 123) -> fastcatan.Env:
    env = fastcatan.Env()
    env.reset(seed)
    rng = random.Random(seed)
    for _ in range(5_000):
        legal = env.legal_actions()
        if fastcatan.action.BUY_DEV in legal:
            return env
        assert not env.step(rng.choice(legal))
    raise AssertionError("no development-card purchase became legal")


def test_illegal_and_deterministic_actions_have_one_exact_outcome() -> None:
    env = fastcatan.Env()
    env.reset(0)
    original = env.snapshot()

    outcomes = env.enumerate_outcomes(fastcatan.action.ROLL_DICE)
    assert outcomes == [(1.0, original)]
    assert env.snapshot() == original

    action = env.legal_actions()[0]
    [(probability, successor)] = env.enumerate_outcomes(action)
    direct = fastcatan.Env()
    direct.load_snapshot(original)
    direct.step(action)
    assert probability == 1.0
    assert successor == direct.snapshot()
    assert env.snapshot() == original


def test_dice_enumeration_has_exact_2d6_distribution() -> None:
    env = _pre_roll_state()
    original = env.snapshot()
    outcomes = env.enumerate_outcomes(fastcatan.action.ROLL_DICE)

    assert len(outcomes) == fastcatan.MAX_ACTION_OUTCOMES == 11
    assert math.isclose(sum(probability for probability, _ in outcomes), 1.0)

    child = fastcatan.Env()
    expected_ways = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
    for roll, ways, (probability, snapshot) in zip(
        range(2, 13), expected_ways, outcomes, strict=True
    ):
        child.load_snapshot(snapshot)
        assert child.dice_roll == roll
        assert probability == ways / 36

    sampled = fastcatan.Env()
    sampled.load_snapshot(original)
    sampled.step(fastcatan.action.ROLL_DICE)
    assert sampled.snapshot() in [snapshot for _, snapshot in outcomes]

    assert env.snapshot() == original


def test_development_draw_distribution_uses_physical_deck() -> None:
    env = _buy_dev_state()
    actor = env.actor_to_act
    deck_before = [env.dev_deck(card) for card in range(5)]
    total = sum(deck_before)
    original = env.snapshot()

    outcomes = env.enumerate_outcomes(fastcatan.action.BUY_DEV)
    assert len(outcomes) == sum(count > 0 for count in deck_before)
    assert math.isclose(sum(probability for probability, _ in outcomes), 1.0)

    child = fastcatan.Env()
    seen: set[int] = set()
    for probability, snapshot in outcomes:
        child.load_snapshot(snapshot)
        deck_after = [child.dev_deck(card) for card in range(5)]
        drawn = [
            card for card in range(5)
            if deck_after[card] == deck_before[card] - 1
        ]
        assert len(drawn) == 1
        card = drawn[0]
        seen.add(card)
        assert probability == deck_before[card] / total
        assert child.player_total_dev(actor) == env.player_total_dev(actor) + 1

    assert seen == {card for card, count in enumerate(deck_before) if count}
    sampled = fastcatan.Env()
    sampled.load_snapshot(original)
    sampled.step(fastcatan.action.BUY_DEV)
    assert sampled.snapshot() in [snapshot for _, snapshot in outcomes]
    assert env.snapshot() == original


def test_robber_steal_distribution_uses_victim_hand() -> None:
    env = fastcatan.Env()
    env.reset(0)
    rng = random.Random(0)
    steal_action = None
    for _ in range(1_000):
        legal = env.legal_actions()
        steal = [
            action for action in legal
            if fastcatan.action.STEAL_BASE
            <= action
            < fastcatan.action.STEAL_BASE + fastcatan.NUM_PLAYERS
        ]
        if steal:
            steal_action = steal[0]
            break
        assert not env.step(rng.choice(legal))

    assert steal_action is not None
    actor = env.current_player
    victim = steal_action - fastcatan.action.STEAL_BASE
    held = [env.player_resource(victim, resource) for resource in range(5)]
    total = sum(held)

    outcomes = env.enumerate_outcomes(steal_action)
    assert len(outcomes) == sum(count > 0 for count in held)

    child = fastcatan.Env()
    for probability, snapshot in outcomes:
        child.load_snapshot(snapshot)
        stolen = [
            resource for resource in range(5)
            if child.player_resource(victim, resource) == held[resource] - 1
        ]
        assert len(stolen) == 1
        resource = stolen[0]
        assert probability == held[resource] / total
        assert child.player_resource(actor, resource) == (
            env.player_resource(actor, resource) + 1
        )

    sampled = fastcatan.Env()
    sampled.load_snapshot(env.snapshot())
    sampled.step(steal_action)
    assert sampled.snapshot() in [snapshot for _, snapshot in outcomes]


def test_robber_move_enumerates_automatic_steal() -> None:
    env = fastcatan.Env()
    env.reset(0)
    rng = random.Random(0)
    selected = None
    for _ in range(500):
        legal = env.legal_actions()
        if env.flag == 2:
            for action in legal:
                outcomes = env.enumerate_outcomes(action)
                if len(outcomes) > 1:
                    selected = action, outcomes
                    break
        if selected is not None:
            break
        assert not env.step(rng.choice(legal))

    assert selected is not None
    action, outcomes = selected
    assert math.isclose(sum(probability for probability, _ in outcomes), 1.0)

    child = fastcatan.Env()
    for _, snapshot in outcomes:
        child.load_snapshot(snapshot)
        assert child.robber_hex == action - fastcatan.action.MOVE_ROBBER_BASE
        assert child.flag == 0
