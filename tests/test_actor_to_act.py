"""Regression coverage for turn-owner versus acting-seat semantics."""

from __future__ import annotations

import random

import fastcatan


def _first_cross_seat_discard() -> fastcatan.Env:
    """Reach a deterministic state where the discarder is not the turn owner."""
    rng = random.Random(0)
    env = fastcatan.Env()
    env.reset(0)

    for _ in range(2_000):
        if env.flag == 1 and env.discarding_player != env.current_player:
            return env
        legal = env.legal_actions()
        assert legal
        done = env.step(rng.choice(legal))
        assert not done, "seed-0 fixture ended before reaching discard state"

    raise AssertionError("seed-0 fixture did not reach a cross-seat discard")


def test_actor_to_act_uses_discarding_player() -> None:
    env = _first_cross_seat_discard()

    assert env.current_player != env.discarding_player
    assert env.actor_to_act == env.discarding_player
