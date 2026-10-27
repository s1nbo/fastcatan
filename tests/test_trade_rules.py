"""Player-to-player trade rule regressions."""

from __future__ import annotations

import random

import fastcatan


def _state_with_give_action() -> tuple[fastcatan.Env, int]:
    rng = random.Random(17)
    env = fastcatan.Env()
    env.reset(17)

    for _ in range(2_000):
        legal = env.legal_actions()
        give = [
            action for action in legal
            if fastcatan.action.TRADE_ADD_GIVE_BASE
            <= action < fastcatan.action.TRADE_ADD_GIVE_BASE + 5
        ]
        if give:
            return env, give[0] - fastcatan.action.TRADE_ADD_GIVE_BASE
        assert legal
        done = env.step(rng.choice(legal))
        assert not done

    raise AssertionError("seed-17 fixture never reached trade composition")


def test_trade_cannot_request_a_resource_already_offered() -> None:
    env, resource = _state_with_give_action()
    env.step(fastcatan.action.TRADE_ADD_GIVE_BASE + resource)

    legal = env.legal_actions()
    same_resource_want = fastcatan.action.TRADE_ADD_WANT_BASE + resource
    assert same_resource_want not in legal

    before = env.snapshot()
    done = env.step(same_resource_want)
    assert env.snapshot() == before
    assert int(done) == 0
