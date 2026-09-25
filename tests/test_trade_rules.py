"""Player-to-player trade rule regressions."""

from __future__ import annotations

import random

import numpy as np

import fastcatan


def _legal_actions(env: fastcatan.Env, mask: np.ndarray) -> list[int]:
    env.action_mask(mask)
    return [
        action
        for action in range(fastcatan.NUM_ACTIONS)
        if (int(mask[action >> 6]) >> (action & 63)) & 1
    ]


def _state_with_give_action() -> tuple[fastcatan.Env, int, np.ndarray]:
    rng = random.Random(17)
    env = fastcatan.Env()
    env.reset(17)
    mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)

    for _ in range(2_000):
        legal = _legal_actions(env, mask)
        give = [
            action for action in legal
            if fastcatan.action.TRADE_ADD_GIVE_BASE
            <= action < fastcatan.action.TRADE_ADD_GIVE_BASE + 5
        ]
        if give:
            return env, give[0] - fastcatan.action.TRADE_ADD_GIVE_BASE, mask
        assert legal
        done = env.step(rng.choice(legal))
        assert not done

    raise AssertionError("seed-17 fixture never reached trade composition")


def test_trade_cannot_request_a_resource_already_offered() -> None:
    env, resource, mask = _state_with_give_action()
    env.step(fastcatan.action.TRADE_ADD_GIVE_BASE + resource)

    legal = _legal_actions(env, mask)
    same_resource_want = fastcatan.action.TRADE_ADD_WANT_BASE + resource
    assert same_resource_want not in legal

    before = env.snapshot()
    done = env.step(same_resource_want)
    assert env.snapshot() == before
    assert int(done) == 0
