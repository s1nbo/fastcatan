"""Information-set regression tests for deployable observations."""

from __future__ import annotations

import random

import numpy as np

import fastcatan


TRADE_WIDTH = 5 + 5 + 5 + 3 * 4
TRADE_OFFSET = fastcatan.OBS_SIZE - TRADE_WIDTH


def _legal_actions(env: fastcatan.Env, mask: np.ndarray) -> list[int]:
    env.action_mask(mask)
    legal: list[int] = []
    for word_idx, word in enumerate(mask):
        bits = int(word)
        while bits:
            bit = (bits & -bits).bit_length() - 1
            legal.append(word_idx * 64 + bit)
            bits &= bits - 1
    return legal


def _state_where_buy_dev_is_legal() -> fastcatan.Env:
    rng = random.Random(123)
    env = fastcatan.Env()
    env.reset(123)
    mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)

    for _ in range(2_000):
        legal = _legal_actions(env, mask)
        if fastcatan.action.BUY_DEV in legal:
            return env
        assert legal
        _, done = env.step(rng.choice(legal))
        assert not done, "seed-123 fixture ended before a dev purchase was legal"

    raise AssertionError("seed-123 fixture never made BUY_DEV legal")


def test_hidden_dev_draw_identity_does_not_change_opponent_observation() -> None:
    """Different hidden cards from one public state must look identical."""
    source = _state_where_buy_dev_is_legal()
    snapshot = source.snapshot()
    buyer = source.actor_to_act
    observer = (buyer + 1) % fastcatan.NUM_PLAYERS

    outcomes: dict[bytes, np.ndarray] = {}
    for seed in range(64):
        env = fastcatan.Env()
        env.load_snapshot(snapshot)
        env.reseed(seed)
        _, done = env.step(fastcatan.action.BUY_DEV)
        assert not done

        obs = np.zeros(fastcatan.OBS_SIZE, dtype=np.float32)
        oracle = np.zeros(fastcatan.OBS_FULL_SIZE, dtype=np.float32)
        env.write_obs(observer, obs)
        env.write_obs_full(observer, oracle)
        outcomes.setdefault(oracle[fastcatan.OBS_SIZE:].tobytes(), obs.copy())

    # The oracle confirms that multiple private card identities were sampled.
    assert len(outcomes) >= 2
    public_views = list(outcomes.values())
    for obs in public_views[1:]:
        np.testing.assert_array_equal(obs, public_views[0])


def test_unpublished_trade_draft_is_visible_only_to_its_owner() -> None:
    source = _state_where_buy_dev_is_legal()
    actor = source.actor_to_act
    observer = (actor + 1) % fastcatan.NUM_PLAYERS
    mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)

    legal = _legal_actions(source, mask)
    give_actions = [
        action for action in legal
        if fastcatan.action.TRADE_ADD_GIVE_BASE
        <= action < fastcatan.action.TRADE_ADD_GIVE_BASE + 5
    ]
    assert give_actions
    give_action = give_actions[0]
    give_resource = give_action - fastcatan.action.TRADE_ADD_GIVE_BASE
    want_resource = (give_resource + 1) % 5
    want_action = fastcatan.action.TRADE_ADD_WANT_BASE + want_resource

    opponent_before = np.zeros(fastcatan.OBS_SIZE, dtype=np.float32)
    opponent_draft = np.zeros_like(opponent_before)
    owner_draft = np.zeros_like(opponent_before)
    opponent_open = np.zeros_like(opponent_before)
    source.write_obs(observer, opponent_before)

    source.step(give_action)
    assert want_action in _legal_actions(source, mask)
    source.step(want_action)
    source.write_obs(observer, opponent_draft)
    source.write_obs(actor, owner_draft)

    np.testing.assert_array_equal(opponent_draft, opponent_before)
    owner_trade = owner_draft[TRADE_OFFSET:]
    assert owner_trade[5 + give_resource] > 0
    assert owner_trade[10 + want_resource] > 0
    np.testing.assert_array_equal(
        owner_trade[15:].reshape(3, 4),
        np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (3, 1)),
    )

    assert fastcatan.action.TRADE_OPEN in _legal_actions(source, mask)
    source.step(fastcatan.action.TRADE_OPEN)
    source.write_obs(observer, opponent_open)
    assert not np.array_equal(opponent_open, opponent_before)
    open_trade = opponent_open[TRADE_OFFSET:]
    assert open_trade[5 + give_resource] > 0
    assert open_trade[10 + want_resource] > 0
