"""Regression coverage for turn-owner versus acting-seat semantics."""

from __future__ import annotations

import random

import numpy as np

import fastcatan


def _legal_actions(mask: np.ndarray) -> list[int]:
    legal: list[int] = []
    for word_idx, word in enumerate(mask):
        bits = int(word)
        while bits:
            bit = (bits & -bits).bit_length() - 1
            legal.append(word_idx * 64 + bit)
            bits &= bits - 1
    return legal


def _first_cross_seat_discard() -> fastcatan.Env:
    """Reach a deterministic state where the discarder is not the turn owner."""
    rng = random.Random(0)
    env = fastcatan.Env()
    env.reset(0)
    mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)

    for _ in range(2_000):
        if env.flag == 1 and env.discarding_player != env.current_player:
            return env
        env.action_mask(mask)
        legal = _legal_actions(mask)
        assert legal
        _, done = env.step(rng.choice(legal))
        assert not done, "seed-0 fixture ended before reaching discard state"

    raise AssertionError("seed-0 fixture did not reach a cross-seat discard")


def test_actor_to_act_uses_discarding_player() -> None:
    env = _first_cross_seat_discard()

    assert env.current_player != env.discarding_player
    assert env.actor_to_act == env.discarding_player


def test_batched_default_obs_and_signature_use_actor_to_act() -> None:
    env = _first_cross_seat_discard()
    snapshot = np.frombuffer(env.snapshot(), dtype=np.uint8).copy()[None, :]

    batch = fastcatan.BatchedEnv(1, 123)
    batch.load_snapshots(snapshot)

    expected = np.zeros(fastcatan.OBS_SIZE, dtype=np.float32)
    actual = np.zeros((1, fastcatan.OBS_SIZE), dtype=np.float32)
    env.write_obs(env.actor_to_act, expected)
    batch.write_obs(actual)
    np.testing.assert_array_equal(actual[0], expected)

    sig = np.zeros((1, fastcatan.SIG_INTS), dtype=np.int32)
    batch.write_sigs(sig)
    assert batch.current_player(0) == env.current_player
    assert batch.actor_to_act(0) == env.actor_to_act
    assert sig[0, 0] == env.actor_to_act
