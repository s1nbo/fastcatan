"""Shared helpers for simulator tests."""
from __future__ import annotations

import random
import numpy as np

import fastcatan


MAX_STEPS = 100_000


def legal_actions(mask: np.ndarray) -> list[int]:
    """Return action IDs whose bit is set in the uint64 bitmask buffer."""
    out: list[int] = []
    for word_idx, word in enumerate(mask):
        w = int(word)
        base = word_idx * 64
        while w:
            bit = (w & -w).bit_length() - 1
            out.append(base + bit)
            w &= w - 1
    return out


def buy_dev_state(seed: int = 123) -> fastcatan.Env:
    """Play random legal moves until buying a development card is legal."""
    env = fastcatan.Env()
    env.reset(seed)
    rng = random.Random(seed)
    for _ in range(5_000):
        legal = env.legal_actions()
        if fastcatan.action.BUY_DEV in legal:
            return env
        assert not env.step(rng.choice(legal))
    raise AssertionError("no development-card purchase became legal")


def find_state(predicate, avoid=frozenset(), max_seeds: int = 20) -> fastcatan.Env:
    """Return the first seeded state where ``predicate(env, legal)`` holds.

    Rollouts buy every affordable development card and otherwise play a random
    legal action outside ``avoid``, so rare dev-card states are reached quickly
    and deterministically.
    """
    buy = fastcatan.action.BUY_DEV
    for seed in range(max_seeds):
        rng = random.Random(seed)
        env = fastcatan.Env()
        env.reset(seed)
        while not env.done:
            legal = env.legal_actions()
            if predicate(env, legal):
                return env
            if buy in legal:
                env.step(buy)
            else:
                env.step(rng.choice([a for a in legal if a not in avoid] or legal))
    raise AssertionError(f"no state matched across {max_seeds} seeded games")


def play_random_game(seed: int, max_steps: int = MAX_STEPS, on_step=None):
    """Play one full game using random legal moves."""
    rng = random.Random(seed)
    env = fastcatan.Env()
    env.reset(seed)
    mask_buf = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)

    for step_idx in range(max_steps):
        env.action_mask(mask_buf)
        legals = legal_actions(mask_buf)
        assert legals, f"no legal actions at step {step_idx}, phase={env.phase} flag={env.flag}"
        action = rng.choice(legals)
        if on_step is not None:
            on_step(env, mask_buf, action, step_idx)
        done = env.step(action)
        if done:
            return env, step_idx + 1
    raise AssertionError(f"game did not terminate in {max_steps} steps (seed={seed})")
