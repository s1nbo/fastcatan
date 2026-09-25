"""Measure batched simulator throughput with uniformly random legal moves."""
from __future__ import annotations

import argparse
import time

import numpy as np

import fastcatan


def fill_random_legal_actions(
    masks: np.ndarray, actions: np.ndarray, rng: np.random.Generator
) -> None:
    for row, mask in enumerate(masks):
        legal: list[int] = []
        for word_index, word in enumerate(mask):
            bits = int(word)
            while bits:
                bit = (bits & -bits).bit_length() - 1
                action = word_index * 64 + bit
                if action < fastcatan.NUM_ACTIONS:
                    legal.append(action)
                bits &= bits - 1
        actions[row] = (
            rng.choice(legal) if legal else fastcatan.NO_ACTION
        )


def run(num_envs: int, steps: int, seed: int) -> None:
    games = fastcatan.BatchedEnv(num_envs, seed)
    games.reset()
    masks = np.zeros((num_envs, fastcatan.MASK_WORDS), dtype=np.uint64)
    actions = np.zeros(num_envs, dtype=np.uint32)
    dones = np.zeros(num_envs, dtype=np.uint8)
    rng = np.random.default_rng(seed)

    started = time.perf_counter()
    completed = 0
    for _ in range(steps):
        games.write_masks(masks)
        fill_random_legal_actions(masks, actions, rng)
        games.step_autoreset(actions, dones)
        completed += int(np.count_nonzero(dones))
    elapsed = time.perf_counter() - started

    simulator_steps = num_envs * steps
    print(f"games:          {num_envs:,}")
    print(f"steps/game:     {steps:,}")
    print(f"completed:      {completed:,}")
    print(f"elapsed:        {elapsed:.3f} s")
    print(f"simulator rate: {simulator_steps / elapsed:,.0f} steps/s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=1_024)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.envs, args.steps, args.seed)


if __name__ == "__main__":
    main()
