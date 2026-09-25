"""Measure completed games/second with uniformly random legal moves."""
from __future__ import annotations

import argparse
import time

import fastcatan


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def run(num_envs: int, num_games: int, seed: int) -> None:
    games = fastcatan.BatchedEnv(num_envs, seed)
    games.reset()

    started = time.perf_counter()
    completed, simulator_steps = games.run_random_games(num_games)
    elapsed = time.perf_counter() - started

    if completed != num_games:
        raise RuntimeError(
            f"native rollout completed {completed:,} of {num_games:,} games"
        )

    print(f"parallel games:  {num_envs:,}")
    print(f"completed games: {completed:,}")
    print(f"simulator steps: {simulator_steps:,}")
    print(f"steps/game:      {simulator_steps / completed:,.1f}")
    print(f"elapsed:         {elapsed:.3f} s")
    print(f"game rate:       {completed / elapsed:,.1f} games/s")
    print(f"simulator rate:  {simulator_steps / elapsed:,.0f} steps/s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=positive_int, default=1_024)
    parser.add_argument("--games", type=positive_int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.envs, args.games, args.seed)


if __name__ == "__main__":
    main()
