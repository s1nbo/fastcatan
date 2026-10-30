#!/usr/bin/env python3
"""Record demo evaluation games as a detailed game log.

Plays `--games` games on ``Env`` with simple random agents and writes every
step through `gamelog.GameLogWriter`, exactly as an evaluation loop with real
agents would. Drop the result on ``viz/viewer.html`` to watch it.

    python viz/record_demo.py                                  # -> viz/logs/demo.jsonl.gz
    python viz/record_demo.py --agents builder,builder,random,random --games 2

Agents:
    builder   random legal actions, biased toward building and away from
              composing trades, so games finish in a watchable length
    random    uniformly random legal actions (long games, lots of trade spam)
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import fastcatan

from gamelog import GameLogWriter

ACT = fastcatan.action
HERE = Path(__file__).resolve().parent

# Relative weight per action family for the `builder` agent.
_WEIGHTS = [
    (ACT.SETTLE_BASE, ACT.CITY_BASE, 40.0),
    (ACT.CITY_BASE, ACT.ROAD_BASE, 60.0),
    (ACT.ROAD_BASE, ACT.ROLL_DICE, 4.0),
    (ACT.BUY_DEV, ACT.BUY_DEV + 1, 10.0),
    (ACT.PLAY_KNIGHT, ACT.PLAY_KNIGHT + 1, 6.0),
    (ACT.TRADE_BASE, ACT.BUY_DEV, 1.5),
    (ACT.TRADE_ADD_GIVE_BASE, ACT.TRADE_OPEN, 0.15),
    (ACT.TRADE_OPEN, ACT.TRADE_OPEN + 1, 2.0),
    (ACT.TRADE_ACCEPT, ACT.TRADE_ACCEPT + 1, 3.0),
    (ACT.TRADE_CONFIRM_BASE, ACT.TRADE_CANCEL, 8.0),
]


def _weight(a: int) -> float:
    for lo, hi, w in _WEIGHTS:
        if lo <= a < hi:
            return w
    return 1.0


WEIGHTS = [_weight(a) for a in range(fastcatan.NUM_ACTIONS)]


class RandomAgent:
    def __init__(self, kind: str, seed: int):
        self.kind = kind
        self.rng = random.Random(seed)

    def act(self, env: fastcatan.Env) -> int:
        legal = env.legal_actions()
        if self.kind == "random":
            return self.rng.choice(legal)
        return self.rng.choices(legal, weights=[WEIGHTS[a] for a in legal])[0]


def main() -> None:
    p = argparse.ArgumentParser(description="Record demo evaluation games as a detailed game log.")
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--agents", default="builder,builder,builder,builder",
                   help="four comma-separated agent kinds, one per seat")
    p.add_argument("--max-steps", type=int, default=50_000,
                   help="actions per game before giving up on it")
    p.add_argument("--out", type=Path, default=HERE / "logs" / "demo.jsonl.gz")
    args = p.parse_args()

    kinds = [k.strip() for k in args.agents.split(",")]
    if len(kinds) != 4 or any(k not in ("builder", "random") for k in kinds):
        p.error("--agents needs four of: builder, random")

    rng = random.Random(args.seed)
    env = fastcatan.Env()
    with GameLogWriter(args.out) as log:
        for g in range(args.games):
            seed = rng.getrandbits(64)
            agents = [RandomAgent(k, rng.getrandbits(32)) for k in kinds]
            env.reset(seed)
            rec = log.start_game(env, seed=seed,
                                 agents=[f"{k} (seat {s})" for s, k in enumerate(kinds)],
                                 meta={"run": "record_demo", "game": g})
            while not env.done and rec.steps < args.max_steps:
                action = agents[env.actor_to_act].act(env)
                env.step(action)
                rec.record(action)
            vps = [env.player_vp(s) for s in range(4)]
            result = f"P{env.winner} wins" if env.done else "unfinished"
            print(f"game {g}: {result} in {env.turn_count} turns, {rec.steps} steps, vp {vps}")
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
