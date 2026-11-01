"""Turn a native event-log file into a detailed game log for the viewer.

Training records compact int32 rows through ``fastcatan.events.EventWriter``
(``LOG_GAMES | LOG_ACTIONS`` at least). This replays the chosen games through
the simulator from their seed and actions, checking every step against the
rows, and writes them with `gamelog.GameLogWriter`. No agents or checkpoints
are involved; the replay needs only the fastcatan version that recorded the
file.

    fastcatan-viz-convert run.fcev                        # every game -> run.jsonl.gz
    fastcatan-viz-convert run.fcev --finished --limit 20  # first 20 finished games
    fastcatan-viz-convert run.fcev --env 3 --game 7 -o one.jsonl.gz
    fastcatan-viz-convert run.fcev --list                 # what is in the file

Agent output (values, policies) is not in the native rows; log evaluation
games with `GameLogWriter` directly when you want it in the viewer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fastcatan
from fastcatan.events import ReplayError, check_version, read_events, replay

from .gamelog import GameLogWriter


def _result(game) -> str:
    if not game.finished:
        return "unfinished"
    if game.truncated:
        return "truncated"
    return f"P{game.winner} won"


def main() -> int:
    p = argparse.ArgumentParser(description="Convert an event-log file into a viewer game log.")
    p.add_argument("path", type=Path)
    p.add_argument("-o", "--out", type=Path,
                   help="output .jsonl / .jsonl.gz (default: next to the input)")
    p.add_argument("--env", type=int, help="only games from this batch slot")
    p.add_argument("--game", type=int, help="only this game_id")
    p.add_argument("--finished", action="store_true", help="skip games that did not end")
    p.add_argument("--limit", type=int, help="convert at most this many games")
    p.add_argument("--list", action="store_true", help="list the games and exit")
    p.add_argument("--allow-version-mismatch", action="store_true",
                   help="replay with a different fastcatan than the one that recorded "
                        "the file; a divergence is still detected and reported")
    args = p.parse_args()

    log = read_events(args.path)
    games = [g for g in log.games()
             if (args.env is None or g.env_idx == args.env)
             and (args.game is None or g.game_id == args.game)
             and (not args.finished or g.finished)]
    if args.limit is not None:
        games = games[:args.limit]

    if args.list:
        print(f"{args.path}: fastcatan {log.header['fastcatan']}, "
              f"{len(log.rows)} rows{' (cut off)' if log.cut else ''}")
        for g in games:
            print(f"  env {g.env_idx:4d} game {g.game_id:5d}  {len(g.actions):6d} actions  "
                  f"{_result(g):10s}  {g.agents or ''}")
        return 0

    try:
        check_version(log)
    except ReplayError as e:
        if not args.allow_version_mismatch:
            print(f"error: {e} (or pass --allow-version-mismatch)", file=sys.stderr)
            return 2
        print(f"warning: {e}", file=sys.stderr)

    if not games:
        print("no games match", file=sys.stderr)
        return 1

    out = args.out or args.path.with_name(args.path.name.split(".")[0] + ".jsonl.gz")
    env = fastcatan.Env()
    written = failed = 0
    with GameLogWriter(out) as writer:
        for g in games:
            if g.seed is None:
                print(f"skip env {g.env_idx} game {g.game_id}: no GAME_START row", file=sys.stderr)
                failed += 1
                continue
            meta = {**g.meta, "source": {"file": args.path.name, "env_idx": g.env_idx,
                                         "game_id": g.game_id,
                                         "recorded_with": log.header["fastcatan"]}}
            # Replay once to verify before writing anything: a game that
            # diverges midway must not leave half a game in the output.
            try:
                for _ in replay(g, env):
                    pass
            except ReplayError as e:
                print(f"skip: {e}", file=sys.stderr)
                failed += 1
                continue
            steps = replay(g, env)
            env.reset(g.seed)
            rec = writer.start_game(env, seed=g.seed, agents=g.agents, meta=meta)
            for action in steps:
                rec.record(action)
            writer.flush()
            written += 1
            print(f"env {g.env_idx} game {g.game_id}: {_result(g)}, {rec.steps} steps")
    print(f"wrote {written} game(s) to {out}" + (f", skipped {failed}" if failed else ""))
    return 0 if written else 1


def main_cli() -> None:
    sys.exit(main())


if __name__ == "__main__":
    main_cli()
