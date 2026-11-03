"""The ``fastcatan`` command (also ``python -m fastcatan``).

    fastcatan view                          # open the game-log viewer
    fastcatan convert run.fcev --finished   # event-log file -> viewer game log
    fastcatan demo                          # record demo games to logs/demo.jsonl.gz
"""
from __future__ import annotations

import argparse
import sys

from .viz import from_events, open_viewer, record_demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fastcatan")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("view", help="open the game-log viewer in a browser")
    from_events.add_arguments(commands.add_parser(
        "convert", help="convert an event-log file (.fcev) into a viewer game log"))
    record_demo.add_arguments(commands.add_parser(
        "demo", help="record demo games with simple random agents"))

    args = parser.parse_args(argv)
    if args.command == "view":
        open_viewer()
        return 0
    if args.command == "convert":
        return from_events.run(args)
    return record_demo.run(args)


if __name__ == "__main__":
    sys.exit(main())
