"""Detailed game logs and the browser viewer.

- `GameLogWriter` / `read_game_log`: detailed per-step game logs (JSONL).
- `from_events`: native event-log file -> game log, by replay (``fastcatan convert``).
- `record_demo`: demo games with simple random agents (``fastcatan demo``).
- `viewer_path` / `open_viewer`: the single-file viewer (``fastcatan view``).
"""
from __future__ import annotations

import sys
from pathlib import Path

from .gamelog import GameLogWriter, GameRecorder, read_game_log  # noqa: F401

__all__ = ["GameLogWriter", "GameRecorder", "read_game_log", "viewer_path", "open_viewer"]


def viewer_path() -> Path:
    """Path of the bundled single-file viewer (vanilla JS, no network)."""
    return Path(__file__).resolve().parent / "viewer.html"


def open_viewer() -> None:
    """Open the viewer in the default browser; drop a log file on it."""
    import webbrowser

    path = viewer_path()
    print(path)
    if not webbrowser.open(path.as_uri()):
        print("could not start a browser; open the file above by hand", file=sys.stderr)
