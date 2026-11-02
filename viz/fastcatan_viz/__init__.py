"""Game logs and a browser viewer for fastcatan.

Kept out of the core simulator and off PyPI; install from the repository:
``pip install "fastcatan-viz @ git+https://github.com/s1nbo/fastcatan#subdirectory=viz"``.

- `GameLogWriter` / `read_game_log`: detailed per-step game logs (JSONL).
- ``fastcatan-viz-convert``: native event-log file -> game log, by replay.
- ``fastcatan-viz-demo``: record demo games with simple random agents.
- ``fastcatan-viz``: open the viewer in a browser.
"""
from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

from .gamelog import GameLogWriter, GameRecorder, read_game_log  # noqa: F401

__all__ = ["GameLogWriter", "GameRecorder", "read_game_log", "viewer_path", "open_viewer"]


def viewer_path() -> Path:
    """Path of the bundled single-file viewer (vanilla JS, no network)."""
    return Path(__file__).resolve().parent / "viewer.html"


def open_viewer() -> None:
    """Open the viewer in the default browser; drop a log file on it."""
    path = viewer_path()
    print(path)
    if not webbrowser.open(path.as_uri()):
        print("could not start a browser; open the file above by hand", file=sys.stderr)
