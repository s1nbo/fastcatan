"""Release metadata stays synchronized across package entry points."""
from __future__ import annotations

from pathlib import Path
import tomllib

import fastcatan


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_and_runtime_versions_match() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)["project"]

    assert project["version"] == fastcatan.__version__


def test_release_description_files_exist() -> None:
    assert (PROJECT_ROOT / "README.md").is_file()
    assert (PROJECT_ROOT / "CHANGELOG.md").is_file()
