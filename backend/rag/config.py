"""YAML config loader for the rag package."""
from __future__ import annotations

from pathlib import Path

import yaml

_DEFAULT_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or _DEFAULT_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
