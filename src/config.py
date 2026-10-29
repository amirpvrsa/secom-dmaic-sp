"""Load config.yaml. Kept tiny on purpose: one place that reads settings."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Turn a path from config.yaml into an absolute path under the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
