from __future__ import annotations

import random
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    directory = Path(checkpoint_dir)
    if not directory.exists():
        return None
    candidates = sorted(directory.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def load_frontier_actions(path: str | Path | None) -> tuple[int, ...]:
    """Load a replayable action prefix from a JSON list or object."""
    if path is None:
        return ()

    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    actions = data.get("actions") if isinstance(data, dict) else data
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"No actions found in frontier file: {source}")

    try:
        return tuple(int(action) for action in actions)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid action in frontier file: {source}") from error
