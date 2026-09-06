from __future__ import annotations

import random
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
