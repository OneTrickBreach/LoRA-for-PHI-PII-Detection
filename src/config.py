"""Config + seed utilities. Single loader so every module reads config.yaml the same way.

rules.md §6.5 (config out of code) and §6.2 (fix and log seeds).
"""
from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load config.yaml once and cache it."""
    cfg_path = Path(path) if path else CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_global_seed(seed: int | None = None) -> int:
    """Seed Python, NumPy, and torch (if present). Returns the seed used, for logging."""
    cfg = load_config()
    seed = int(seed if seed is not None else cfg["seed"])
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


def label_list(cfg: dict[str, Any] | None = None) -> list[str]:
    """Entity type list in fixed order (label id stability)."""
    cfg = cfg or load_config()
    return list(cfg["label_types"])


def bio_label_list(cfg: dict[str, Any] | None = None) -> list[str]:
    """BIO label space: O + B-/I- per type. Index order is the model's label id order."""
    types = label_list(cfg)
    labels = ["O"]
    for t in types:
        labels.append(f"B-{t}")
        labels.append(f"I-{t}")
    return labels
