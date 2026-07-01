"""Reproducibility controls for Python, NumPy, and optional ML frameworks."""

import importlib
import os
import random
from typing import Any

import numpy as np


def set_random_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed available random generators before data loading or model creation."""
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    _seed_torch(seed, deterministic=deterministic)
    _seed_paddle(seed)


def _optional_module(name: str) -> Any | None:
    """Import an optional framework without making it a core dependency."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _seed_torch(seed: int, *, deterministic: bool) -> None:
    torch = _optional_module("torch")
    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def _seed_paddle(seed: int) -> None:
    paddle = _optional_module("paddle")
    if paddle is not None:
        paddle.seed(seed)
