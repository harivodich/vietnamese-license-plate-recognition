"""Điều khiển tính tái lập cho Python, NumPy và các framework ML tùy chọn."""

import importlib
import os
import random
from typing import Any

import numpy as np


def set_random_seed(seed: int, *, deterministic: bool = True) -> None:
    """Đặt cùng seed cho mọi bộ sinh số ngẫu nhiên trước khi tạo dữ liệu hoặc model."""
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    _seed_torch(seed, deterministic=deterministic)
    _seed_paddle(seed)


def _optional_module(name: str) -> Any | None:
    """Import framework tùy chọn và trả về ``None`` nếu môi trường chưa cài framework đó."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _seed_torch(seed: int, *, deterministic: bool) -> None:
    """Đặt seed PyTorch trên CPU/GPU và bật thuật toán deterministic khi được yêu cầu."""
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
    """Đặt seed cho PaddlePaddle nếu package này có trong môi trường."""
    paddle = _optional_module("paddle")
    if paddle is not None:
        paddle.seed(seed)
