"""Kiểm thử việc khởi tạo số ngẫu nhiên có thể tái lập."""

import random

import numpy as np
import pytest

from vlpr.training.reproducibility import set_random_seed


def test_set_random_seed_repeats_python_and_numpy_sequences() -> None:
    """Xác nhận cùng một seed tạo lại cùng chuỗi random của Python và NumPy."""
    set_random_seed(42)
    first_python = random.random()
    first_numpy = np.random.random()

    set_random_seed(42)

    assert random.random() == first_python
    assert np.random.random() == first_numpy


def test_set_random_seed_rejects_negative_seed() -> None:
    """Xác nhận seed âm bị từ chối bằng thông báo lỗi có ý nghĩa."""
    with pytest.raises(ValueError, match="non-negative"):
        set_random_seed(-1)
