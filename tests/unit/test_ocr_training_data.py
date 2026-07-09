"""Kiểm thử row splitting và export OCR line-level dataset."""

import numpy as np
from PIL import Image

from vlpr.data.ocr_layout import find_compact_row_split, split_compact_crop


def _two_line_image() -> Image.Image:
    """Tạo ảnh có hai vùng biến thiên và một khe trắng quanh chính giữa."""
    array = np.full((40, 30), 255, dtype=np.uint8)
    array[5:15, 3:27:2] = 0
    array[25:35, 4:26:2] = 0
    return Image.fromarray(array, mode="L")


def test_find_compact_row_split_selects_center_gap() -> None:
    """Projection phải chọn khe giữa hai dòng thay vì cắt qua ký tự."""
    split = find_compact_row_split(
        _two_line_image(),
        search_start=0.35,
        search_end=0.65,
    )

    assert 15 <= split <= 24


def test_split_compact_crop_preserves_full_height() -> None:
    """Hai line crops phải ghép lại đúng chiều cao ảnh gốc."""
    image = _two_line_image()

    top, bottom = split_compact_crop(
        image,
        search_start=0.35,
        search_end=0.65,
    )

    assert top.width == image.width == bottom.width
    assert top.height + bottom.height == image.height
