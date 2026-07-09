"""OCR crop layout helpers shared by training and evaluation."""

import numpy as np
from PIL import Image


def find_compact_row_split(
    image: Image.Image,
    *,
    search_start: float,
    search_end: float,
) -> int:
    """Find the low-variation row between two text lines in a compact plate crop."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    height = gray.shape[0]
    if height < 4:
        raise ValueError(f"compact crop is too short to split rows: {height}")
    first = max(1, round(height * search_start))
    last = min(height - 1, round(height * search_end))
    if first >= last:
        raise ValueError(f"empty split search range with height={height}")

    # Horizontal row variance is low in the gap between two plate text rows.
    row_variation = gray.std(axis=1)
    kernel_size = min(5, max(1, last - first))
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    smoothed = np.convolve(row_variation, kernel, mode="same")
    return int(first + np.argmin(smoothed[first:last]))


def split_compact_crop(
    image: Image.Image,
    *,
    search_start: float,
    search_end: float,
) -> tuple[Image.Image, Image.Image]:
    """Split a compact two-line crop into top and bottom line crops."""
    split_row = find_compact_row_split(
        image,
        search_start=search_start,
        search_end=search_end,
    )
    top = image.crop((0, 0, image.width, split_row))
    bottom = image.crop((0, split_row, image.width, image.height))
    if top.height == 0 or bottom.height == 0:
        raise ValueError("row split produced an empty crop")
    return top, bottom
