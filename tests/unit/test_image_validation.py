"""Kiểm thử decode ảnh dùng trong validation dữ liệu."""

from pathlib import Path

import pytest
from PIL import Image

from vlpr.data.image_validation import ImageDecodeError, decode_image


def test_decode_image_returns_metadata_after_loading_pixels(tmp_path: Path) -> None:
    """Xác nhận ảnh hợp lệ được decode và trả metadata có kiểu."""
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (12, 7), color="white").save(image_path)

    metadata = decode_image(image_path)

    assert (metadata.width, metadata.height) == (12, 7)
    assert metadata.format == "PNG"
    assert metadata.mode == "RGB"


def test_decode_image_rejects_corrupt_file(tmp_path: Path) -> None:
    """Xác nhận file có đuôi ảnh nhưng byte hỏng không được chấp nhận."""
    image_path = tmp_path / "broken.jpg"
    image_path.write_bytes(b"not-an-image")

    with pytest.raises(ImageDecodeError, match=r"broken\.jpg"):
        decode_image(image_path)


def test_decode_image_rejects_missing_file(tmp_path: Path) -> None:
    """Xác nhận ảnh bị thiếu được báo bằng cùng error contract."""
    with pytest.raises(ImageDecodeError, match=r"missing\.jpg"):
        decode_image(tmp_path / "missing.jpg")
