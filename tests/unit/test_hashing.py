"""Kiểm thử dấu vân tay SHA-256 của file dữ liệu."""

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from vlpr.data.hashing import difference_hash, hamming_distance, sha256_file


def test_sha256_file_matches_standard_library(tmp_path: Path) -> None:
    """Xác nhận hàm trả đúng SHA-256 cho nội dung đã biết."""
    path = tmp_path / "sample.bin"
    content = b"vietnamese-license-plate"
    path.write_bytes(content)

    assert sha256_file(path) == hashlib.sha256(content).hexdigest()


def test_sha256_file_returns_same_hash_for_identical_bytes(tmp_path: Path) -> None:
    """Xác nhận tên file khác nhau không ảnh hưởng hash của cùng nội dung."""
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"same-image-bytes")
    second.write_bytes(b"same-image-bytes")

    assert sha256_file(first) == sha256_file(second)


def test_sha256_file_changes_when_content_changes(tmp_path: Path) -> None:
    """Xác nhận thay đổi một byte làm dấu vân tay nội dung thay đổi."""
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"image-A")
    second.write_bytes(b"image-B")

    assert sha256_file(first) != sha256_file(second)


def test_difference_hash_ignores_image_encoding_for_same_pixels(tmp_path: Path) -> None:
    """Xác nhận cùng pixel lưu bằng PNG và BMP tạo cùng perceptual hash."""
    png_path = tmp_path / "sample.png"
    bmp_path = tmp_path / "sample.bmp"
    image = Image.new("L", (32, 32))
    image.putdata([(x * 8 + y) % 256 for y in range(32) for x in range(32)])
    image.save(png_path)
    image.save(bmp_path)

    assert sha256_file(png_path) != sha256_file(bmp_path)
    assert difference_hash(png_path) == difference_hash(bmp_path)


def test_hamming_distance_counts_visual_hash_differences(tmp_path: Path) -> None:
    """Xác nhận hai gradient ngược hướng tạo dHash khác nhau rõ rệt."""
    left_to_right = tmp_path / "left-to-right.png"
    right_to_left = tmp_path / "right-to-left.png"
    Image.linear_gradient("L").rotate(90, expand=True).save(left_to_right)
    Image.linear_gradient("L").rotate(270, expand=True).save(right_to_left)

    distance = hamming_distance(
        difference_hash(left_to_right),
        difference_hash(right_to_left),
    )

    assert distance > 0


def test_hamming_distance_rejects_invalid_hashes() -> None:
    """Xác nhận lỗi input được báo rõ thay vì tạo khoảng cách sai."""
    with pytest.raises(ValueError, match="cùng độ dài"):
        hamming_distance("0f", "000f")
    with pytest.raises(ValueError, match="hexadecimal"):
        hamming_distance("zz", "00")
