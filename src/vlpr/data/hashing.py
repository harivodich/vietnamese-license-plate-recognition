"""Tính dấu vân tay nội dung file để phát hiện bản sao chính xác."""

import hashlib
from pathlib import Path

from PIL import Image

_HASH_CHUNK_SIZE = 1024 * 1024
_DIFFERENCE_HASH_SIZE = 8


def sha256_file(path: Path) -> str:
    """Đọc file theo từng khối và trả SHA-256 dạng 64 ký tự hexadecimal."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: Path) -> str:
    """Tạo dHash 64 bit từ quan hệ độ sáng giữa các pixel kề nhau."""
    with Image.open(path) as image:
        grayscale = image.convert("L")
        resized = grayscale.resize(
            (_DIFFERENCE_HASH_SIZE + 1, _DIFFERENCE_HASH_SIZE),
            Image.Resampling.LANCZOS,
        )
        pixels = list(resized.getdata())

    bits = 0
    for row in range(_DIFFERENCE_HASH_SIZE):
        offset = row * (_DIFFERENCE_HASH_SIZE + 1)
        for column in range(_DIFFERENCE_HASH_SIZE):
            bits = (bits << 1) | (pixels[offset + column] > pixels[offset + column + 1])
    return f"{bits:016x}"


def hamming_distance(first_hash: str, second_hash: str) -> int:
    """Đếm số bit khác nhau giữa hai perceptual hash cùng độ dài."""
    if len(first_hash) != len(second_hash):
        raise ValueError("perceptual hash phải có cùng độ dài")
    try:
        difference = int(first_hash, 16) ^ int(second_hash, 16)
    except ValueError as exc:
        raise ValueError("perceptual hash phải là chuỗi hexadecimal") from exc
    return difference.bit_count()
