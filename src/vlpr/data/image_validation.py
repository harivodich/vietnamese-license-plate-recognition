"""Decode ảnh nguồn để phát hiện file hỏng và lấy metadata tối thiểu."""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


class ImageDecodeError(ValueError):
    """Báo một file không thể được decode thành ảnh hợp lệ."""


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    """Lưu kích thước và định dạng đọc được sau khi decode toàn bộ pixel."""

    width: int
    height: int
    format: str | None
    mode: str


def decode_image(path: Path) -> ImageMetadata:
    """Decode toàn bộ pixel thay vì chỉ tin phần mở rộng hoặc header của file."""
    try:
        with Image.open(path) as image:
            image.load()
            return ImageMetadata(
                width=image.width,
                height=image.height,
                format=image.format,
                mode=image.mode,
            )
    except (OSError, ValueError) as exc:
        raise ImageDecodeError(f"không thể decode ảnh: {path}") from exc
