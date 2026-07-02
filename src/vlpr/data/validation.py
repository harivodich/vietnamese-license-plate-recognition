"""Kiểm tra raw source trước khi Gate 2 triển khai validation đầy đủ."""

from collections.abc import Sequence

from vlpr.data.source_status import run_source_check


def main(argv: Sequence[str] | None = None) -> int:
    """Xác nhận receipt; hiện chưa kiểm tra ảnh hỏng, bbox hoặc OCR label."""
    return run_source_check(argv, __doc__ or "Check raw sources")
