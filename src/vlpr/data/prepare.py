"""Kiểm tra điều kiện Gate 1 trước khi Gate 2 triển khai tạo manifest."""

from collections.abc import Sequence

from vlpr.data.source_status import run_source_check


def main(argv: Sequence[str] | None = None) -> int:
    """Xác nhận mọi raw source sẵn sàng; hiện chưa tạo manifest hoặc processed data."""
    return run_source_check(argv, __doc__ or "Check raw sources")
