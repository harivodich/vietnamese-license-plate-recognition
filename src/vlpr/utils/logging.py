"""Cấu hình logging thống nhất cho script và dịch vụ."""

import logging
import os


def configure_logging() -> None:
    """Thiết lập level và format log chung, lấy level từ biến môi trường nếu có."""
    level_name = os.getenv("VLPR_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
