"""Kiểm tra các nguồn raw bất biến trong cấu hình đã sẵn sàng hay chưa."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from vlpr.config import load_config, project_root, resolve_project_path
from vlpr.data.receipt import read_receipt, receipt_matches
from vlpr.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def find_unready_sources(config_path: Path) -> tuple[str, ...]:
    """Trả về tên các dataset chưa có completion receipt khớp với cấu hình."""
    config = load_config(config_path)
    root = project_root(config_path)
    unready: list[str] = []
    for name, dataset in config.datasets.items():
        raw_dir = resolve_project_path(root, dataset.raw_dir)
        if not receipt_matches(read_receipt(raw_dir), name, dataset):
            unready.append(name)
    return tuple(sorted(unready))


def build_parser(description: str) -> argparse.ArgumentParser:
    """Tạo parser dùng chung cho các command cần kiểm tra raw source."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Dataset YAML configuration.",
    )
    return parser


def run_source_check(argv: Sequence[str] | None, description: str) -> int:
    """Chạy kiểm tra readiness Gate 1 và trả exit code khác 0 nếu còn thiếu nguồn."""
    configure_logging()
    args = build_parser(description).parse_args(argv)
    unready = find_unready_sources(args.config)
    if unready:
        LOGGER.error("Raw sources are not ready: %s", ", ".join(unready))
        return 1
    LOGGER.info("All configured raw sources have matching completion receipts")
    return 0
