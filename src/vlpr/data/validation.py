"""Audit thống kê và duplicate từ hai manifest đã chuẩn hóa."""

import logging
from collections.abc import Sequence
from pathlib import Path

from vlpr.config import load_config, project_root, resolve_project_path
from vlpr.data.audit import DatasetAuditReport, audit_manifests, write_audit_report
from vlpr.data.source_status import build_parser, find_unready_sources
from vlpr.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def validate_manifests(config_path: Path) -> tuple[DatasetAuditReport, Path]:
    """Audit hai manifest theo config và publish báo cáo JSON nguyên tử."""
    config = load_config(config_path)
    root = project_root(config_path)
    detection_manifest = resolve_project_path(
        root,
        config.dataset("detection").manifest_path,
    )
    ocr_manifest = resolve_project_path(
        root,
        config.dataset("ocr").manifest_path,
    )
    report_path = resolve_project_path(root, config.validation.report_path)
    report = audit_manifests(
        detection_manifest,
        ocr_manifest,
        near_duplicate_hamming_distance=config.validation.near_duplicate_hamming_distance,
    )
    write_audit_report(report_path, report)
    return report, report_path


def main(argv: Sequence[str] | None = None) -> int:
    """Kiểm tra source readiness, audit manifest và trả exit code cho CLI."""
    configure_logging()
    args = build_parser(__doc__ or "Validate dataset manifests").parse_args(argv)
    try:
        unready = find_unready_sources(args.config)
        if unready:
            raise RuntimeError(f"raw sources chưa sẵn sàng: {', '.join(unready)}")
        report, report_path = validate_manifests(args.config)
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("Dataset validation failed: %s", exc)
        return 1

    LOGGER.info(
        "Dataset audit completed report=%s detection=%d ocr=%d",
        report_path,
        report.detection.record_count,
        report.ocr.record_count,
    )
    return 0
