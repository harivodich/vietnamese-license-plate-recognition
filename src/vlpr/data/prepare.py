"""Tạo manifest detection và OCR từ hai nguồn raw đã có completion receipt."""

import logging
from collections.abc import Sequence
from pathlib import Path

from vlpr.config import load_config, project_root, resolve_project_path
from vlpr.data.manifest_io import write_manifest
from vlpr.data.manifest_sources import iter_detection_records, iter_ocr_records
from vlpr.data.source_status import build_parser, find_unready_sources
from vlpr.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def prepare_manifests(config_path: Path) -> dict[str, Path]:
    """Sinh và publish hai manifest tới đường dẫn đã khai báo trong config."""
    config = load_config(config_path)
    root = project_root(config_path)
    image_extensions = config.validation.image_extensions

    detection = config.dataset("detection")
    detection_raw = resolve_project_path(root, detection.raw_dir)
    detection_manifest = resolve_project_path(root, detection.manifest_path)
    write_manifest(
        detection_manifest,
        iter_detection_records(
            detection_raw,
            dataset_name="detection",
            image_extensions=image_extensions,
        ),
    )

    ocr = config.dataset("ocr")
    ocr_raw = resolve_project_path(root, ocr.raw_dir)
    ocr_manifest = resolve_project_path(root, ocr.manifest_path)
    write_manifest(
        ocr_manifest,
        iter_ocr_records(
            ocr_raw,
            dataset_name="ocr",
            image_extensions=image_extensions,
        ),
    )

    return {
        "detection": detection_manifest,
        "ocr": ocr_manifest,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Kiểm tra raw source, tạo manifest và chuyển lỗi dự kiến thành exit code 1."""
    configure_logging()
    args = build_parser(__doc__ or "Prepare dataset manifests").parse_args(argv)
    try:
        unready = find_unready_sources(args.config)
        if unready:
            raise RuntimeError(f"raw sources chưa sẵn sàng: {', '.join(unready)}")
        manifests = prepare_manifests(args.config)
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("Manifest preparation failed: %s", exc)
        return 1

    for dataset_name, manifest_path in manifests.items():
        LOGGER.info("Manifest prepared dataset=%s path=%s", dataset_name, manifest_path)
    return 0
