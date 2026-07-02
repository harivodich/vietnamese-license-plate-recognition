"""Tải dataset Kaggle theo quy trình có version, staging và khả năng tái lập."""

import argparse
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import kagglehub

from vlpr.config import load_config, project_root, resolve_project_path
from vlpr.data.receipt import RECEIPT_NAME, read_receipt, receipt_matches, write_receipt
from vlpr.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def _publish_staging(staging_dir: Path, target_dir: Path, *, force: bool) -> None:
    """Publish staging thành raw chính thức và khôi phục bản cũ nếu thao tác thay thế lỗi."""
    if not target_dir.exists():
        staging_dir.replace(target_dir)
        return
    if not force:
        raise FileExistsError(
            f"incomplete dataset directory already exists: {target_dir}; "
            "inspect it and rerun with --force to replace it"
        )

    backup_dir = target_dir.with_name(f".{target_dir.name}.backup-{uuid4().hex}")
    target_dir.replace(backup_dir)
    try:
        staging_dir.replace(target_dir)
    except OSError:
        backup_dir.replace(target_dir)
        raise
    shutil.rmtree(backup_dir)


def _download_from_kaggle(versioned_handle: str, staging_dir: Path) -> Path:
    """Bọc KaggleHub để mọi dữ liệu luôn được tải vào staging do dự án quản lý."""
    return Path(
        kagglehub.dataset_download(
            versioned_handle,
            output_dir=str(staging_dir),
            force_download=True,
        )
    ).resolve()


def download_dataset(
    config_path: Path,
    dataset_name: str = "detection",
    force: bool = False,
) -> Path:
    """Tải một version dataset cố định, ghi receipt rồi publish nguyên tử vào raw."""
    config = load_config(config_path)
    dataset = config.dataset(dataset_name)
    root = project_root(config_path)
    output_dir = resolve_project_path(root, dataset.raw_dir)
    existing_receipt = read_receipt(output_dir)
    if receipt_matches(existing_receipt, dataset_name, dataset) and not force:
        LOGGER.info("Dataset already complete name=%s output=%s", dataset_name, output_dir)
        return output_dir
    if output_dir.exists() and not force:
        raise FileExistsError(
            f"incomplete or mismatched dataset directory already exists: {output_dir}; "
            "inspect it and rerun with --force to replace it"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.with_name(f".{output_dir.name}.tmp-{uuid4().hex}")
    staging_dir.mkdir()

    LOGGER.info(
        "Downloading dataset name=%s handle=%s staging=%s",
        dataset_name,
        dataset.versioned_handle,
        staging_dir,
    )
    try:
        resolved_download = _download_from_kaggle(dataset.versioned_handle, staging_dir)
        if not any(path.is_file() for path in staging_dir.rglob("*")):
            raise RuntimeError(f"Kaggle download produced no files: {staging_dir}")
        write_receipt(staging_dir, dataset_name, dataset, resolved_download)
        _publish_staging(staging_dir, output_dir, force=force)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    LOGGER.info("Dataset download completed receipt=%s", output_dir / RECEIPT_NAME)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    """Tạo parser cho các tham số config, tên dataset và tùy chọn tải lại."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Dataset YAML configuration.",
    )
    parser.add_argument(
        "--dataset",
        default="detection",
        help="Logical dataset name declared under the datasets config mapping.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and overwrite KaggleHub-managed output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy CLI tải dataset và chuyển exception dự kiến thành exit code 1."""
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        download_dataset(
            config_path=args.config,
            dataset_name=args.dataset,
            force=args.force,
        )
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("Dataset download failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
