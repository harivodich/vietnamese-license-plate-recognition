"""Chuẩn bị line-level OCR dataset từ crop biển wide và compact."""

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from vlpr.config import project_root, resolve_project_path
from vlpr.data.manifest_io import read_manifest
from vlpr.data.manifest_schema import OcrManifestRecord
from vlpr.data.ocr_layout import split_compact_crop
from vlpr.evaluation.ocr import normalize_plate_text
from vlpr.training.ocr_config import (
    OcrTrainingDataSettings,
    load_ocr_training_config,
)
from vlpr.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def _ocr_records(path: Path) -> tuple[OcrManifestRecord, ...]:
    """Lọc đúng OCR records đã có project split."""
    records = tuple(
        record for record in read_manifest(path) if isinstance(record, OcrManifestRecord)
    )
    if not records:
        raise ValueError("OCR manifest không có record")
    return records


def _source_image(dataset_root: Path, record: OcrManifestRecord) -> Path:
    """Resolve ảnh nguồn và chặn path thoát khỏi dataset root."""
    image_path = (dataset_root / record.image_path).resolve()
    if not image_path.is_relative_to(dataset_root.resolve()):
        raise ValueError(f"OCR image thoát khỏi dataset root: {record.image_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"không tìm thấy OCR image: {image_path}")
    return image_path


def _write_image(image: Image.Image, path: Path) -> None:
    """Ghi PNG lossless để không thêm JPEG artifact vào crop vốn đã nhỏ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG", optimize=True)


def _materialize_record(
    record: OcrManifestRecord,
    *,
    dataset_root: Path,
    output_dir: Path,
    settings: OcrTrainingDataSettings,
) -> tuple[tuple[Path, str], ...]:
    """Xuất một wide sample hoặc hai line samples có nhãn tương ứng."""
    if record.split not in {"train", "validation"}:
        return ()
    source = _source_image(dataset_root, record)
    split_dir = "train" if record.split == "train" else "validation"
    base_name = record.sha256
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    tokens = record.annotation.raw_text.split()
    if image.width / image.height < settings.compact_aspect_ratio:
        if not settings.include_compact:
            return ()
        if len(tokens) != 2:
            raise ValueError(f"compact label phải có đúng 2 token: {record.sample_id}")
        top, bottom = split_compact_crop(
            image,
            search_start=settings.split_search_start,
            search_end=settings.split_search_end,
        )
        top_path = output_dir / "images" / split_dir / f"{base_name}_top.png"
        bottom_path = output_dir / "images" / split_dir / f"{base_name}_bottom.png"
        _write_image(top, top_path)
        _write_image(bottom, bottom_path)
        return (
            (top_path, normalize_plate_text(tokens[0])),
            (bottom_path, normalize_plate_text(tokens[1])),
        )

    output_path = output_dir / "images" / split_dir / f"{base_name}.png"
    _write_image(image, output_path)
    return ((output_path, normalize_plate_text(record.annotation.raw_text)),)


def _write_labels(
    output_dir: Path,
    split: str,
    samples: list[tuple[Path, str]],
) -> Path:
    """Ghi PaddleOCR-compatible `relative_path<TAB>label` theo thứ tự ổn định."""
    label_path = output_dir / f"{split}.txt"
    with label_path.open("w", encoding="utf-8", newline="\n") as stream:
        for image_path, label in samples:
            relative_path = image_path.relative_to(output_dir).as_posix()
            stream.write(f"{relative_path}\t{label}\n")
    return label_path


def prepare_ocr_training_data(config_path: Path) -> dict[str, int]:
    """Materialize train/validation lines, charset và summary không chứa timestamp."""
    config = load_ocr_training_config(config_path)
    root = project_root(config_path)
    manifest = resolve_project_path(root, config.data.manifest)
    dataset_root = resolve_project_path(root, config.data.dataset_root)
    output_dir = resolve_project_path(root, config.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_by_split: dict[str, list[tuple[Path, str]]] = {
        "train": [],
        "validation": [],
    }
    for record in _ocr_records(manifest):
        record_samples = _materialize_record(
            record,
            dataset_root=dataset_root,
            output_dir=output_dir,
            settings=config.data,
        )
        if record.split in samples_by_split:
            samples_by_split[record.split].extend(record_samples)

    for split, split_samples in samples_by_split.items():
        if not split_samples:
            raise ValueError(f"OCR training split rỗng: {split}")
        _write_labels(output_dir, split, split_samples)

    characters = sorted(
        {
            character
            for samples in samples_by_split.values()
            for _, label in samples
            for character in label
        }
    )
    charset_path = output_dir / "charset.txt"
    charset_path.write_text("".join(f"{character}\n" for character in characters), encoding="utf-8")
    summary = {
        "train_samples": len(samples_by_split["train"]),
        "validation_samples": len(samples_by_split["validation"]),
        "characters": len(characters),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    """Tạo CLI dùng cùng config với trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ocr-crnn.yaml"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Chuẩn bị dữ liệu và chuyển lỗi dự kiến thành exit code 1."""
    configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        summary = prepare_ocr_training_data(args.config)
    except (OSError, ValueError) as exc:
        LOGGER.error("OCR training data preparation failed: %s", exc)
        return 1
    LOGGER.info("OCR training data prepared: %s", summary)
    return 0
