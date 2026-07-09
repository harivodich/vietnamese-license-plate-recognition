"""Export line-level OCR data in PaddleOCR recognition format."""

import argparse
import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from vlpr.config import project_root, resolve_project_path
from vlpr.data.manifest_io import read_manifest
from vlpr.data.manifest_schema import OcrManifestRecord
from vlpr.data.ocr_layout import split_compact_crop
from vlpr.evaluation.ocr import normalize_plate_text
from vlpr.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


class OcrFinetuneDataConfig(BaseModel):
    """Input/output contract for reproducible PaddleOCR fine-tune data export."""

    model_config = ConfigDict(extra="forbid")

    manifest: Path
    dataset_root: Path
    output_dir: Path
    compact_aspect_ratio: float = Field(gt=0.0)
    split_search_start: float = Field(gt=0.0, lt=1.0)
    split_search_end: float = Field(gt=0.0, lt=1.0)
    extra_characters: str = ""


class OcrFinetuneConfig(BaseModel):
    """Strict top-level config for PaddleOCR fine-tune data preparation."""

    model_config = ConfigDict(extra="forbid")

    data: OcrFinetuneDataConfig


@dataclass(frozen=True)
class _LineSample:
    """One line image and its normalized text label."""

    image_path: Path
    label: str
    source_split: Literal["train", "validation", "test"]


def load_ocr_finetune_config(path: Path) -> OcrFinetuneConfig:
    """Load strict YAML so misspelled export settings fail early."""
    with path.open("r", encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"OCR fine-tune config root must be a mapping: {path}")
    return OcrFinetuneConfig.model_validate(raw)


def _ocr_records(path: Path) -> tuple[OcrManifestRecord, ...]:
    """Read OCR records only and reject an empty manifest."""
    records = tuple(
        record for record in read_manifest(path) if isinstance(record, OcrManifestRecord)
    )
    if not records:
        raise ValueError("OCR manifest has no OCR records")
    return records


def _source_image(dataset_root: Path, record: OcrManifestRecord) -> Path:
    """Resolve a source image while preventing path traversal outside dataset root."""
    image_path = (dataset_root / record.image_path).resolve()
    if not image_path.is_relative_to(dataset_root.resolve()):
        raise ValueError(f"OCR image escapes dataset root: {record.image_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"OCR image not found: {image_path}")
    return image_path


def _write_image(image: Image.Image, path: Path) -> None:
    """Write exported line crops as lossless PNG files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG", optimize=True)


def _split_name(record: OcrManifestRecord, suffix: str) -> str:
    """Build a deterministic filename for a split line crop."""
    return f"{record.sha256}_{suffix}.png"


def _export_record(
    record: OcrManifestRecord,
    *,
    dataset_root: Path,
    output_dir: Path,
    settings: OcrFinetuneDataConfig,
) -> tuple[_LineSample, ...]:
    """Export one manifest record as one wide line or two compact line samples."""
    if record.split not in {"train", "validation", "test"}:
        return ()

    split = record.split
    source = _source_image(dataset_root, record)
    with Image.open(source) as opened:
        image = opened.convert("RGB")

    target_dir = output_dir / "images" / split
    is_compact = record.width / record.height < settings.compact_aspect_ratio
    if not is_compact:
        image_path = target_dir / f"{record.sha256}.png"
        _write_image(image, image_path)
        return (_LineSample(image_path, normalize_plate_text(record.annotation.raw_text), split),)

    tokens = record.annotation.raw_text.split()
    if len(tokens) != 2:
        raise ValueError(f"compact label must contain exactly 2 tokens: {record.sample_id}")
    top, bottom = split_compact_crop(
        image,
        search_start=settings.split_search_start,
        search_end=settings.split_search_end,
    )
    top_path = target_dir / _split_name(record, "top")
    bottom_path = target_dir / _split_name(record, "bottom")
    _write_image(top, top_path)
    _write_image(bottom, bottom_path)
    return (
        _LineSample(top_path, normalize_plate_text(tokens[0]), split),
        _LineSample(bottom_path, normalize_plate_text(tokens[1]), split),
    )


def _write_label_file(output_dir: Path, name: str, samples: Sequence[_LineSample]) -> Path:
    """Write PaddleOCR relative-path tab label files."""
    path = output_dir / name
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for sample in samples:
            relative = sample.image_path.relative_to(output_dir).as_posix()
            stream.write(f"{relative}\t{sample.label}\n")
    return path


def export_ocr_finetune_data(config_path: Path) -> dict[str, object]:
    """Export train/val/test line crops and labels for PaddleOCR recognition fine-tuning."""
    config = load_ocr_finetune_config(config_path)
    root = project_root(config_path)
    manifest = resolve_project_path(root, config.data.manifest)
    dataset_root = resolve_project_path(root, config.data.dataset_root)
    output_dir = resolve_project_path(root, config.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_by_split: dict[str, list[_LineSample]] = defaultdict(list)
    for record in _ocr_records(manifest):
        for sample in _export_record(
            record,
            dataset_root=dataset_root,
            output_dir=output_dir,
            settings=config.data,
        ):
            samples_by_split[sample.source_split].append(sample)

    for split in ("train", "validation", "test"):
        if not samples_by_split[split]:
            raise ValueError(f"OCR fine-tune split is empty: {split}")

    _write_label_file(output_dir, "train_list.txt", samples_by_split["train"])
    _write_label_file(output_dir, "val_list.txt", samples_by_split["validation"])
    _write_label_file(output_dir, "test_list.txt", samples_by_split["test"])

    train_validation_labels = [
        sample.label for split in ("train", "validation") for sample in samples_by_split[split]
    ]
    characters = sorted(set("".join(train_validation_labels)) | set(config.data.extra_characters))
    test_characters = set("".join(sample.label for sample in samples_by_split["test"]))
    missing_test_characters = sorted(test_characters - set(characters))
    if missing_test_characters:
        raise ValueError(
            "test labels contain characters missing from train/validation charset: "
            + "".join(missing_test_characters)
        )

    (output_dir / "dict.txt").write_text(
        "".join(f"{character}\n" for character in characters),
        encoding="utf-8",
    )
    summary: dict[str, object] = {
        "characters": len(characters),
        "compact_aspect_ratio": config.data.compact_aspect_ratio,
        "test_samples": len(samples_by_split["test"]),
        "train_samples": len(samples_by_split["train"]),
        "validation_samples": len(samples_by_split["validation"]),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for fine-tune data export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ocr-finetune-paddleocr.yaml"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run export from CLI and convert expected failures to exit code 1."""
    configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        summary = export_ocr_finetune_data(args.config)
    except (OSError, ValueError) as exc:
        LOGGER.error("OCR fine-tune data export failed: %s", exc)
        return 1
    LOGGER.info("OCR fine-tune data exported: %s", summary)
    return 0
