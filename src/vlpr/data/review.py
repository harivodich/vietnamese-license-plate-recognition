"""Chọn mẫu ưu tiên và tạo visualization phục vụ kiểm tra thủ công."""

import logging
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict

from vlpr.config import load_config, project_root, resolve_project_path
from vlpr.data.audit import DatasetAuditReport, DuplicateAudit
from vlpr.data.manifest_io import read_manifest
from vlpr.data.manifest_schema import (
    DetectionManifestRecord,
    ManifestRecord,
    OcrManifestRecord,
)
from vlpr.data.source_status import build_parser, find_unready_sources
from vlpr.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)
_DETECTION_DIRECTORY = "License Plate Detection Dataset"
_OCR_DIRECTORY = "lp_ocr_dataset_vi"


@dataclass(frozen=True, slots=True)
class ReviewSelection:
    """Ghép manifest record được chọn với lý do cần review."""

    record: ManifestRecord
    reasons: tuple[str, ...]


class ReviewQueueItem(BaseModel):
    """Mô tả một visualization đang chờ con người kiểm tra."""

    model_config = ConfigDict(extra="forbid")

    task: str
    sample_id: str
    source_split: str
    image_path: str
    visualization_path: str
    reasons: tuple[str, ...]
    raw_text: str | None = None
    bbox_count: int | None = None
    status: str = "pending"


def select_review_records(
    records: tuple[ManifestRecord, ...],
    *,
    sample_size: int,
    priority_paths: tuple[str, ...],
    reasons_by_path: Mapping[str, tuple[str, ...]],
    seed: int,
) -> tuple[ReviewSelection, ...]:
    """Chọn priority records trước rồi random-fill ổn định tới sample_size."""
    records_by_path = {record.image_path: record for record in records}
    selected_paths: list[str] = []
    selected_set: set[str] = set()

    for path in priority_paths:
        if path in records_by_path and path not in selected_set:
            selected_paths.append(path)
            selected_set.add(path)
            if len(selected_paths) == sample_size:
                break

    remaining = sorted(records_by_path.keys() - selected_set)
    randomizer = random.Random(seed)
    randomizer.shuffle(remaining)
    selected_paths.extend(remaining[: max(0, sample_size - len(selected_paths))])

    return tuple(
        ReviewSelection(
            record=records_by_path[path],
            reasons=reasons_by_path.get(path, ("random_sample",)),
        )
        for path in selected_paths
    )


def render_detection_review(
    image_path: Path,
    record: DetectionManifestRecord,
    output_path: Path,
) -> None:
    """Vẽ mọi bbox detection lên ảnh gốc và lưu visualization JPEG."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, min(image.size) // 200)
    for index, annotation in enumerate(record.annotations, start=1):
        bbox = annotation.bbox
        left = max(0, round((bbox.center_x - bbox.width / 2) * image.width))
        top = max(0, round((bbox.center_y - bbox.height / 2) * image.height))
        right = min(image.width - 1, round((bbox.center_x + bbox.width / 2) * image.width))
        bottom = min(
            image.height - 1,
            round((bbox.center_y + bbox.height / 2) * image.height),
        )
        draw.rectangle((left, top, right, bottom), outline="red", width=line_width)
        draw.text((left, max(0, top - 14)), str(index), fill="yellow")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=95)


def render_ocr_review(
    image_path: Path,
    record: OcrManifestRecord,
    output_path: Path,
) -> None:
    """Phóng crop OCR và vẽ raw label trên dải trắng bên dưới."""
    with Image.open(image_path) as source:
        crop = source.convert("RGB")
    scale = max(1, min(4, 320 // max(1, crop.width)))
    resized = crop.resize(
        (crop.width * scale, crop.height * scale),
        Image.Resampling.NEAREST,
    )
    label_height = 32
    canvas = Image.new("RGB", (max(resized.width, 180), resized.height + label_height), "white")
    canvas.paste(resized, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    draw.text((4, resized.height + 4), record.annotation.raw_text, fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=95)


def create_review_samples(config_path: Path) -> Path:
    """Chọn mẫu, render visualization và ghi review queue JSONL."""
    config = load_config(config_path)
    root = project_root(config_path)
    detection_config = config.dataset("detection")
    ocr_config = config.dataset("ocr")
    detection_manifest = resolve_project_path(root, detection_config.manifest_path)
    ocr_manifest = resolve_project_path(root, ocr_config.manifest_path)
    audit_path = resolve_project_path(root, config.validation.report_path)
    audit = DatasetAuditReport.model_validate_json(audit_path.read_text(encoding="utf-8"))

    detection_records = read_manifest(detection_manifest)
    ocr_records = read_manifest(ocr_manifest)
    detection_priority, detection_reasons = _priority_paths(audit.detection.duplicates)
    ocr_priority, ocr_reasons = _priority_paths(audit.ocr.duplicates)
    sample_size = config.validation.manual_review_sample_size

    detection_selection = select_review_records(
        detection_records,
        sample_size=sample_size,
        priority_paths=detection_priority,
        reasons_by_path=detection_reasons,
        seed=config.split.seed,
    )
    ocr_selection = select_review_records(
        ocr_records,
        sample_size=sample_size,
        priority_paths=ocr_priority,
        reasons_by_path=ocr_reasons,
        seed=config.split.seed + 1,
    )

    run_id = (
        f"{audit.detection.manifest_sha256[:8]}-{audit.ocr.manifest_sha256[:8]}-{config.split.seed}"
    )
    review_root = resolve_project_path(root, config.validation.review_dir) / run_id
    detection_root = resolve_project_path(root, detection_config.raw_dir) / _DETECTION_DIRECTORY
    ocr_root = resolve_project_path(root, ocr_config.raw_dir) / _OCR_DIRECTORY
    queue: list[ReviewQueueItem] = []

    for index, selection in enumerate(detection_selection, start=1):
        record = selection.record
        if not isinstance(record, DetectionManifestRecord):
            raise ValueError("detection review selection chứa record sai task")
        output_path = review_root / "detection" / f"{index:03d}_{Path(record.image_path).stem}.jpg"
        render_detection_review(detection_root / record.image_path, record, output_path)
        queue.append(_queue_item(root, record, output_path, selection.reasons))

    for index, selection in enumerate(ocr_selection, start=1):
        record = selection.record
        if not isinstance(record, OcrManifestRecord):
            raise ValueError("OCR review selection chứa record sai task")
        output_path = review_root / "ocr" / f"{index:03d}_{Path(record.image_path).stem}.jpg"
        render_ocr_review(ocr_root / record.image_path, record, output_path)
        queue.append(_queue_item(root, record, output_path, selection.reasons))

    _write_review_queue(review_root / "review_queue.jsonl", queue)
    return review_root


def _priority_paths(
    duplicates: DuplicateAudit,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Xếp conflict, exact cross-split rồi near cross-split theo distance."""
    ordered: list[str] = []
    reasons: dict[str, list[str]] = {}

    def add(path: str, reason: str) -> None:
        """Thêm reason và giữ thứ tự xuất hiện đầu tiên của path."""
        if path not in reasons:
            ordered.append(path)
            reasons[path] = []
        if reason not in reasons[path]:
            reasons[path].append(reason)

    for group in duplicates.exact_groups:
        if not group.annotations_consistent:
            for path in group.paths:
                add(path, "annotation_conflict")
    for group in duplicates.exact_groups:
        if group.crosses_source_splits:
            for path in group.paths:
                add(path, "exact_duplicate_cross_split")
    for pair in sorted(
        (pair for pair in duplicates.near_pairs if pair.crosses_source_splits),
        key=lambda pair: (pair.distance, pair.first, pair.second),
    ):
        add(pair.first, "near_duplicate_cross_split")
        add(pair.second, "near_duplicate_cross_split")

    return tuple(ordered), {path: tuple(values) for path, values in reasons.items()}


def _queue_item(
    project_root_path: Path,
    record: DetectionManifestRecord | OcrManifestRecord,
    output_path: Path,
    reasons: tuple[str, ...],
) -> ReviewQueueItem:
    """Chuyển typed manifest record thành review queue item."""
    return ReviewQueueItem(
        task=record.task,
        sample_id=record.sample_id,
        source_split=record.source_split,
        image_path=record.image_path,
        visualization_path=output_path.relative_to(project_root_path).as_posix(),
        reasons=reasons,
        raw_text=record.annotation.raw_text if isinstance(record, OcrManifestRecord) else None,
        bbox_count=len(record.annotations) if isinstance(record, DetectionManifestRecord) else None,
    )


def _write_review_queue(path: Path, items: list[ReviewQueueItem]) -> None:
    """Ghi queue JSONL nguyên tử mà không tạo review decisions giả."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            for item in items:
                stream.write(item.model_dump_json())
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Kiểm tra source readiness rồi tạo review visualization và queue."""
    configure_logging()
    args = build_parser(__doc__ or "Create manual review samples").parse_args(argv)
    try:
        unready = find_unready_sources(args.config)
        if unready:
            raise RuntimeError(f"raw sources chưa sẵn sàng: {', '.join(unready)}")
        review_root = create_review_samples(args.config)
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("Manual review generation failed: %s", exc)
        return 1
    LOGGER.info("Manual review samples created path=%s", review_root)
    return 0
