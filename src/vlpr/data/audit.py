"""Tính thống kê và duplicate findings từ hai manifest đã chuẩn hóa."""

import json
import os
from pathlib import Path
from statistics import fmean, median
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from vlpr.data.duplicates import (
    ImageFingerprint,
    find_exact_duplicate_groups,
    find_near_duplicate_pairs,
)
from vlpr.data.hashing import sha256_file
from vlpr.data.manifest_io import read_manifest
from vlpr.data.manifest_schema import DetectionManifestRecord, OcrManifestRecord


class NumericSummary(BaseModel):
    """Tóm tắt một dãy số để report không phải lưu mọi giá trị."""

    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=1)
    minimum: float
    maximum: float
    mean: float
    median: float


class ExactDuplicateFinding(BaseModel):
    """Lưu một nhóm file có cùng SHA-256 và các source split liên quan."""

    model_config = ConfigDict(extra="forbid")

    paths: tuple[str, ...]
    source_splits: tuple[str, ...]
    crosses_source_splits: bool
    annotations_consistent: bool


class NearDuplicateFinding(BaseModel):
    """Lưu một cặp dHash gần nhau cùng khoảng cách và source split."""

    model_config = ConfigDict(extra="forbid")

    first: str
    second: str
    distance: int = Field(ge=0, le=64)
    first_source_split: str
    second_source_split: str
    crosses_source_splits: bool


class DuplicateAudit(BaseModel):
    """Gom toàn bộ exact và near-duplicate findings của một task."""

    model_config = ConfigDict(extra="forbid")

    exact_groups: tuple[ExactDuplicateFinding, ...]
    near_pairs: tuple[NearDuplicateFinding, ...]


class DetectionAudit(BaseModel):
    """Thống kê ảnh, bbox và duplicate của detection manifest."""

    model_config = ConfigDict(extra="forbid")

    record_count: int
    manifest_sha256: str
    widths: NumericSummary
    heights: NumericSummary
    annotation_count: int
    empty_annotation_images: int
    multi_plate_images: int
    bbox_widths: NumericSummary | None
    bbox_heights: NumericSummary | None
    bbox_areas: NumericSummary | None
    duplicates: DuplicateAudit


class OcrAudit(BaseModel):
    """Thống kê ảnh, text và duplicate của OCR manifest."""

    model_config = ConfigDict(extra="forbid")

    record_count: int
    manifest_sha256: str
    widths: NumericSummary
    heights: NumericSummary
    text_lengths: NumericSummary
    character_set: str
    duplicates: DuplicateAudit


class DatasetAuditReport(BaseModel):
    """Báo cáo audit có kiểu cho cả detection và OCR."""

    model_config = ConfigDict(extra="forbid")

    near_duplicate_hamming_distance: int = Field(ge=0, le=63)
    detection: DetectionAudit
    ocr: OcrAudit


def audit_manifests(
    detection_manifest: Path,
    ocr_manifest: Path,
    *,
    near_duplicate_hamming_distance: int,
) -> DatasetAuditReport:
    """Đọc hai manifest, xác nhận task và tính báo cáo audit đầy đủ."""
    detection_records = read_manifest(detection_manifest)
    ocr_records = read_manifest(ocr_manifest)
    if not all(isinstance(record, DetectionManifestRecord) for record in detection_records):
        raise ValueError("detection manifest chứa record không thuộc task detection")
    if not all(isinstance(record, OcrManifestRecord) for record in ocr_records):
        raise ValueError("OCR manifest chứa record không thuộc task OCR")

    typed_detection = tuple(
        record for record in detection_records if isinstance(record, DetectionManifestRecord)
    )
    typed_ocr = tuple(record for record in ocr_records if isinstance(record, OcrManifestRecord))
    if not typed_detection or not typed_ocr:
        raise ValueError("manifest không được rỗng")

    return DatasetAuditReport(
        near_duplicate_hamming_distance=near_duplicate_hamming_distance,
        detection=_audit_detection(
            typed_detection,
            detection_manifest,
            near_duplicate_hamming_distance,
        ),
        ocr=_audit_ocr(
            typed_ocr,
            ocr_manifest,
            near_duplicate_hamming_distance,
        ),
    )


def write_audit_report(path: Path, report: DatasetAuditReport) -> None:
    """Ghi báo cáo JSON qua file tạm rồi publish nguyên tử."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(report.model_dump_json(indent=2))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _audit_detection(
    records: tuple[DetectionManifestRecord, ...],
    manifest_path: Path,
    max_distance: int,
) -> DetectionAudit:
    """Tính thống kê riêng cho ảnh và bounding box detection."""
    annotations = [annotation for record in records for annotation in record.annotations]
    return DetectionAudit(
        record_count=len(records),
        manifest_sha256=sha256_file(manifest_path),
        widths=_numeric_summary([record.width for record in records]),
        heights=_numeric_summary([record.height for record in records]),
        annotation_count=len(annotations),
        empty_annotation_images=sum(not record.annotations for record in records),
        multi_plate_images=sum(len(record.annotations) > 1 for record in records),
        bbox_widths=_optional_numeric_summary(
            [annotation.bbox.width for annotation in annotations]
        ),
        bbox_heights=_optional_numeric_summary(
            [annotation.bbox.height for annotation in annotations]
        ),
        bbox_areas=_optional_numeric_summary(
            [annotation.bbox.width * annotation.bbox.height for annotation in annotations]
        ),
        duplicates=_audit_duplicates(
            records,
            max_distance,
            annotation_by_path={
                record.image_path: json.dumps(
                    [annotation.model_dump(mode="json") for annotation in record.annotations],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                for record in records
            },
        ),
    )


def _audit_ocr(
    records: tuple[OcrManifestRecord, ...],
    manifest_path: Path,
    max_distance: int,
) -> OcrAudit:
    """Tính thống kê riêng cho ảnh crop và raw OCR text."""
    texts = [record.annotation.raw_text for record in records]
    return OcrAudit(
        record_count=len(records),
        manifest_sha256=sha256_file(manifest_path),
        widths=_numeric_summary([record.width for record in records]),
        heights=_numeric_summary([record.height for record in records]),
        text_lengths=_numeric_summary([len(text) for text in texts]),
        character_set="".join(sorted({character for text in texts for character in text})),
        duplicates=_audit_duplicates(
            records,
            max_distance,
            annotation_by_path={
                record.image_path: record.annotation.raw_text for record in records
            },
        ),
    )


def _audit_duplicates(
    records: tuple[DetectionManifestRecord, ...] | tuple[OcrManifestRecord, ...],
    max_distance: int,
    *,
    annotation_by_path: dict[str, str],
) -> DuplicateAudit:
    """Đổi record thành fingerprint rồi tạo duplicate findings có split context."""
    fingerprints = tuple(
        ImageFingerprint(
            path=Path(record.image_path),
            sha256=record.sha256,
            perceptual_hash=record.perceptual_hash,
        )
        for record in records
    )
    split_by_path = {record.image_path: record.source_split for record in records}

    exact_findings: list[ExactDuplicateFinding] = []
    for group in find_exact_duplicate_groups(fingerprints):
        paths = tuple(path.as_posix() for path in group)
        source_splits = tuple(sorted({split_by_path[path] for path in paths}))
        exact_findings.append(
            ExactDuplicateFinding(
                paths=paths,
                source_splits=source_splits,
                crosses_source_splits=len(source_splits) > 1,
                annotations_consistent=len({annotation_by_path[path] for path in paths}) == 1,
            )
        )

    near_findings: list[NearDuplicateFinding] = []
    for pair in find_near_duplicate_pairs(fingerprints, max_distance=max_distance):
        first = pair.first.as_posix()
        second = pair.second.as_posix()
        first_split = split_by_path[first]
        second_split = split_by_path[second]
        near_findings.append(
            NearDuplicateFinding(
                first=first,
                second=second,
                distance=pair.distance,
                first_source_split=first_split,
                second_source_split=second_split,
                crosses_source_splits=first_split != second_split,
            )
        )

    return DuplicateAudit(
        exact_groups=tuple(exact_findings),
        near_pairs=tuple(near_findings),
    )


def _numeric_summary(values: list[int] | list[float]) -> NumericSummary:
    """Tính count, min, max, mean và median cho một dãy không rỗng."""
    if not values:
        raise ValueError("không thể tóm tắt dãy số rỗng")
    return NumericSummary(
        count=len(values),
        minimum=min(values),
        maximum=max(values),
        mean=fmean(values),
        median=median(values),
    )


def _optional_numeric_summary(values: list[float]) -> NumericSummary | None:
    """Trả None khi không có bbox, ngược lại trả thống kê số."""
    return _numeric_summary(values) if values else None
