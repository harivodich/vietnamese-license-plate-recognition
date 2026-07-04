"""Kiểm thử thống kê và duplicate findings từ manifest chuẩn hóa."""

import json
from pathlib import Path

import pytest

from vlpr.data.audit import audit_manifests, write_audit_report
from vlpr.data.manifest_io import write_manifest
from vlpr.data.manifest_schema import (
    DetectionManifestRecord,
    OcrAnnotation,
    OcrManifestRecord,
)


def _detection_record(
    name: str,
    *,
    source_split: str,
    sha256: str,
    perceptual_hash: str,
    annotation_count: int = 1,
) -> DetectionManifestRecord:
    """Tạo detection record với số annotation tùy chọn."""
    from vlpr.data.manifest_schema import DetectionAnnotation, YoloBox

    annotation = DetectionAnnotation(
        bbox=YoloBox(center_x=0.5, center_y=0.5, width=0.2, height=0.1)
    )
    return DetectionManifestRecord(
        sample_id=f"detection:{name}",
        dataset_name="detection",
        task="detection",
        image_path=f"images/{source_split}/{name}.jpg",
        source_split=source_split,
        width=100,
        height=50,
        sha256=sha256,
        perceptual_hash=perceptual_hash,
        annotations=(annotation,) * annotation_count,
    )


def _ocr_record(name: str, text: str) -> OcrManifestRecord:
    """Tạo OCR record tối thiểu cho thống kê text."""
    return OcrManifestRecord(
        sample_id=f"ocr:{name}",
        dataset_name="ocr",
        task="ocr",
        image_path=f"imgs/train/{name}.jpg",
        source_split="train",
        width=80,
        height=24,
        sha256=name[0] * 64,
        perceptual_hash=f"{int(name[-1]):016x}",
        annotation=OcrAnnotation(raw_text=text),
    )


def test_audit_manifests_computes_statistics_and_cross_split_duplicates(
    tmp_path: Path,
) -> None:
    """Xác nhận audit tính stats và đánh dấu duplicate chéo source split."""
    detection_path = tmp_path / "detection.jsonl"
    ocr_path = tmp_path / "ocr.jsonl"
    write_manifest(
        detection_path,
        [
            _detection_record(
                "a",
                source_split="train",
                sha256="a" * 64,
                perceptual_hash="0000000000000000",
            ),
            _detection_record(
                "b",
                source_split="test",
                sha256="a" * 64,
                perceptual_hash="0000000000000000",
                annotation_count=2,
            ),
            _detection_record(
                "c",
                source_split="val",
                sha256="c" * 64,
                perceptual_hash="0000000000000001",
            ),
        ],
    )
    write_manifest(
        ocr_path,
        [
            _ocr_record("a1", "30A 12345"),
            _ocr_record("b2", "60MĐ1 01835"),
        ],
    )

    report = audit_manifests(
        detection_path,
        ocr_path,
        near_duplicate_hamming_distance=1,
    )

    assert report.detection.record_count == 3
    assert report.detection.annotation_count == 4
    assert report.detection.multi_plate_images == 1
    assert report.detection.duplicates.exact_groups[0].crosses_source_splits
    assert not report.detection.duplicates.exact_groups[0].annotations_consistent
    assert len(report.detection.duplicates.near_pairs) == 2
    assert report.ocr.text_lengths.maximum == 11
    assert "Đ" in report.ocr.character_set


def test_write_audit_report_publishes_valid_utf8_json(tmp_path: Path) -> None:
    """Xác nhận báo cáo JSON giữ Unicode và đọc được bằng standard library."""
    detection_path = tmp_path / "detection.jsonl"
    ocr_path = tmp_path / "ocr.jsonl"
    write_manifest(
        detection_path,
        [
            _detection_record(
                "a",
                source_split="train",
                sha256="a" * 64,
                perceptual_hash="0" * 16,
            )
        ],
    )
    write_manifest(ocr_path, [_ocr_record("a1", "60MĐ1 01835")])
    report = audit_manifests(
        detection_path,
        ocr_path,
        near_duplicate_hamming_distance=1,
    )
    report_path = tmp_path / "audit.json"

    write_audit_report(report_path, report)
    decoded = json.loads(report_path.read_text(encoding="utf-8"))

    assert decoded["ocr"]["character_set"].endswith("Đ")
    assert list(tmp_path.glob(".audit.json.tmp-*")) == []


def test_audit_rejects_manifest_with_wrong_task(tmp_path: Path) -> None:
    """Xác nhận detection input không được chứa OCR record."""
    detection_path = tmp_path / "detection.jsonl"
    ocr_path = tmp_path / "ocr.jsonl"
    write_manifest(detection_path, [_ocr_record("a1", "30A 12345")])
    write_manifest(ocr_path, [_ocr_record("b2", "60A 12345")])

    with pytest.raises(ValueError, match="không thuộc task detection"):
        audit_manifests(
            detection_path,
            ocr_path,
            near_duplicate_hamming_distance=1,
        )
