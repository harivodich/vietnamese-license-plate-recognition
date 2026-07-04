"""Kiểm thử điều phối Dataset Audit từ config và manifest."""

import json
from pathlib import Path

import pytest

from vlpr.data.manifest_io import write_manifest
from vlpr.data.manifest_schema import (
    DetectionManifestRecord,
    OcrAnnotation,
    OcrManifestRecord,
)
from vlpr.data.validation import main, validate_manifests


def _write_config(root: Path) -> Path:
    """Tạo config tối thiểu cho hai manifest và audit report fixture."""
    config_path = root / "configs" / "dataset.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
datasets:
  detection:
    handle: owner/detection
    version: 1
    country: VN
    task: detection
    expected_license: MIT
    raw_dir: data/raw/detection/v1
    manifest_path: data/interim/detection.jsonl
  ocr:
    handle: owner/ocr
    version: 1
    country: VN
    task: ocr
    expected_license: MIT
    raw_dir: data/raw/ocr/v1
    manifest_path: data/interim/ocr.jsonl
validation:
  image_extensions: [".jpg"]
  near_duplicate_hamming_distance: 1
  manual_review_sample_size: 100
  report_path: data/interim/audit.json
split:
  train: 0.75
  validation: 0.125
  test: 0.125
  seed: 20260701
""".lstrip(),
        encoding="utf-8",
    )
    return config_path


def test_validate_manifests_writes_configured_report(tmp_path: Path) -> None:
    """Xác nhận validation đọc đúng manifest và publish report từ config."""
    config_path = _write_config(tmp_path)
    interim = tmp_path / "data" / "interim"
    detection = DetectionManifestRecord(
        sample_id="detection:a",
        dataset_name="detection",
        task="detection",
        image_path="images/train/a.jpg",
        source_split="train",
        width=100,
        height=50,
        sha256="a" * 64,
        perceptual_hash="0" * 16,
        annotations=(),
    )
    ocr = OcrManifestRecord(
        sample_id="ocr:a",
        dataset_name="ocr",
        task="ocr",
        image_path="imgs/train/a.jpg",
        source_split="train",
        width=80,
        height=24,
        sha256="b" * 64,
        perceptual_hash="1" * 16,
        annotation=OcrAnnotation(raw_text="30A 12345"),
    )
    write_manifest(interim / "detection.jsonl", [detection])
    write_manifest(interim / "ocr.jsonl", [ocr])

    report, report_path = validate_manifests(config_path)
    decoded = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_path == interim / "audit.json"
    assert report.detection.record_count == 1
    assert decoded["ocr"]["record_count"] == 1


def test_validation_main_stops_when_source_is_unready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Xác nhận CLI không audit manifest khi completion receipt chưa sẵn sàng."""
    monkeypatch.setattr(
        "vlpr.data.validation.find_unready_sources",
        lambda _: ("detection",),
    )

    assert main([]) == 1
