"""Kiểm thử đọc, ghi và publish manifest JSONL."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from vlpr.data.manifest_io import ManifestReadError, read_manifest, write_manifest
from vlpr.data.manifest_schema import (
    DetectionManifestRecord,
    ManifestRecord,
    OcrAnnotation,
    OcrManifestRecord,
)


def _detection_record() -> DetectionManifestRecord:
    """Tạo record detection tối thiểu cho test JSONL."""
    return DetectionManifestRecord(
        sample_id="detection:a",
        dataset_name="detection",
        task="detection",
        image_path="images/train/a.jpg",
        source_split="train",
        width=640,
        height=480,
        sha256="a" * 64,
        perceptual_hash="b" * 16,
        annotations=(),
    )


def _ocr_record() -> OcrManifestRecord:
    """Tạo record OCR có Unicode để kiểm tra encoding UTF-8."""
    return OcrManifestRecord(
        sample_id="ocr:a",
        dataset_name="ocr",
        task="ocr",
        image_path="imgs/val/a.jpg",
        source_split="val",
        width=120,
        height=40,
        sha256="c" * 64,
        perceptual_hash="d" * 16,
        annotation=OcrAnnotation(raw_text="60MĐ1 01835"),
    )


def test_manifest_round_trip_preserves_record_types_and_unicode(tmp_path: Path) -> None:
    """Xác nhận ghi rồi đọc lại giữ đúng task, kiểu record và raw text."""
    manifest_path = tmp_path / "manifest.jsonl"

    write_manifest(manifest_path, [_detection_record(), _ocr_record()])
    records = read_manifest(manifest_path)

    assert len(manifest_path.read_text(encoding="utf-8").splitlines()) == 2
    assert isinstance(records[0], DetectionManifestRecord)
    assert isinstance(records[1], OcrManifestRecord)
    assert records[1].annotation.raw_text == "60MĐ1 01835"


@pytest.mark.parametrize(
    ("content", "line_number"),
    [
        ('{"task":"unknown"}\n', 1),
        (_detection_record().model_dump_json() + "\n\n", 2),
    ],
)
def test_read_manifest_reports_invalid_line(
    tmp_path: Path,
    content: str,
    line_number: int,
) -> None:
    """Xác nhận lỗi JSON/schema và dòng rỗng đều kèm số dòng vật lý."""
    manifest_path = tmp_path / "broken.jsonl"
    manifest_path.write_text(content, encoding="utf-8")

    with pytest.raises(ManifestReadError, match=rf"broken\.jsonl:{line_number}:"):
        read_manifest(manifest_path)


def test_write_manifest_keeps_previous_file_when_generation_fails(tmp_path: Path) -> None:
    """Xác nhận lỗi giữa chừng không publish manifest chưa hoàn chỉnh."""
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("previous manifest\n", encoding="utf-8")

    def failing_records() -> Iterator[ManifestRecord]:
        """Mô phỏng generator lỗi sau khi đã ghi được record đầu tiên."""
        yield _detection_record()
        raise RuntimeError("generation failed")

    with pytest.raises(RuntimeError, match="generation failed"):
        write_manifest(manifest_path, failing_records())

    assert manifest_path.read_text(encoding="utf-8") == "previous manifest\n"
    assert list(tmp_path.glob(".manifest.jsonl.tmp-*")) == []
