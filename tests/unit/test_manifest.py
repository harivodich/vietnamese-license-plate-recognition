"""Kiểm thử tạo manifest record từ ảnh và annotation nguồn."""

from pathlib import Path

import pytest
from PIL import Image

from vlpr.data.manifest import build_detection_record, build_ocr_record
from vlpr.data.ocr_parser import OcrLabel


def test_build_detection_record_composes_image_hash_and_annotations(tmp_path: Path) -> None:
    """Xác nhận builder detection kết nối đủ metadata, hash và YOLO parser."""
    image_path = tmp_path / "images" / "train" / "sample.jpg"
    label_path = tmp_path / "labels" / "train" / "sample.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 50), color="white").save(image_path)
    label_path.write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")

    record = build_detection_record(
        dataset_root=tmp_path,
        image_path=image_path,
        label_path=label_path,
        dataset_name="detection",
        source_split="train",
    )

    assert record.sample_id == "detection:images/train/sample.jpg"
    assert (record.width, record.height) == (100, 50)
    assert len(record.sha256) == 64
    assert len(record.perceptual_hash) == 16
    assert record.annotations[0].bbox.width == pytest.approx(0.2)


def test_build_ocr_record_preserves_relative_path_and_raw_text(tmp_path: Path) -> None:
    """Xác nhận builder OCR tìm đúng ảnh và giữ nguyên nội dung nhãn."""
    image_path = tmp_path / "imgs" / "val" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (120, 40), color="gray").save(image_path)
    label = OcrLabel(
        image_path=Path("imgs/val/sample.jpg"),
        text="60MĐ1 01835",
    )

    record = build_ocr_record(
        dataset_root=tmp_path,
        label=label,
        dataset_name="ocr",
        source_split="val",
    )

    assert record.sample_id == "ocr:imgs/val/sample.jpg"
    assert record.image_path == "imgs/val/sample.jpg"
    assert record.annotation.raw_text == "60MĐ1 01835"
    assert record.split is None


def test_build_detection_record_rejects_image_outside_dataset(tmp_path: Path) -> None:
    """Xác nhận manifest không thể tham chiếu ảnh ngoài dataset root."""
    dataset_root = tmp_path / "dataset"
    outside_image = tmp_path / "outside.jpg"
    label_path = dataset_root / "sample.txt"
    dataset_root.mkdir()
    Image.new("RGB", (10, 10), color="white").save(outside_image)
    label_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="ngoài dataset root"):
        build_detection_record(
            dataset_root=dataset_root,
            image_path=outside_image,
            label_path=label_path,
            dataset_name="detection",
            source_split="train",
        )
