"""Kiểm thử adapter tạo manifest từ cấu trúc hai nguồn dataset."""

from pathlib import Path

import pytest
from PIL import Image

from vlpr.data.corrections import OcrCorrection
from vlpr.data.manifest_sources import (
    DatasetStructureError,
    iter_detection_records,
    iter_ocr_records,
)

IMAGE_EXTENSIONS = (".jpg", ".png")


def _write_image(path: Path, size: tuple[int, int] = (20, 10)) -> None:
    """Tạo fixture ảnh nhỏ tại path bất kỳ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="white").save(path)


def _create_detection_split(
    raw_root: Path,
    split: str,
    stem: str,
) -> None:
    """Tạo một cặp ảnh-label detection hợp lệ."""
    dataset_root = raw_root / "License Plate Detection Dataset"
    image_path = dataset_root / "images" / split / f"{stem}.jpg"
    label_path = dataset_root / "labels" / split / f"{stem}.txt"
    _write_image(image_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def test_iter_detection_records_preserves_split_order(tmp_path: Path) -> None:
    """Xác nhận detection generator đi train, val, test theo thứ tự cố định."""
    for split in ("train", "val", "test"):
        _create_detection_split(tmp_path, split, f"{split}-sample")

    records = tuple(
        iter_detection_records(
            tmp_path,
            dataset_name="detection",
            image_extensions=IMAGE_EXTENSIONS,
        )
    )

    assert [record.source_split for record in records] == ["train", "val", "test"]
    assert all(len(record.annotations) == 1 for record in records)


def test_iter_detection_records_rejects_missing_label(tmp_path: Path) -> None:
    """Xác nhận ảnh detection thiếu label dừng generation trước khi publish."""
    dataset_root = tmp_path / "License Plate Detection Dataset"
    for split in ("train", "val", "test"):
        (dataset_root / "images" / split).mkdir(parents=True)
        (dataset_root / "labels" / split).mkdir(parents=True)
    _write_image(dataset_root / "images" / "train" / "orphan.jpg")

    with pytest.raises(DatasetStructureError, match="thiếu label"):
        tuple(
            iter_detection_records(
                tmp_path,
                dataset_name="detection",
                image_extensions=IMAGE_EXTENSIONS,
            )
        )


def _create_ocr_source(raw_root: Path) -> Path:
    """Tạo cấu trúc OCR gồm một ảnh train và một ảnh val."""
    dataset_root = raw_root / "lp_ocr_dataset_vi"
    _write_image(dataset_root / "imgs" / "train" / "train.jpg")
    _write_image(dataset_root / "imgs" / "val" / "val.jpg")
    labels_dir = dataset_root / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "train.txt").write_text(
        "imgs/train/train.jpg\t30A 12345\n",
        encoding="utf-8",
    )
    (labels_dir / "val.txt").write_text(
        "imgs/val/val.jpg\t60MĐ1 01835\n",
        encoding="utf-8",
    )
    return dataset_root


def test_iter_ocr_records_preserves_labels_and_order(tmp_path: Path) -> None:
    """Xác nhận OCR generator giữ split, path và raw text nguồn."""
    _create_ocr_source(tmp_path)

    records = tuple(
        iter_ocr_records(
            tmp_path,
            dataset_name="ocr",
            image_extensions=IMAGE_EXTENSIONS,
        )
    )

    assert [record.source_split for record in records] == ["train", "val"]
    assert [record.annotation.raw_text for record in records] == [
        "30A 12345",
        "60MĐ1 01835",
    ]


def test_iter_ocr_records_skips_excluded_correction(tmp_path: Path) -> None:
    """Correction exclude loại record nhưng vẫn cho phép ảnh tồn tại trong raw source."""
    _create_ocr_source(tmp_path)

    records = tuple(
        iter_ocr_records(
            tmp_path,
            dataset_name="ocr",
            image_extensions=(".jpg",),
            corrections={
                "imgs/train/train.jpg": OcrCorrection(
                    image_path="imgs/train/train.jpg",
                    original_text="30A 12345",
                    exclude=True,
                    reason="Nhãn không đầy đủ.",
                    review_method="visual_review",
                )
            },
        )
    )

    assert [record.source_split for record in records] == ["val"]
    assert [record.annotation.raw_text for record in records] == ["60MĐ1 01835"]


def test_iter_ocr_records_rejects_unreferenced_image(tmp_path: Path) -> None:
    """Xác nhận ảnh OCR không có dòng label bị báo lỗi cấu trúc."""
    dataset_root = _create_ocr_source(tmp_path)
    _write_image(dataset_root / "imgs" / "train" / "orphan.jpg")

    with pytest.raises(DatasetStructureError, match="ảnh không nhãn"):
        tuple(
            iter_ocr_records(
                tmp_path,
                dataset_name="ocr",
                image_extensions=IMAGE_EXTENSIONS,
            )
        )
