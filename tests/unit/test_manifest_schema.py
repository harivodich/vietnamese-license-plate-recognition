"""Kiểm thử schema annotation dùng trong detection manifest."""

import pytest
from pydantic import TypeAdapter, ValidationError

from vlpr.data.manifest_schema import (
    DetectionAnnotation,
    DetectionManifestRecord,
    ManifestRecord,
    OcrAnnotation,
    OcrManifestRecord,
    YoloBox,
)


def test_detection_annotation_accepts_real_sample() -> None:
    """Xác nhận schema chấp nhận bounding box thật của ảnh boderngoaigiao0."""
    annotation = DetectionAnnotation(
        class_id=0,
        bbox=YoloBox(
            center_x=0.541016,
            center_y=0.484375,
            width=0.097656,
            height=0.03125,
        ),
    )

    assert annotation.class_name == "license_plate"
    assert annotation.bbox.width == pytest.approx(0.097656)


def test_yolo_box_rejects_coordinate_outside_normalized_range() -> None:
    """Xác nhận tọa độ lớn hơn 1 bị Pydantic từ chối."""
    with pytest.raises(ValidationError):
        YoloBox(center_x=1.2, center_y=0.5, width=0.1, height=0.1)


def test_yolo_box_rejects_edge_outside_image() -> None:
    """Xác nhận tâm hợp lệ vẫn bị từ chối nếu kích thước làm bbox tràn khỏi ảnh."""
    with pytest.raises(ValidationError, match="chiều ngang"):
        YoloBox(center_x=0.05, center_y=0.5, width=0.2, height=0.1)


def test_yolo_box_accepts_six_decimal_boundary_rounding() -> None:
    """Xác nhận sai số làm tròn sáu chữ số không tạo lỗi vượt biên giả."""
    box = YoloBox(
        center_x=0.436667,
        center_y=0.954082,
        width=0.176667,
        height=0.091837,
    )

    assert box.center_y + box.height / 2 == pytest.approx(1.0000005)


def test_detection_annotation_rejects_unknown_class() -> None:
    """Xác nhận project một class không chấp nhận class id ngoài 0."""
    with pytest.raises(ValidationError):
        DetectionAnnotation(
            class_id=1,
            bbox=YoloBox(center_x=0.5, center_y=0.5, width=0.1, height=0.1),
        )


def test_detection_manifest_record_accepts_multiple_annotations() -> None:
    """Xác nhận record detection chứa metadata chung và nhiều bbox."""
    record = DetectionManifestRecord(
        sample_id="detection:a",
        dataset_name="vietnamese-license-plate-detection",
        task="detection",
        image_path="images/train/a.jpg",
        source_split="train",
        width=1280,
        height=720,
        sha256="a" * 64,
        perceptual_hash="b" * 16,
        annotations=(
            DetectionAnnotation(
                bbox=YoloBox(
                    center_x=0.5,
                    center_y=0.5,
                    width=0.2,
                    height=0.1,
                )
            ),
        ),
    )

    assert record.task == "detection"
    assert record.split is None
    assert len(record.annotations) == 1


def test_ocr_manifest_record_preserves_raw_text() -> None:
    """Xác nhận record OCR giữ nguyên khoảng trắng và ký tự tiếng Việt."""
    record = OcrManifestRecord(
        sample_id="ocr:a",
        dataset_name="vietnamese-license-plate-ocr",
        task="ocr",
        image_path="imgs/train/a.jpg",
        source_split="train",
        width=160,
        height=48,
        sha256="c" * 64,
        perceptual_hash="d" * 16,
        annotation=OcrAnnotation(raw_text="60MĐ1 01835"),
    )

    assert record.annotation.raw_text == "60MĐ1 01835"
    assert record.validation_status == "valid"


def test_manifest_union_uses_task_as_discriminator() -> None:
    """Xác nhận task chọn đúng schema khi đọc dữ liệu JSON."""
    record: ManifestRecord = TypeAdapter(ManifestRecord).validate_python(
        {
            "sample_id": "ocr:a",
            "dataset_name": "ocr-source",
            "task": "ocr",
            "image_path": "imgs/val/a.jpg",
            "source_split": "val",
            "width": 120,
            "height": 40,
            "sha256": "e" * 64,
            "perceptual_hash": "f" * 16,
            "annotation": {"raw_text": "51G 46455"},
        }
    )

    assert isinstance(record, OcrManifestRecord)


@pytest.mark.parametrize(
    "overrides",
    [
        {"image_path": "../outside.jpg"},
        {"image_path": r"images\train\a.jpg"},
        {"width": 0},
        {"sha256": "not-a-sha256"},
        {"perceptual_hash": "short"},
    ],
)
def test_manifest_record_rejects_invalid_metadata(overrides: dict[str, object]) -> None:
    """Xác nhận metadata không an toàn hoặc sai cấu trúc bị từ chối."""
    values: dict[str, object] = {
        "sample_id": "detection:a",
        "dataset_name": "detection-source",
        "task": "detection",
        "image_path": "images/train/a.jpg",
        "source_split": "train",
        "width": 640,
        "height": 480,
        "sha256": "a" * 64,
        "perceptual_hash": "b" * 16,
        "annotations": (),
    }
    values.update(overrides)

    with pytest.raises(ValidationError):
        DetectionManifestRecord.model_validate(values)


def test_ocr_annotation_rejects_blank_text() -> None:
    """Xác nhận nhãn OCR chỉ chứa khoảng trắng không vào được manifest."""
    with pytest.raises(ValidationError, match="raw_text không được rỗng"):
        OcrAnnotation(raw_text="   ")
