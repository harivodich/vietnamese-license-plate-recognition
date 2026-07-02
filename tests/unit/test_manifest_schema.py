"""Kiểm thử schema annotation dùng trong detection manifest."""

import pytest
from pydantic import ValidationError

from vlpr.data.manifest_schema import DetectionAnnotation, YoloBox


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


def test_detection_annotation_rejects_unknown_class() -> None:
    """Xác nhận project một class không chấp nhận class id ngoài 0."""
    with pytest.raises(ValidationError):
        DetectionAnnotation(
            class_id=1,
            bbox=YoloBox(center_x=0.5, center_y=0.5, width=0.1, height=0.1),
        )
