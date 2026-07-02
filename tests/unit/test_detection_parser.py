"""Kiểm thử parser chuyển một dòng YOLO thành detection annotation."""

import pytest

from vlpr.data.detection_parser import AnnotationParseError, parse_yolo_line


def test_parse_yolo_line_returns_annotation_for_real_sample() -> None:
    """Xác nhận parser đọc đúng dòng label thật của ảnh boderngoaigiao0."""
    annotation = parse_yolo_line("0 0.541016 0.484375 0.097656 0.031250")

    assert annotation.class_id == 0
    assert annotation.bbox.center_x == pytest.approx(0.541016)
    assert annotation.bbox.height == pytest.approx(0.03125)


def test_parse_yolo_line_rejects_wrong_field_count() -> None:
    """Xác nhận dòng thiếu field bị từ chối và có vị trí nguồn trong lỗi."""
    with pytest.raises(AnnotationParseError, match=r"label.txt:3: cần đúng 5 field"):
        parse_yolo_line("0 0.5 0.5 0.1", source="label.txt", line_number=3)


def test_parse_yolo_line_rejects_non_numeric_value() -> None:
    """Xác nhận tọa độ không phải số bị chuyển thành lỗi parser dễ hiểu."""
    with pytest.raises(AnnotationParseError, match="không phải số"):
        parse_yolo_line("0 center 0.5 0.1 0.1")


def test_parse_yolo_line_rejects_box_outside_image() -> None:
    """Xác nhận parser không bỏ qua lỗi bbox tràn khỏi vùng ảnh chuẩn hóa."""
    with pytest.raises(AnnotationParseError, match="tọa độ hoặc class id"):
        parse_yolo_line("0 0.05 0.5 0.2 0.1")
