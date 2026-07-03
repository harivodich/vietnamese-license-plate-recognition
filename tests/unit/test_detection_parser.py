"""Kiểm thử parser chuyển dòng hoặc file YOLO thành detection annotation."""

from pathlib import Path

import pytest

from vlpr.data.detection_parser import AnnotationParseError, parse_yolo_file, parse_yolo_line


def test_parse_yolo_file_preserves_annotation_order(tmp_path: Path) -> None:
    """Xác nhận parser file giữ thứ tự bbox và bỏ qua dòng trắng."""
    label_path = tmp_path / "sample.txt"
    label_path.write_text(
        "0 0.25 0.25 0.1 0.1\n\n0 0.75 0.75 0.2 0.2\n",
        encoding="utf-8",
    )

    annotations = parse_yolo_file(label_path)

    assert len(annotations) == 2
    assert annotations[0].bbox.center_x == pytest.approx(0.25)
    assert annotations[1].bbox.center_x == pytest.approx(0.75)


def test_parse_yolo_file_returns_empty_tuple_for_empty_label(tmp_path: Path) -> None:
    """Xác nhận label không có annotation được biểu diễn bằng tuple rỗng."""
    label_path = tmp_path / "empty.txt"
    label_path.write_text("", encoding="utf-8")

    assert parse_yolo_file(label_path) == ()


def test_parse_yolo_file_reports_failing_line_number(tmp_path: Path) -> None:
    """Xác nhận lỗi trong file chứa cả đường dẫn và số dòng gây lỗi."""
    label_path = tmp_path / "broken.txt"
    label_path.write_text(
        "0 0.25 0.25 0.1 0.1\n\n0 invalid 0.5 0.1 0.1\n",
        encoding="utf-8",
    )

    with pytest.raises(AnnotationParseError, match=r"broken.txt:3:"):
        parse_yolo_file(label_path)


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
