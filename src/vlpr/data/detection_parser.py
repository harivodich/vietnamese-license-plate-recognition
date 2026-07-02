"""Chuyển annotation YOLO dạng text thành schema detection có kiểm tra kiểu."""

from pydantic import ValidationError

from vlpr.data.manifest_schema import DetectionAnnotation, YoloBox


class AnnotationParseError(ValueError):
    """Báo một dòng annotation không thể chuyển thành dữ liệu detection hợp lệ."""


def parse_yolo_line(
    line: str,
    *,
    source: str = "<memory>",
    line_number: int = 1,
) -> DetectionAnnotation:
    """Parse một dòng YOLO gồm class id, tâm bbox, chiều rộng và chiều cao."""
    fields = line.strip().split()
    if len(fields) != 5:
        raise AnnotationParseError(
            f"{source}:{line_number}: cần đúng 5 field YOLO, nhận được {len(fields)}"
        )

    try:
        class_id = int(fields[0])
        center_x, center_y, width, height = (float(value) for value in fields[1:])
    except ValueError as exc:
        raise AnnotationParseError(
            f"{source}:{line_number}: annotation chứa giá trị không phải số"
        ) from exc

    try:
        return DetectionAnnotation(
            class_id=class_id,
            bbox=YoloBox(
                center_x=center_x,
                center_y=center_y,
                width=width,
                height=height,
            ),
        )
    except ValidationError as exc:
        raise AnnotationParseError(
            f"{source}:{line_number}: tọa độ hoặc class id không hợp lệ"
        ) from exc
