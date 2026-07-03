"""Chuyển dòng hoặc file nhãn OCR thành record có kiểu và giữ nguyên raw text."""

from dataclasses import dataclass
from pathlib import Path


class OcrLabelParseError(ValueError):
    """Báo một dòng nhãn OCR không đủ điều kiện để đưa vào manifest."""


@dataclass(frozen=True, slots=True)
class OcrLabel:
    """Ghép đường dẫn ảnh tương đối với nội dung biển số chưa chuẩn hóa."""

    image_path: Path
    text: str


def parse_ocr_file(path: Path) -> tuple[OcrLabel, ...]:
    """Đọc file nhãn OCR UTF-8 và parse mọi dòng không rỗng theo đúng thứ tự."""
    labels: list[OcrLabel] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        labels.append(
            parse_ocr_line(
                line,
                source=str(path),
                line_number=line_number,
            )
        )
    return tuple(labels)


def parse_ocr_line(
    line: str,
    *,
    source: str = "<memory>",
    line_number: int = 1,
) -> OcrLabel:
    """Parse một đường dẫn ảnh và raw OCR text phân cách bằng đúng một ký tự TAB."""
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) != 2:
        raise OcrLabelParseError(
            f"{source}:{line_number}: cần đúng 2 field phân cách bằng TAB, nhận được {len(fields)}"
        )

    raw_image_path, text = fields
    if not raw_image_path.strip():
        raise OcrLabelParseError(f"{source}:{line_number}: đường dẫn ảnh không được rỗng")
    if not text.strip():
        raise OcrLabelParseError(f"{source}:{line_number}: nhãn OCR không được rỗng")

    image_path = Path(raw_image_path)
    if image_path.is_absolute() or ".." in image_path.parts:
        raise OcrLabelParseError(f"{source}:{line_number}: đường dẫn ảnh phải nằm trong dataset")

    return OcrLabel(image_path=image_path, text=text)
