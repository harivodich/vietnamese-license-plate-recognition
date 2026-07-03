"""Kiểm thử parser chuyển dòng hoặc file nhãn OCR thành dữ liệu có kiểu."""

from pathlib import Path

import pytest

from vlpr.data.ocr_parser import OcrLabelParseError, parse_ocr_file, parse_ocr_line


def test_parse_ocr_line_preserves_plate_text_and_unicode() -> None:
    """Xác nhận parser giữ khoảng trắng nội bộ và ký tự tiếng Việt trong nhãn."""
    label = parse_ocr_line("imgs/train/type3_513.jpg\t60MĐ1 01835")

    assert label.image_path == Path("imgs/train/type3_513.jpg")
    assert label.text == "60MĐ1 01835"


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("imgs/train/car_1.jpg 30A 12345", "cần đúng 2 field phân cách bằng TAB"),
        ("imgs/train/car_1.jpg\t30A 12345\textra", "cần đúng 2 field phân cách bằng TAB"),
        ("\t30A 12345", "đường dẫn ảnh không được rỗng"),
        ("imgs/train/car_1.jpg\t   ", "nhãn OCR không được rỗng"),
        ("../outside.jpg\t30A 12345", "đường dẫn ảnh phải nằm trong dataset"),
    ],
)
def test_parse_ocr_line_rejects_invalid_structure(line: str, message: str) -> None:
    """Xác nhận parser từ chối record không đủ an toàn để đưa vào manifest."""
    with pytest.raises(OcrLabelParseError, match=message):
        parse_ocr_line(line, source="train.txt", line_number=7)


def test_parse_ocr_file_preserves_order_and_reports_physical_line(tmp_path: Path) -> None:
    """Xác nhận parser file giữ thứ tự và báo đúng dòng vật lý sau dòng trắng."""
    label_path = tmp_path / "train.txt"
    label_path.write_text(
        "imgs/train/a.jpg\t30A 12345\n\nimgs/train/b.jpg\t   \n",
        encoding="utf-8",
    )

    with pytest.raises(OcrLabelParseError, match=r"train.txt:3:"):
        parse_ocr_file(label_path)


def test_parse_ocr_file_returns_all_non_empty_lines(tmp_path: Path) -> None:
    """Xác nhận file hợp lệ được parse thành tuple theo đúng thứ tự nguồn."""
    label_path = tmp_path / "train.txt"
    label_path.write_text(
        "imgs/train/a.jpg\t30A 12345\nimgs/train/b.jpg\t60MĐ1 01835\n",
        encoding="utf-8",
    )

    labels = parse_ocr_file(label_path)

    assert [label.image_path for label in labels] == [
        Path("imgs/train/a.jpg"),
        Path("imgs/train/b.jpg"),
    ]
