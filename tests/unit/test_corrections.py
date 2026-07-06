"""Kiểm thử lớp correction bảo vệ dữ liệu raw và sửa manifest OCR."""

from pathlib import Path

import pytest

from vlpr.data.corrections import (
    CorrectionError,
    OcrCorrection,
    apply_ocr_corrections,
    read_ocr_corrections,
)
from vlpr.data.ocr_parser import OcrLabel


def _correction(**overrides: str) -> OcrCorrection:
    """Tạo correction hợp lệ và cho phép test ghi đè từng trường."""
    values = {
        "image_path": "imgs/train/car_352.jpg",
        "original_text": "30A 31588",
        "corrected_text": "30A 34588",
        "reason": "Nhãn nguồn sai một ký tự.",
        "review_method": "visual_review",
    }
    values.update(overrides)
    return OcrCorrection.model_validate(values)


def test_apply_ocr_corrections_changes_only_matching_label() -> None:
    """Xác nhận correction thay text nhưng giữ đường dẫn và thứ tự."""
    labels = (
        OcrLabel(image_path=Path("imgs/train/other.jpg"), text="29A 12345"),
        OcrLabel(image_path=Path("imgs/train/car_352.jpg"), text="30A 31588"),
    )

    result = apply_ocr_corrections(
        labels,
        {"imgs/train/car_352.jpg": _correction()},
    )

    assert [label.text for label in result] == ["29A 12345", "30A 34588"]


def test_apply_ocr_corrections_rejects_stale_original_text() -> None:
    """Không sửa âm thầm nếu dataset nguồn đã thay nhãn."""
    labels = (OcrLabel(image_path=Path("imgs/train/car_352.jpg"), text="30A 99999"),)

    with pytest.raises(CorrectionError, match="nguồn hiện tại"):
        apply_ocr_corrections(
            labels,
            {"imgs/train/car_352.jpg": _correction()},
        )


def test_apply_ocr_corrections_rejects_unused_entry() -> None:
    """Phát hiện correction gõ sai đường dẫn thay vì bỏ qua."""
    with pytest.raises(CorrectionError, match="không khớp ảnh nào"):
        apply_ocr_corrections((), {"imgs/train/car_352.jpg": _correction()})


def test_apply_ocr_corrections_excludes_incomplete_label() -> None:
    """Mẫu không đủ bằng chứng được loại thay vì đoán corrected text."""
    label = OcrLabel(image_path=Path("imgs/train/incomplete.jpg"), text="71578")
    correction = OcrCorrection(
        image_path="imgs/train/incomplete.jpg",
        original_text="71578",
        exclude=True,
        reason="Không đọc được tiền tố.",
        review_method="visual_review",
    )

    assert (
        apply_ocr_corrections(
            (label,),
            {"imgs/train/incomplete.jpg": correction},
        )
        == ()
    )


def test_read_ocr_corrections_rejects_duplicate_path(tmp_path: Path) -> None:
    """Mỗi ảnh chỉ được có một quyết định correction."""
    path = tmp_path / "corrections.jsonl"
    line = _correction().model_dump_json()
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(CorrectionError, match="bị trùng"):
        read_ocr_corrections(path)
