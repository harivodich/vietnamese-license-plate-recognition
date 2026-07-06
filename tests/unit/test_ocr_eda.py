"""Kiểm thử phép tính OCR EDA trước khi vẽ biểu đồ."""

from typing import Literal

from vlpr.data.manifest_schema import OcrAnnotation, OcrManifestRecord
from vlpr.data.ocr_eda import collect_ocr_eda


def _record(
    name: str,
    split: Literal["train", "validation", "test"],
    text: str,
    *,
    width: int = 100,
    height: int = 50,
) -> OcrManifestRecord:
    """Tạo OCR record có text, kích thước và split do test kiểm soát."""
    return OcrManifestRecord(
        sample_id=f"ocr:{name}",
        dataset_name="ocr",
        task="ocr",
        image_path=f"imgs/{name}.jpg",
        source_split="source",
        split=split,
        width=width,
        height=height,
        sha256=f"{len(name):064x}",
        perceptual_hash="0" * 16,
        annotation=OcrAnnotation(raw_text=text),
    )


def test_collect_ocr_eda_counts_text_geometry_and_splits() -> None:
    """Xác nhận OCR EDA đếm Unicode, pattern và crop aspect ratio."""
    records = (
        _record("a", "train", "30A 12345"),
        _record("b", "validation", "60MĐ1 01835", width=120, height=40),
        _record("c", "test", "29A 99999"),
    )

    data = collect_ocr_eda(records)

    assert data.image_counts == {"train": 1, "validation": 1, "test": 1}
    assert data.character_counts["Đ"] == 1
    assert data.normalized_length_counts == {8: 2, 10: 1}
    assert data.pattern_counts == {"DDL DDDDD": 2, "DDLLD DDDDD": 1}
    assert data.aspects_by_split["validation"] == (3.0,)
