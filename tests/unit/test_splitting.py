"""Kiểm thử phép chia dữ liệu tái lập và không rò exact duplicate."""

from vlpr.data.manifest_schema import OcrAnnotation, OcrManifestRecord
from vlpr.data.splitting import (
    SplitName,
    assign_splits,
    deduplicate_exact_records,
)


def _record(name: str, sha256: str) -> OcrManifestRecord:
    """Tạo OCR record tối thiểu với nội dung ảnh do test chỉ định."""
    return OcrManifestRecord(
        sample_id=f"ocr:{name}",
        dataset_name="ocr",
        task="ocr",
        image_path=f"imgs/{name}.jpg",
        source_split="train",
        width=100,
        height=50,
        sha256=sha256,
        perceptual_hash="0" * 16,
        annotation=OcrAnnotation(raw_text="30A 12345"),
    )


def _ratios() -> dict[SplitName, float]:
    """Trả tỷ lệ chia chuẩn dùng chung trong các test."""
    return {"train": 0.75, "validation": 0.125, "test": 0.125}


def test_assign_splits_keeps_exact_duplicates_together() -> None:
    """Hai record cùng nội dung ảnh phải nhận cùng group và split."""
    duplicate_sha = "a" * 64
    records = (
        _record("a", duplicate_sha),
        _record("b", duplicate_sha),
        *(_record(f"unique-{index}", f"{index:064x}") for index in range(20)),
    )

    result = assign_splits(records, ratios=_ratios(), seed=42)

    assert result[0].split == result[1].split
    assert result[0].group_id == result[1].group_id == f"sha256:{duplicate_sha}"


def test_assign_splits_is_reproducible_and_preserves_input_order() -> None:
    """Cùng seed tạo đúng một kết quả mà không đảo thứ tự manifest."""
    records = tuple(_record(str(index), f"{index:064x}") for index in range(40))

    first = assign_splits(records, ratios=_ratios(), seed=7)
    second = assign_splits(records, ratios=_ratios(), seed=7)

    assert first == second
    assert [record.sample_id for record in first] == [record.sample_id for record in records]
    assert {record.split for record in first} == {"train", "validation", "test"}


def test_deduplicate_exact_records_keeps_smallest_sample_id() -> None:
    """Exact duplicate chỉ đóng góp một record canonical cho training."""
    duplicate_sha = "a" * 64
    records = (
        _record("z-copy", duplicate_sha),
        _record("a-canonical", duplicate_sha),
        _record("unique", "b" * 64),
    )

    result = deduplicate_exact_records(records)

    assert [record.sample_id for record in result] == [
        "ocr:a-canonical",
        "ocr:unique",
    ]
