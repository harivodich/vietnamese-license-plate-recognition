"""Kiểm thử chuẩn hóa, edit distance và metric OCR baseline."""

from pathlib import Path
from typing import Literal

import pytest
from PIL import Image

from vlpr.data.manifest_schema import OcrAnnotation, OcrManifestRecord
from vlpr.evaluation.ocr import (
    OcrEvaluationConfig,
    OcrEvaluationInputs,
    _build_recognition_inputs,
    _match_results_to_records,
    _record_matches_geometry,
    levenshtein_distance,
    normalize_plate_text,
)


def _record(image_path: str, text: str) -> OcrManifestRecord:
    """Tạo OCR record nhỏ nhất cho unit test output matching."""
    return OcrManifestRecord(
        sample_id=f"ocr:{image_path}",
        dataset_name="ocr",
        image_path=image_path,
        source_split="train",
        width=100,
        height=50,
        sha256="a" * 64,
        perceptual_hash="b" * 16,
        group_id="sha256:" + "a" * 64,
        split="test",
        validation_status="valid",
        task="ocr",
        annotation=OcrAnnotation(raw_text=text),
    )


def _evaluation_config(
    geometry: Literal["all", "wide", "compact"] = "all",
    layout: Literal["original", "split_compact"] = "original",
) -> OcrEvaluationConfig:
    """Tạo config tối thiểu để test metric/filter không cần khởi tạo PaddleOCR."""
    return OcrEvaluationConfig(
        model_name="model",
        manifest=Path("manifest.jsonl"),
        dataset_root=Path("."),
        project=Path("artifacts"),
        name="baseline",
        split="test",
        geometry=geometry,
        layout=layout,
        device="cpu",
        batch_size=2,
        cpu_threads=1,
        enable_mkldnn=False,
        compact_aspect_ratio=1.5,
        failure_examples=1,
    )


def test_normalize_plate_text_handles_unicode_case_and_separators() -> None:
    """Normalizer phải giữ chữ Đ nhưng bỏ space/hyphen và chuẩn hóa full-width."""
    assert normalize_plate_text(" ６０đ-123.45 ") == "60Đ12345"


@pytest.mark.parametrize(
    ("reference", "hypothesis", "distance"),
    [
        ("ABC", "ABC", 0),
        ("ABC", "ADC", 1),
        ("ABC", "AB", 1),
        ("", "12", 2),
    ],
)
def test_levenshtein_distance(
    reference: str,
    hypothesis: str,
    distance: int,
) -> None:
    """Edit distance phải hỗ trợ match, replace, delete và chuỗi rỗng."""
    assert levenshtein_distance(reference, hypothesis) == distance
    assert levenshtein_distance(hypothesis, reference) == distance


def test_record_geometry_filter_separates_wide_and_compact() -> None:
    """Pretrained one-line OCR baseline chỉ nên score crop wide, không trộn crop compact."""
    wide = _record("wide.jpg", "30A 12345")
    compact = _record("compact.jpg", "30A 12345").model_copy(update={"width": 60, "height": 50})

    assert _record_matches_geometry(wide, _evaluation_config("all"))
    assert _record_matches_geometry(compact, _evaluation_config("all"))
    assert _record_matches_geometry(wide, _evaluation_config("wide"))
    assert not _record_matches_geometry(compact, _evaluation_config("wide"))
    assert not _record_matches_geometry(wide, _evaluation_config("compact"))
    assert _record_matches_geometry(compact, _evaluation_config("compact"))


def test_split_compact_layout_builds_two_line_inputs(tmp_path: Path) -> None:
    """Compact layout phải tạo top/bottom line crops để recognizer đọc từng dòng."""
    image_path = tmp_path / "compact.jpg"
    Image.new("RGB", (60, 50), "white").save(image_path)
    record = _record("compact.jpg", "30A 12345").model_copy(update={"width": 60, "height": 50})
    inputs = OcrEvaluationInputs(
        config=_evaluation_config(layout="split_compact"),
        manifest=tmp_path / "manifest.jsonl",
        dataset_root=tmp_path,
        records=(record,),
        image_paths=(image_path.resolve(),),
    )

    recognition_inputs = _build_recognition_inputs(inputs, tmp_path / "artifacts")

    assert [item.part for item in recognition_inputs] == ["top", "bottom"]
    assert all(item.path.is_file() for item in recognition_inputs)
    assert {item.record.sample_id for item in recognition_inputs} == {record.sample_id}


def test_match_results_uses_input_path_not_backend_order(tmp_path: Path) -> None:
    """Output bị đảo thứ tự vẫn phải ghép đúng ground truth bằng input_path."""
    first_path = (tmp_path / "first.jpg").resolve()
    second_path = (tmp_path / "second.jpg").resolve()
    records = (_record("first.jpg", "30A 12345"), _record("second.jpg", "51B 67890"))
    config = _evaluation_config()
    inputs = OcrEvaluationInputs(
        config=config,
        manifest=tmp_path / "manifest.jsonl",
        dataset_root=tmp_path,
        records=records,
        image_paths=(first_path, second_path),
    )
    raw_results = [
        {"input_path": str(second_path), "rec_text": "51B67890", "rec_score": 0.8},
        {"input_path": str(first_path), "rec_text": "30a-12345", "rec_score": 0.9},
    ]

    predictions = _match_results_to_records(inputs, raw_results)

    assert [prediction.sample_id for prediction in predictions] == [
        "ocr:first.jpg",
        "ocr:second.jpg",
    ]
    assert all(prediction.exact_match for prediction in predictions)
