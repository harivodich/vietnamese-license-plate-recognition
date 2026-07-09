"""Đánh giá OCR độc lập trên plate crops, không sử dụng detector."""

import argparse
import json
import logging
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from vlpr.config import project_root, resolve_project_path
from vlpr.data.manifest_io import read_manifest
from vlpr.data.manifest_schema import OcrManifestRecord
from vlpr.data.ocr_layout import split_compact_crop
from vlpr.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RecognitionInput:
    """One image sent to the recognizer, linked back to its source plate record."""

    path: Path
    record: OcrManifestRecord
    part: Literal["full", "top", "bottom"]


class OcrEvaluationConfig(BaseModel):
    """Đóng băng model, test split và tài nguyên dùng cho OCR baseline."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(default="")
    model_dir: str | None = None
    manifest: Path
    dataset_root: Path
    project: Path
    name: str = Field(min_length=1)
    split: Literal["test"]
    geometry: Literal["all", "wide", "compact"] = "all"
    layout: Literal["original", "split_compact"] = "original"
    split_search_start: float = Field(default=0.35, gt=0.0, lt=1.0)
    split_search_end: float = Field(default=0.65, gt=0.0, lt=1.0)
    device: str = Field(min_length=1)
    batch_size: int = Field(gt=0)
    cpu_threads: int = Field(gt=0)
    enable_mkldnn: bool
    compact_aspect_ratio: float = Field(gt=0.0)
    failure_examples: int = Field(ge=0)


class OcrEvaluationInputs(BaseModel):
    """Các path đã resolve và record test đã qua preflight."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    config: OcrEvaluationConfig
    manifest: Path
    dataset_root: Path
    records: tuple[OcrManifestRecord, ...]
    image_paths: tuple[Path, ...]


class OcrPrediction(BaseModel):
    """Một prediction cùng ground truth và edit distance sau chuẩn hóa."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    image_path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    ground_truth: str
    prediction: str
    normalized_ground_truth: str
    normalized_prediction: str
    confidence: float
    edit_distance: int = Field(ge=0)
    exact_match: bool


class OcrMetricGroup(BaseModel):
    """Metric OCR tổng hợp cho một nhóm hình học."""

    model_config = ConfigDict(extra="forbid")

    samples: int = Field(gt=0)
    exact_matches: int = Field(ge=0)
    exact_match_rate: float = Field(ge=0.0, le=1.0)
    character_count: int = Field(gt=0)
    edit_distance: int = Field(ge=0)
    cer: float = Field(ge=0.0)
    character_accuracy: float = Field(ge=0.0, le=1.0)


class OcrEvaluationResult(BaseModel):
    """Kết quả baseline đủ để so sánh model và subgroup sau này."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    split: Literal["test"]
    geometry: Literal["all", "wide", "compact"]
    layout: Literal["original", "split_compact"]
    metrics: OcrMetricGroup
    metrics_by_geometry: dict[str, OcrMetricGroup]
    mean_confidence: float
    model_initialization_seconds: float
    inference_ms_per_image: float
    failure_examples: tuple[OcrPrediction, ...]


def normalize_plate_text(text: str) -> str:
    """Chuẩn hóa Unicode/case và bỏ separator nhưng không sửa nhầm ký tự."""
    normalized = unicodedata.normalize("NFKC", text).upper()
    return "".join(character for character in normalized if character.isalnum())


def levenshtein_distance(reference: str, hypothesis: str) -> int:
    """Tính edit distance với bộ nhớ O(min(m, n))."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_character in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_character in enumerate(hypothesis, start=1):
            insertion = current[hypothesis_index - 1] + 1
            deletion = previous[hypothesis_index] + 1
            substitution = previous[hypothesis_index - 1] + (
                reference_character != hypothesis_character
            )
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def load_ocr_evaluation(path: Path) -> OcrEvaluationConfig:
    """Đọc strict YAML để key sai không âm thầm dùng default ngoài ý muốn."""
    with path.open("r", encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"OCR evaluation config root phải là mapping: {path}")
    return OcrEvaluationConfig.model_validate(raw)


def validate_ocr_evaluation(path: Path) -> OcrEvaluationInputs:
    """Kiểm tra toàn bộ test crop tồn tại trước khi khởi tạo pretrained model."""
    config = load_ocr_evaluation(path)
    root = project_root(path)
    manifest = resolve_project_path(root, config.manifest)
    dataset_root = resolve_project_path(root, config.dataset_root)
    if not manifest.is_file():
        raise FileNotFoundError(f"không tìm thấy OCR manifest: {manifest}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"không tìm thấy OCR dataset root: {dataset_root}")

    records = tuple(
        record
        for record in read_manifest(manifest)
        if isinstance(record, OcrManifestRecord)
        and record.split == config.split
        and _record_matches_geometry(record, config)
    )
    if not records:
        raise ValueError(f"OCR test split rỗng với geometry={config.geometry}")

    image_paths: list[Path] = []
    for record in records:
        image_path = (dataset_root / record.image_path).resolve()
        if not image_path.is_relative_to(dataset_root.resolve()):
            raise ValueError(f"OCR image thoát khỏi dataset root: {record.image_path}")
        if not image_path.is_file():
            raise FileNotFoundError(f"không tìm thấy OCR image: {image_path}")
        image_paths.append(image_path)
    return OcrEvaluationInputs(
        config=config,
        manifest=manifest,
        dataset_root=dataset_root,
        records=records,
        image_paths=tuple(image_paths),
    )


def _is_compact_record(record: OcrManifestRecord, compact_aspect_ratio: float) -> bool:
    """Classify plate crop geometry with the same reproducible aspect-ratio rule."""
    return record.width / record.height < compact_aspect_ratio


def _record_matches_geometry(
    record: OcrManifestRecord,
    config: OcrEvaluationConfig,
) -> bool:
    """Filter OCR crops by shape so one-line recognizers are not scored on two-line plates."""
    if config.geometry == "all":
        return True
    is_compact = _is_compact_record(record, config.compact_aspect_ratio)
    return is_compact if config.geometry == "compact" else not is_compact


def _write_line_crop(image: Image.Image, path: Path) -> None:
    """Write split line crops losslessly so recognizer input can be audited."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG", optimize=True)


def _build_recognition_inputs(
    inputs: OcrEvaluationInputs,
    output_dir: Path,
) -> tuple[_RecognitionInput, ...]:
    """Build recognizer inputs, optionally splitting compact plates into top/bottom lines."""
    config = inputs.config
    recognition_inputs: list[_RecognitionInput] = []
    line_crop_dir = output_dir / "line_crops"

    for record, image_path in zip(inputs.records, inputs.image_paths, strict=True):
        should_split = config.layout == "split_compact" and _is_compact_record(
            record,
            config.compact_aspect_ratio,
        )
        if not should_split:
            recognition_inputs.append(
                _RecognitionInput(path=image_path, record=record, part="full")
            )
            continue

        tokens = record.annotation.raw_text.split()
        if len(tokens) != 2:
            raise ValueError(f"compact label must contain exactly 2 tokens: {record.sample_id}")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        top, bottom = split_compact_crop(
            image,
            search_start=config.split_search_start,
            search_end=config.split_search_end,
        )
        top_path = line_crop_dir / f"{record.sha256}_top.png"
        bottom_path = line_crop_dir / f"{record.sha256}_bottom.png"
        _write_line_crop(top, top_path)
        _write_line_crop(bottom, bottom_path)
        recognition_inputs.extend(
            (
                _RecognitionInput(path=top_path, record=record, part="top"),
                _RecognitionInput(path=bottom_path, record=record, part="bottom"),
            )
        )

    return tuple(recognition_inputs)


def _result_by_path(raw_results: list[Any]) -> dict[Path, Any]:
    """Index recognizer output by resolved path and reject duplicate backend outputs."""
    result_by_path: dict[Path, Any] = {}
    for result in raw_results:
        result_path = Path(result["input_path"]).resolve()
        if result_path in result_by_path:
            raise ValueError(f"PaddleOCR returned duplicate image: {result_path}")
        result_by_path[result_path] = result
    return result_by_path


def _match_results_to_recognition_inputs(
    inputs: OcrEvaluationInputs,
    recognition_inputs: tuple[_RecognitionInput, ...],
    raw_results: list[Any],
) -> tuple[OcrPrediction, ...]:
    """Merge line-level recognizer outputs back into full-plate predictions."""
    remaining_results = _result_by_path(raw_results)
    results_by_sample: dict[str, list[tuple[_RecognitionInput, Any]]] = defaultdict(list)

    for recognition_input in recognition_inputs:
        result_path = recognition_input.path.resolve()
        try:
            result = remaining_results.pop(result_path)
        except KeyError as exc:
            raise ValueError(f"PaddleOCR missing output for image: {result_path}") from exc
        results_by_sample[recognition_input.record.sample_id].append((recognition_input, result))

    if remaining_results:
        raise ValueError(f"PaddleOCR returned {len(remaining_results)} outputs outside test split")

    predictions: list[OcrPrediction] = []
    for record in inputs.records:
        grouped_results = results_by_sample.get(record.sample_id, [])
        if not grouped_results:
            raise ValueError(f"PaddleOCR missing output for record: {record.sample_id}")
        raw_texts = [str(result["rec_text"]) for _, result in grouped_results]
        normalized_prediction = "".join(normalize_plate_text(text) for text in raw_texts)
        normalized_ground_truth = normalize_plate_text(record.annotation.raw_text)
        distance = levenshtein_distance(normalized_ground_truth, normalized_prediction)
        predictions.append(
            OcrPrediction(
                sample_id=record.sample_id,
                image_path=record.image_path,
                width=record.width,
                height=record.height,
                ground_truth=record.annotation.raw_text,
                prediction=" ".join(text for text in raw_texts if text),
                normalized_ground_truth=normalized_ground_truth,
                normalized_prediction=normalized_prediction,
                confidence=fmean(float(result["rec_score"]) for _, result in grouped_results),
                edit_distance=distance,
                exact_match=normalized_ground_truth == normalized_prediction,
            )
        )
    return tuple(predictions)


def _metric_group(predictions: tuple[OcrPrediction, ...]) -> OcrMetricGroup:
    """Tính micro CER để mỗi ký tự, không phải mỗi ảnh, có trọng số như nhau."""
    if not predictions:
        raise ValueError("không thể tính OCR metric trên nhóm rỗng")
    exact_matches = sum(prediction.exact_match for prediction in predictions)
    character_count = sum(len(prediction.normalized_ground_truth) for prediction in predictions)
    edit_distance = sum(prediction.edit_distance for prediction in predictions)
    cer = edit_distance / character_count
    return OcrMetricGroup(
        samples=len(predictions),
        exact_matches=exact_matches,
        exact_match_rate=exact_matches / len(predictions),
        character_count=character_count,
        edit_distance=edit_distance,
        cer=cer,
        character_accuracy=max(0.0, 1.0 - cer),
    )


def _prediction_from_result(
    record: OcrManifestRecord,
    raw_result: Any,
) -> OcrPrediction:
    """Chuyển result phụ thuộc PaddleX thành schema ổn định của project."""
    prediction = str(raw_result["rec_text"])
    normalized_ground_truth = normalize_plate_text(record.annotation.raw_text)
    normalized_prediction = normalize_plate_text(prediction)
    distance = levenshtein_distance(normalized_ground_truth, normalized_prediction)
    return OcrPrediction(
        sample_id=record.sample_id,
        image_path=record.image_path,
        width=record.width,
        height=record.height,
        ground_truth=record.annotation.raw_text,
        prediction=prediction,
        normalized_ground_truth=normalized_ground_truth,
        normalized_prediction=normalized_prediction,
        confidence=float(raw_result["rec_score"]),
        edit_distance=distance,
        exact_match=normalized_ground_truth == normalized_prediction,
    )


def _match_results_to_records(
    inputs: OcrEvaluationInputs,
    raw_results: list[Any],
) -> tuple[OcrPrediction, ...]:
    """Ghép output bằng path thay vì tin rằng backend luôn giữ nguyên thứ tự batch."""
    result_by_path: dict[Path, Any] = {}
    for result in raw_results:
        result_path = Path(result["input_path"]).resolve()
        if result_path in result_by_path:
            raise ValueError(f"PaddleOCR trả trùng image: {result_path}")
        result_by_path[result_path] = result

    predictions: list[OcrPrediction] = []
    for record, image_path in zip(inputs.records, inputs.image_paths, strict=True):
        try:
            result = result_by_path.pop(image_path)
        except KeyError as exc:
            raise ValueError(f"PaddleOCR thiếu output cho image: {image_path}") from exc
        predictions.append(_prediction_from_result(record, result))
    if result_by_path:
        raise ValueError(f"PaddleOCR trả {len(result_by_path)} output không thuộc test split")
    return tuple(predictions)


def _group_by_geometry(
    predictions: tuple[OcrPrediction, ...],
    compact_aspect_ratio: float,
) -> dict[str, OcrMetricGroup]:
    """Tách crop compact/wide như một proxy hình học, không gán nhãn hai dòng."""
    groups: dict[str, list[OcrPrediction]] = defaultdict(list)
    for prediction in predictions:
        geometry = (
            "compact" if prediction.width / prediction.height < compact_aspect_ratio else "wide"
        )
        groups[geometry].append(prediction)
    return {
        name: _metric_group(tuple(group_predictions))
        for name, group_predictions in sorted(groups.items())
    }


def _write_predictions(path: Path, predictions: tuple[OcrPrediction, ...]) -> None:
    """Ghi raw prediction để report metric luôn có thể audit lại."""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for prediction in predictions:
            stream.write(prediction.model_dump_json())
            stream.write("\n")


def evaluate_ocr(config_path: Path) -> OcrEvaluationResult:
    """Chạy recognition-only baseline trên toàn bộ ground-truth test crops."""
    inputs = validate_ocr_evaluation(config_path)
    config = inputs.config
    root = project_root(config_path)
    output_dir = resolve_project_path(root, config.project) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    recognition_inputs = _build_recognition_inputs(inputs, output_dir)

    # PaddleOCR imports Transformers, vì vậy PyTorch phải nạp DLL trước Paddle trên Windows.
    # import torch
    from paddleocr import TextRecognition

    initialization_started = time.perf_counter()
    model = TextRecognition(
        model_name=config.model_name,
        device=config.device,
        enable_mkldnn=config.enable_mkldnn,
        cpu_threads=config.cpu_threads,
    )
    initialization_seconds = time.perf_counter() - initialization_started
    try:
        inference_started = time.perf_counter()
        raw_results = model.predict(
            [str(recognition_input.path) for recognition_input in recognition_inputs],
            batch_size=config.batch_size,
        )
        inference_seconds = time.perf_counter() - inference_started
    finally:
        model.close()

    predictions = _match_results_to_recognition_inputs(
        inputs,
        recognition_inputs,
        raw_results,
    )
    ranked_failures = tuple(
        sorted(
            (prediction for prediction in predictions if not prediction.exact_match),
            key=lambda prediction: (
                -prediction.edit_distance,
                prediction.confidence,
                prediction.sample_id,
            ),
        )[: config.failure_examples]
    )
    result = OcrEvaluationResult(
        model_name=config.model_name,
        split="test",
        geometry=config.geometry,
        layout=config.layout,
        metrics=_metric_group(predictions),
        metrics_by_geometry=_group_by_geometry(predictions, config.compact_aspect_ratio),
        mean_confidence=fmean(prediction.confidence for prediction in predictions),
        model_initialization_seconds=initialization_seconds,
        inference_ms_per_image=inference_seconds * 1000 / len(predictions),
        failure_examples=ranked_failures,
    )

    _write_predictions(output_dir / "predictions.jsonl", predictions)
    (output_dir / "metrics.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("OCR baseline metrics saved to %s", output_dir / "metrics.json")
    return result


def _build_parser() -> argparse.ArgumentParser:
    """Tạo CLI preflight hoặc evaluate recognition-only baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ocr-baseline.yaml"),
    )
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Chuyển lỗi config/runtime OCR thành exit code và log ngắn gọn."""
    configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        if args.check_only:
            inputs = validate_ocr_evaluation(args.config)
            LOGGER.info(
                "OCR preflight passed images=%d model=%s geometry=%s layout=%s",
                len(inputs.records),
                inputs.config.model_name,
                inputs.config.geometry,
                inputs.config.layout,
            )
        else:
            result = evaluate_ocr(args.config)
            LOGGER.info(
                "OCR baseline completed exact=%.4f CER=%.4f char_acc=%.4f",
                result.metrics.exact_match_rate,
                result.metrics.cer,
                result.metrics.character_accuracy,
            )
    except (AttributeError, ImportError, OSError, ValueError) as exc:
        LOGGER.error("OCR evaluation failed: %s", exc)
        return 1
    return 0
