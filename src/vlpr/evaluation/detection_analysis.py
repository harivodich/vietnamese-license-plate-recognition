"""Phân tích detection tại một confidence threshold cố định."""

import argparse
import json
import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field

from vlpr.config import project_root, resolve_project_path
from vlpr.evaluation.detection import (
    DetectionEvaluationConfig,
    DetectionEvaluationInputs,
    validate_detection_evaluation,
)
from vlpr.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class GroundTruth:
    """Một bbox thật ở hệ tọa độ pixel và nhóm kích thước sau letterbox."""

    box: Box
    size_group: str


@dataclass(frozen=True)
class MatchResult:
    """Chỉ số bbox thật và dự đoán đã ghép một-một tại ngưỡng IoU."""

    matched_ground_truth: frozenset[int]
    matched_predictions: frozenset[int]


class SizeRecall(BaseModel):
    """Recall của một nhóm kích thước bbox."""

    model_config = ConfigDict(extra="forbid")

    instances: int = Field(ge=0)
    matched: int = Field(ge=0)
    recall: float = Field(ge=0.0, le=1.0)


class FailureExample(BaseModel):
    """Metadata tối thiểu để truy lại một ảnh lỗi đã render."""

    model_config = ConfigDict(extra="forbid")

    image: str
    false_negatives: int = Field(ge=0)
    false_positives: int = Field(ge=0)


class DetectionOperatingPointResult(BaseModel):
    """Chất lượng detection tại confidence và IoU cố định."""

    model_config = ConfigDict(extra="forbid")

    confidence_threshold: float
    match_iou_threshold: float
    image_count: int = Field(gt=0)
    ground_truth_count: int = Field(gt=0)
    prediction_count: int = Field(ge=0)
    matched_count: int = Field(ge=0)
    false_negative_count: int = Field(ge=0)
    false_positive_count: int = Field(ge=0)
    recall_by_size: dict[str, SizeRecall]
    failure_examples: tuple[FailureExample, ...]


def intersection_over_union(first: Box, second: Box) -> float:
    """Tính phần giao trên phần hợp của hai bbox xyxy."""
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def match_boxes(
    ground_truth: list[GroundTruth],
    predictions: list[tuple[Box, float]],
    iou_threshold: float,
) -> MatchResult:
    """Ghép greedy theo confidence để mỗi dự đoán và bbox thật chỉ được dùng một lần."""
    matched_ground_truth: set[int] = set()
    matched_predictions: set[int] = set()
    ranked_predictions = sorted(
        enumerate(predictions),
        key=lambda item: item[1][1],
        reverse=True,
    )
    for prediction_index, (prediction_box, _) in ranked_predictions:
        candidates = (
            (intersection_over_union(prediction_box, target.box), target_index)
            for target_index, target in enumerate(ground_truth)
            if target_index not in matched_ground_truth
        )
        best_iou, best_target = max(candidates, default=(0.0, -1))
        if best_iou >= iou_threshold:
            matched_ground_truth.add(best_target)
            matched_predictions.add(prediction_index)
    return MatchResult(
        matched_ground_truth=frozenset(matched_ground_truth),
        matched_predictions=frozenset(matched_predictions),
    )


def _size_group(
    normalized_width: float,
    normalized_height: float,
    image_width: int,
    image_height: int,
    image_size: int,
) -> str:
    """Phân nhóm theo diện tích sau letterbox để khớp định nghĩa trong dataset audit."""
    scale = min(image_size / image_width, image_size / image_height)
    resized_area = normalized_width * image_width * scale * normalized_height * image_height * scale
    if resized_area < 32**2:
        return "small"
    if resized_area < 96**2:
        return "medium"
    return "large"


def _read_ground_truth(
    image_path: Path,
    inputs: DetectionEvaluationInputs,
    image_width: int,
    image_height: int,
) -> list[GroundTruth]:
    """Đọc YOLO label và đổi bbox chuẩn hóa sang pixel để matching."""
    images_root = inputs.dataset_yaml.parent / "images"
    labels_root = inputs.dataset_yaml.parent / "labels"
    relative_path = image_path.resolve().relative_to(images_root.resolve())
    label_path = (labels_root / relative_path).with_suffix(".txt")
    targets: list[GroundTruth] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _, center_x, center_y, width, height = line.split()
        x, y, w, h = map(float, (center_x, center_y, width, height))
        targets.append(
            GroundTruth(
                box=(
                    (x - w / 2) * image_width,
                    (y - h / 2) * image_height,
                    (x + w / 2) * image_width,
                    (y + h / 2) * image_height,
                ),
                size_group=_size_group(
                    w,
                    h,
                    image_width,
                    image_height,
                    inputs.config.imgsz,
                ),
            )
        )
    return targets


def _draw_failure(
    image_path: Path,
    ground_truth: list[GroundTruth],
    predictions: list[tuple[Box, float]],
    matches: MatchResult,
    output_path: Path,
) -> None:
    """Render ground truth và prediction bằng màu riêng để rà lỗi định tính."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, target in enumerate(ground_truth):
        color = "#00b050" if index in matches.matched_ground_truth else "#ff2020"
        draw.rectangle(target.box, outline=color, width=3)
        text_y = max(0.0, target.box[1] - 12)
        draw.text((target.box[0], text_y), f"GT {target.size_group}", fill=color)
    for index, (box, confidence) in enumerate(predictions):
        color = "#00a0ff" if index in matches.matched_predictions else "#ff00ff"
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0], box[1]), f"P {confidence:.2f}", fill=color)
    image.save(output_path, quality=92)


def _to_box(values: list[float]) -> Box:
    """Đổi một hàng xyxy động thành tuple bốn phần tử đã kiểm tra."""
    if len(values) != 4:
        raise ValueError(f"prediction box phải có 4 tọa độ, nhận được {len(values)}")
    return values[0], values[1], values[2], values[3]


def _predict_in_batches(
    model: Any,
    image_paths: list[str],
    config: DetectionEvaluationConfig,
) -> Iterator[tuple[Path, Any]]:
    """Giới hạn số ảnh in-memory mỗi lượt để không vượt VRAM."""
    for start in range(0, len(image_paths), config.batch):
        batch_paths = image_paths[start : start + config.batch]
        batch_results = model.predict(
            source=batch_paths,
            imgsz=config.imgsz,
            batch=config.batch,
            device=config.device,
            conf=config.operating_conf,
            iou=config.iou,
            max_det=config.max_det,
            verbose=False,
        )
        for image_path, result in zip(batch_paths, batch_results, strict=True):
            yield Path(image_path), result


def analyze_detection_operating_point(config_path: Path) -> DetectionOperatingPointResult:
    """Chạy predict và đo lỗi tại operating point đã đóng băng trong config."""
    from ultralytics.models.yolo.model import YOLO

    inputs = validate_detection_evaluation(config_path)
    config = inputs.config
    image_paths = [
        line.strip()
        for line in inputs.test_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model = YOLO(str(inputs.checkpoint))
    results = _predict_in_batches(model, image_paths, config)

    size_instances: Counter[str] = Counter()
    size_matches: Counter[str] = Counter()
    prediction_count = 0
    matched_count = 0
    false_positive_count = 0
    failure_rows: list[
        tuple[int, Path, list[GroundTruth], list[tuple[Box, float]], MatchResult]
    ] = []

    for image_path, result in results:
        image_height, image_width = result.orig_shape
        targets = _read_ground_truth(image_path, inputs, image_width, image_height)
        predictions: list[tuple[Box, float]] = []
        if result.boxes is not None:
            predictions = [
                (_to_box([float(value) for value in box]), float(confidence))
                for box, confidence in zip(
                    result.boxes.xyxy.tolist(),
                    result.boxes.conf.tolist(),
                    strict=True,
                )
            ]
        matches = match_boxes(targets, predictions, config.match_iou)
        for index, target in enumerate(targets):
            size_instances[target.size_group] += 1
            if index in matches.matched_ground_truth:
                size_matches[target.size_group] += 1
        false_negatives = len(targets) - len(matches.matched_ground_truth)
        false_positives = len(predictions) - len(matches.matched_predictions)
        prediction_count += len(predictions)
        matched_count += len(matches.matched_ground_truth)
        false_positive_count += false_positives
        if false_negatives or false_positives:
            failure_rows.append(
                (
                    false_negatives + false_positives,
                    image_path,
                    targets,
                    predictions,
                    matches,
                )
            )

    root = project_root(config_path)
    output_dir = resolve_project_path(root, config.project) / config.name
    examples_dir = output_dir / "failure_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    failure_examples: list[FailureExample] = []
    ranked_failures = sorted(failure_rows, key=lambda item: item[0], reverse=True)
    for rank, (_, image_path, targets, predictions, matches) in enumerate(
        ranked_failures[: config.failure_examples],
        start=1,
    ):
        output_path = examples_dir / f"{rank:02d}_{image_path.stem}.jpg"
        _draw_failure(image_path, targets, predictions, matches, output_path)
        failure_examples.append(
            FailureExample(
                image=output_path.name,
                false_negatives=len(targets) - len(matches.matched_ground_truth),
                false_positives=len(predictions) - len(matches.matched_predictions),
            )
        )

    recall_by_size = {
        name: SizeRecall(
            instances=size_instances[name],
            matched=size_matches[name],
            recall=size_matches[name] / size_instances[name] if size_instances[name] else 0.0,
        )
        for name in ("small", "medium", "large")
    }
    analysis = DetectionOperatingPointResult(
        confidence_threshold=config.operating_conf,
        match_iou_threshold=config.match_iou,
        image_count=inputs.image_count,
        ground_truth_count=inputs.instance_count,
        prediction_count=prediction_count,
        matched_count=matched_count,
        false_negative_count=inputs.instance_count - matched_count,
        false_positive_count=false_positive_count,
        recall_by_size=recall_by_size,
        failure_examples=tuple(failure_examples),
    )
    output_path = output_dir / "operating_point.json"
    output_path.write_text(
        json.dumps(analysis.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Detection operating-point analysis saved to %s", output_path)
    return analysis


def _build_parser() -> argparse.ArgumentParser:
    """Tạo CLI dùng chung evaluation config với operating-point analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/detection-evaluation.yaml"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Chạy error analysis và chuyển lỗi dự kiến thành exit code 1."""
    configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        result = analyze_detection_operating_point(args.config)
    except (AttributeError, ImportError, OSError, ValueError) as exc:
        LOGGER.error("Detection error analysis failed: %s", exc)
        return 1
    LOGGER.info(
        "Operating point completed matched=%d FN=%d FP=%d",
        result.matched_count,
        result.false_negative_count,
        result.false_positive_count,
    )
    return 0
