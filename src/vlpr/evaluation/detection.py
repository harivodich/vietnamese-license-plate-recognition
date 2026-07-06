"""Đánh giá checkpoint detection trên test set cố định bằng Ultralytics."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from vlpr.config import project_root, resolve_project_path
from vlpr.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


class DetectionEvaluationConfig(BaseModel):
    """Đóng băng checkpoint, test split và tham số inference dùng khi đánh giá."""

    model_config = ConfigDict(extra="forbid")

    checkpoint: Path
    data: Path
    project: Path
    name: str = Field(min_length=1)
    split: Literal["test"]
    imgsz: int = Field(gt=0)
    batch: int = Field(gt=0)
    workers: int = Field(ge=0)
    device: str | int | None
    conf: float = Field(ge=0.0, le=1.0)
    iou: float = Field(gt=0.0, le=1.0)
    max_det: int = Field(gt=0)
    plots: bool
    operating_conf: float = Field(ge=0.0, le=1.0)
    match_iou: float = Field(gt=0.0, le=1.0)
    failure_examples: int = Field(ge=0)


class DetectionEvaluationResult(BaseModel):
    """Các số liệu đủ để nhận diện checkpoint và so sánh các lần đánh giá."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_sha256: str
    split: Literal["test"]
    image_count: int = Field(gt=0)
    instance_count: int = Field(gt=0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    map50: float = Field(ge=0.0, le=1.0)
    map50_95: float = Field(ge=0.0, le=1.0)
    speed_ms_per_image: dict[str, float]


class DetectionEvaluationInputs(BaseModel):
    """Đường dẫn đã resolve và quy mô test set sau bước preflight."""

    model_config = ConfigDict(extra="forbid")

    config: DetectionEvaluationConfig
    checkpoint: Path
    dataset_yaml: Path
    test_list: Path
    image_count: int = Field(gt=0)
    instance_count: int = Field(gt=0)


def load_detection_evaluation(path: Path) -> DetectionEvaluationConfig:
    """Đọc YAML bằng schema strict để phát hiện key sai trước khi nạp model."""
    with path.open("r", encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"evaluation config root phải là mapping: {path}")
    config = DetectionEvaluationConfig.model_validate(raw)
    if config.imgsz % 32 != 0:
        raise ValueError("imgsz phải chia hết cho 32")
    return config


def validate_detection_evaluation(path: Path) -> DetectionEvaluationInputs:
    """Xác nhận checkpoint và test labels tồn tại trước khi dùng GPU."""
    config = load_detection_evaluation(path)
    root = project_root(path)
    checkpoint = resolve_project_path(root, config.checkpoint)
    dataset_yaml = resolve_project_path(root, config.data)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"không tìm thấy checkpoint: {checkpoint}")
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"không tìm thấy dataset YAML: {dataset_yaml}")

    with dataset_yaml.open("r", encoding="utf-8") as stream:
        dataset: Any = yaml.safe_load(stream)
    if not isinstance(dataset, dict) or not isinstance(dataset.get("test"), str):
        raise ValueError(f"dataset YAML thiếu test list: {dataset_yaml}")

    test_list = (dataset_yaml.parent / dataset["test"]).resolve()
    if not test_list.is_file():
        raise FileNotFoundError(f"không tìm thấy test list: {test_list}")

    images = [
        line.strip() for line in test_list.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not images:
        raise ValueError(f"test list rỗng: {test_list}")

    images_root = dataset_yaml.parent / "images"
    labels_root = dataset_yaml.parent / "labels"
    instance_count = 0
    for line_number, raw_image_path in enumerate(images, start=1):
        image_path = Path(raw_image_path)
        if not image_path.is_absolute():
            image_path = dataset_yaml.parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_relative_to(images_root.resolve()):
            raise ValueError(f"{test_list}:{line_number}: ảnh nằm ngoài processed images")
        if not image_path.is_file():
            raise FileNotFoundError(f"không tìm thấy test image: {image_path}")
        relative_path = image_path.relative_to(images_root.resolve())
        label_path = (labels_root / relative_path).with_suffix(".txt")
        if not label_path.is_file():
            raise FileNotFoundError(f"không tìm thấy test label: {label_path}")
        instance_count += sum(
            1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    if instance_count == 0:
        raise ValueError("test set không có detection instance")

    return DetectionEvaluationInputs(
        config=config,
        checkpoint=checkpoint,
        dataset_yaml=dataset_yaml,
        test_list=test_list,
        image_count=len(images),
        instance_count=instance_count,
    )


def build_evaluation_arguments(
    config_path: Path,
    inputs: DetectionEvaluationInputs,
) -> dict[str, Any]:
    """Chuyển config thành đối số model.val mà không dùng tham số train."""
    config = inputs.config
    root = project_root(config_path)
    return {
        "data": str(inputs.dataset_yaml),
        "split": config.split,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "workers": config.workers,
        "device": config.device,
        "conf": config.conf,
        "iou": config.iou,
        "max_det": config.max_det,
        "plots": config.plots,
        "project": str(resolve_project_path(root, config.project)),
        "name": config.name,
        "exist_ok": True,
    }


def _sha256(path: Path) -> str:
    """Nhận diện chính xác checkpoint mà không đưa file model vào Git."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_evaluation_result(
    inputs: DetectionEvaluationInputs,
    metrics: Any,
) -> DetectionEvaluationResult:
    """Chuẩn hóa object metric của Ultralytics thành schema ổn định của project."""
    box = metrics.box
    speed = {str(name): float(value) for name, value in metrics.speed.items()}
    return DetectionEvaluationResult(
        checkpoint_sha256=_sha256(inputs.checkpoint),
        split="test",
        image_count=inputs.image_count,
        instance_count=inputs.instance_count,
        precision=float(box.mp),
        recall=float(box.mr),
        map50=float(box.map50),
        map50_95=float(box.map),
        speed_ms_per_image=speed,
    )


def evaluate_detection(config_path: Path) -> DetectionEvaluationResult:
    """Chạy model.val trên test và lưu summary cạnh artifact do Ultralytics tạo."""
    from ultralytics.models.yolo.model import YOLO

    inputs = validate_detection_evaluation(config_path)
    model = YOLO(str(inputs.checkpoint))
    metrics = model.val(**build_evaluation_arguments(config_path, inputs))
    result = build_evaluation_result(inputs, metrics)

    root = project_root(config_path)
    output_dir = resolve_project_path(root, inputs.config.project) / inputs.config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metrics.json"
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Detection test metrics saved to %s", output_path)
    return result


def _build_parser() -> argparse.ArgumentParser:
    """Tạo CLI chỉ có một hành động: đánh giá checkpoint đã đóng băng."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/detection-evaluation.yaml"),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Chỉ kiểm tra checkpoint, dataset và test labels; không chạy inference.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Chuyển lỗi preflight/runtime thành exit code và log dễ đọc."""
    configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        if args.check_only:
            inputs = validate_detection_evaluation(args.config)
            LOGGER.info(
                "Evaluation preflight passed images=%d instances=%d checkpoint=%s",
                inputs.image_count,
                inputs.instance_count,
                inputs.checkpoint,
            )
        else:
            result = evaluate_detection(args.config)
            LOGGER.info(
                "Detection test completed P=%.4f R=%.4f mAP50=%.4f mAP50-95=%.4f",
                result.precision,
                result.recall,
                result.map50,
                result.map50_95,
            )
    except (AttributeError, ImportError, OSError, ValueError) as exc:
        LOGGER.error("Detection evaluation failed: %s", exc)
        return 1
    return 0
