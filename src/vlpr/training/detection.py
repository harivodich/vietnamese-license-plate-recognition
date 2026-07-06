"""Kiểm tra cấu hình và điều phối huấn luyện detection baseline bằng Ultralytics."""

import argparse
import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vlpr.config import project_root, resolve_project_path
from vlpr.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


class DetectionTrainSettings(BaseModel):
    """Các hyperparameter ảnh hưởng trực tiếp đến quá trình tối ưu."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(gt=0)
    patience: int = Field(ge=0)
    imgsz: int = Field(gt=0)
    batch: int = Field(gt=0)
    workers: int = Field(ge=0)
    device: str | int | None
    seed: int = Field(ge=0)
    deterministic: bool
    amp: bool
    cache: bool | Literal["ram", "disk"]
    val: bool
    save: bool
    save_period: int = Field(ge=-1)
    plots: bool
    optimizer: str = Field(min_length=1)
    lr0: float = Field(gt=0.0)
    lrf: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    warmup_epochs: float = Field(ge=0.0)
    cos_lr: bool


class DetectionAugmentationSettings(BaseModel):
    """Các phép biến đổi chỉ dùng khi train detection."""

    model_config = ConfigDict(extra="forbid")

    hsv_h: float = Field(ge=0.0, le=1.0)
    hsv_s: float = Field(ge=0.0, le=1.0)
    hsv_v: float = Field(ge=0.0, le=1.0)
    degrees: float = Field(ge=0.0)
    translate: float = Field(ge=0.0, le=1.0)
    scale: float = Field(ge=0.0)
    shear: float = Field(ge=0.0)
    perspective: float = Field(ge=0.0, le=0.001)
    flipud: float = Field(ge=0.0, le=1.0)
    fliplr: float = Field(ge=0.0, le=1.0)
    mosaic: float = Field(ge=0.0, le=1.0)
    mixup: float = Field(ge=0.0, le=1.0)
    close_mosaic: int = Field(ge=0)


class DetectionExperimentConfig(BaseModel):
    """Đóng băng model, dataset, output và toàn bộ tham số baseline."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    data: Path
    project: Path
    name: str = Field(min_length=1)
    train: DetectionTrainSettings
    augmentation: DetectionAugmentationSettings

    @model_validator(mode="after")
    def validate_image_size(self) -> "DetectionExperimentConfig":
        """YOLO stride phổ biến là 32 nên kích thước ảnh phải chia hết cho 32."""
        if self.train.imgsz % 32 != 0:
            raise ValueError("train.imgsz phải chia hết cho 32")
        return self


def load_detection_experiment(path: Path) -> DetectionExperimentConfig:
    """Đọc YAML training với schema strict, không chấp nhận key viết sai."""
    with path.open("r", encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"training config root phải là mapping: {path}")
    return DetectionExperimentConfig.model_validate(raw)


def validate_detection_experiment(path: Path) -> DetectionExperimentConfig:
    """Xác nhận dataset materialized, pairing và split isolation trước khi train."""
    config = load_detection_experiment(path)
    root = project_root(path)
    dataset_yaml = resolve_project_path(root, config.data)
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"chưa có YOLO dataset YAML: {dataset_yaml}")
    with dataset_yaml.open("r", encoding="utf-8") as stream:
        dataset: Any = yaml.safe_load(stream)
    if not isinstance(dataset, dict):
        raise ValueError("YOLO dataset YAML phải là mapping")
    images_root = dataset_yaml.parent / "images"
    labels_root = dataset_yaml.parent / "labels"
    seen_images: set[Path] = set()
    for key in ("train", "val", "test"):
        value = dataset.get(key)
        if not isinstance(value, str):
            raise ValueError(f"YOLO dataset thiếu đường dẫn {key}")
        image_list = dataset_yaml.parent / value
        if not image_list.is_file() or image_list.stat().st_size == 0:
            raise FileNotFoundError(f"image list rỗng hoặc không tồn tại: {image_list}")
        split_images: set[Path] = set()
        for line_number, line in enumerate(
            image_list.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            image_path = Path(line)
            if not image_path.is_absolute():
                image_path = dataset_yaml.parent / image_path
            image_path = image_path.resolve()
            if not image_path.is_relative_to(images_root.resolve()):
                raise ValueError(f"{image_list}:{line_number}: ảnh phải nằm trong processed images")
            if not image_path.is_file():
                raise FileNotFoundError(f"thiếu ảnh YOLO: {image_path}")
            relative_path = image_path.relative_to(images_root.resolve())
            label_path = (labels_root / relative_path).with_suffix(".txt")
            if not label_path.is_file() or label_path.stat().st_size == 0:
                raise FileNotFoundError(f"label YOLO rỗng hoặc không tồn tại: {label_path}")
            if image_path in split_images:
                raise ValueError(f"{image_list}: đường dẫn ảnh bị lặp: {image_path}")
            if image_path in seen_images:
                raise ValueError(f"ảnh xuất hiện ở nhiều project split: {image_path}")
            split_images.add(image_path)
        seen_images.update(split_images)
    return config


def build_training_arguments(
    config_path: Path,
    config: DetectionExperimentConfig,
) -> dict[str, Any]:
    """Chuyển typed config thành keyword arguments truyền cho Ultralytics."""
    root = project_root(config_path)
    return {
        **config.train.model_dump(),
        **config.augmentation.model_dump(),
        "data": str(resolve_project_path(root, config.data)),
        "project": str(resolve_project_path(root, config.project)),
        "name": config.name,
    }


def train_detection(config_path: Path) -> DetectionExperimentConfig:
    """Nạp pretrained weights và bắt đầu một training run mới."""
    from ultralytics.models.yolo.model import YOLO

    config = validate_detection_experiment(config_path)
    model = YOLO(config.model)
    model.train(**build_training_arguments(config_path, config))
    return config


def resume_detection(
    config_path: Path,
    checkpoint_path: Path,
) -> DetectionExperimentConfig:
    """Tiếp tục đúng training state đã lưu trong checkpoint ``last.pt``."""
    config = validate_detection_experiment(config_path)
    root = project_root(config_path)
    checkpoint = resolve_project_path(root, checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"không tìm thấy checkpoint để resume: {checkpoint}")

    from ultralytics.models.yolo.model import YOLO

    model = YOLO(str(checkpoint))
    model.train(resume=True)
    return config


def _build_parser() -> argparse.ArgumentParser:
    """Tạo CLI tách check-only khỏi thao tác huấn luyện tốn tài nguyên."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/detection-baseline.yaml"),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument(
        "--resume",
        type=Path,
        metavar="LAST_PT",
        help="Tiếp tục training state từ checkpoint last.pt.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Kiểm tra hoặc chạy baseline và chuyển lỗi cấu hình thành exit code 1."""
    configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        if args.check_only:
            config = validate_detection_experiment(args.config)
            mode = "check"
        elif args.resume is not None:
            config = resume_detection(args.config, args.resume)
            mode = "resume"
        else:
            config = train_detection(args.config)
            mode = "train"
    except (ImportError, OSError, ValueError) as exc:
        LOGGER.error("Detection baseline failed: %s", exc)
        return 1
    LOGGER.info(
        "Detection baseline completed mode=%s model=%s data=%s",
        mode,
        config.model,
        config.data,
    )
    return 0
