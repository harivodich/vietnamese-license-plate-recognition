"""Kiểm thử schema và preflight của detection baseline."""

from pathlib import Path

import pytest

from vlpr.training.detection import (
    build_training_arguments,
    load_detection_experiment,
    resume_detection,
    validate_detection_experiment,
)


def _write_config(root: Path, *, imgsz: int = 640) -> Path:
    """Tạo training config đầy đủ để test strict validation."""
    config = root / "configs" / "detection.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        f"""
model: yolo11n.pt
data: data/processed/yolo/dataset.yaml
project: artifacts/detection
name: baseline
train:
  epochs: 1
  patience: 0
  imgsz: {imgsz}
  batch: 2
  workers: 0
  device: null
  seed: 7
  deterministic: true
  amp: true
  cache: false
  val: true
  save: true
  save_period: -1
  plots: true
  optimizer: AdamW
  lr0: 0.001
  lrf: 0.01
  weight_decay: 0.0005
  warmup_epochs: 0.0
  cos_lr: true
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 3.0
  translate: 0.1
  scale: 0.5
  shear: 2.0
  perspective: 0.0005
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0
  mixup: 0.1
  close_mosaic: 0
""".lstrip(),
        encoding="utf-8",
    )
    return config


def test_validate_detection_experiment_accepts_complete_dataset(tmp_path: Path) -> None:
    """Preflight đạt khi YAML tham chiếu đủ ba image list không rỗng."""
    config_path = _write_config(tmp_path)
    yolo_root = tmp_path / "data" / "processed" / "yolo"
    yolo_root.mkdir(parents=True)
    (yolo_root / "dataset.yaml").write_text(
        "train: train.txt\nval: validation.txt\ntest: test.txt\n",
        encoding="utf-8",
    )
    for split, name in (
        ("train", "train.txt"),
        ("validation", "validation.txt"),
        ("test", "test.txt"),
    ):
        image = yolo_root / "images" / split / "image.jpg"
        label = yolo_root / "labels" / split / "image.txt"
        image.parent.mkdir(parents=True)
        label.parent.mkdir(parents=True)
        image.write_bytes(b"image")
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (yolo_root / name).write_text(
            image.resolve().as_posix() + "\n",
            encoding="utf-8",
        )

    config = validate_detection_experiment(config_path)

    assert config.train.seed == 7
    assert config.augmentation.mosaic == 1.0

    arguments = build_training_arguments(config_path, config)
    assert arguments["data"] == str(yolo_root / "dataset.yaml")
    assert arguments["batch"] == 2
    assert arguments["close_mosaic"] == 0


def test_load_detection_experiment_rejects_non_stride_image_size(
    tmp_path: Path,
) -> None:
    """Từ chối kích thước không chia hết cho stride 32."""
    config_path = _write_config(tmp_path, imgsz=630)

    with pytest.raises(ValueError, match="chia hết cho 32"):
        load_detection_experiment(config_path)


def test_validate_detection_experiment_rejects_raw_image_path(
    tmp_path: Path,
) -> None:
    """Không cho training list trỏ ra ngoài processed image tree."""
    config_path = _write_config(tmp_path)
    yolo_root = tmp_path / "data" / "processed" / "yolo"
    yolo_root.mkdir(parents=True)
    external_image = tmp_path / "data" / "raw" / "image.jpg"
    external_image.parent.mkdir(parents=True)
    external_image.write_bytes(b"image")
    (yolo_root / "dataset.yaml").write_text(
        "train: train.txt\nval: validation.txt\ntest: test.txt\n",
        encoding="utf-8",
    )
    for name in ("train.txt", "validation.txt", "test.txt"):
        (yolo_root / name).write_text(
            external_image.resolve().as_posix() + "\n",
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="processed images"):
        validate_detection_experiment(config_path)


def test_resume_detection_rejects_missing_checkpoint(tmp_path: Path) -> None:
    """Resume dừng trước khi import model nếu last.pt không tồn tại."""
    config_path = _write_config(tmp_path)
    yolo_root = tmp_path / "data" / "processed" / "yolo"
    yolo_root.mkdir(parents=True)
    (yolo_root / "dataset.yaml").write_text(
        "train: train.txt\nval: validation.txt\ntest: test.txt\n",
        encoding="utf-8",
    )
    for split, name in (
        ("train", "train.txt"),
        ("validation", "validation.txt"),
        ("test", "test.txt"),
    ):
        image = yolo_root / "images" / split / "image.jpg"
        label = yolo_root / "labels" / split / "image.txt"
        image.parent.mkdir(parents=True)
        label.parent.mkdir(parents=True)
        image.write_bytes(b"image")
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (yolo_root / name).write_text(image.resolve().as_posix() + "\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        resume_detection(config_path, Path("artifacts/missing/last.pt"))
