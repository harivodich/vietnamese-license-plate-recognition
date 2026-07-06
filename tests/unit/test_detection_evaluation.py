"""Kiểm thử config, preflight và chuẩn hóa metric detection test."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from vlpr.evaluation.detection import (
    build_evaluation_arguments,
    build_evaluation_result,
    load_detection_evaluation,
    validate_detection_evaluation,
)


def _write_evaluation_fixture(root: Path, *, split: str = "test") -> Path:
    """Tạo checkpoint, dataset và config nhỏ nhất cho evaluator preflight."""
    config_path = root / "configs" / "detection-evaluation.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        f"""
checkpoint: artifacts/detection/baseline/weights/best.pt
data: data/processed/yolo/dataset.yaml
project: artifacts/detection
name: baseline-test
split: {split}
imgsz: 640
batch: 2
workers: 0
device: null
conf: 0.001
iou: 0.7
max_det: 100
plots: true
operating_conf: 0.25
match_iou: 0.5
failure_examples: 5
""".lstrip(),
        encoding="utf-8",
    )
    checkpoint = root / "artifacts" / "detection" / "baseline" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    yolo_root = root / "data" / "processed" / "yolo"
    image_path = yolo_root / "images" / "test" / "image.jpg"
    label_path = yolo_root / "labels" / "test" / "image.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (yolo_root / "test.txt").write_text(
        image_path.resolve().as_posix() + "\n",
        encoding="utf-8",
    )
    (yolo_root / "dataset.yaml").write_text(
        "test: test.txt\nnames:\n  0: license_plate\n",
        encoding="utf-8",
    )
    return config_path


def test_preflight_counts_test_images_and_instances(tmp_path: Path) -> None:
    """Preflight phải đếm đúng ảnh/bbox và khóa đối số split test."""
    config_path = _write_evaluation_fixture(tmp_path)

    inputs = validate_detection_evaluation(config_path)

    assert inputs.image_count == 1
    assert inputs.instance_count == 1
    arguments = build_evaluation_arguments(config_path, inputs)
    assert arguments["split"] == "test"
    assert arguments["conf"] == 0.001
    assert arguments["exist_ok"] is True


def test_config_rejects_validation_split(tmp_path: Path) -> None:
    """Evaluation config không được phép thay test bằng validation."""
    config_path = _write_evaluation_fixture(tmp_path, split="val")

    with pytest.raises(ValueError, match="test"):
        load_detection_evaluation(config_path)


def test_result_records_checkpoint_identity_and_metrics(tmp_path: Path) -> None:
    """Summary phải gắn metric với đúng SHA-256 của checkpoint."""
    config_path = _write_evaluation_fixture(tmp_path)
    inputs = validate_detection_evaluation(config_path)
    metrics = SimpleNamespace(
        box=SimpleNamespace(mp=0.91, mr=0.82, map50=0.88, map=0.61),
        speed={"preprocess": 0.2, "inference": 4.5, "postprocess": 0.3},
    )

    result = build_evaluation_result(inputs, metrics)

    assert result.checkpoint_sha256 == (
        "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef"
    )
    assert result.precision == pytest.approx(0.91)
    assert result.map50_95 == pytest.approx(0.61)
