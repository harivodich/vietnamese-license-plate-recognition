"""Kiểm thử xuất manifest detection sang định dạng Ultralytics."""

from pathlib import Path
from typing import Literal

import pytest
import yaml

from vlpr.data.manifest_io import write_manifest
from vlpr.data.manifest_schema import (
    DetectionAnnotation,
    DetectionManifestRecord,
    YoloBox,
)
from vlpr.data.yolo_export import export_yolo_detection_dataset


def _record(
    name: str,
    split: Literal["train", "validation", "test"],
) -> DetectionManifestRecord:
    """Tạo detection record đã có project split cho fixture."""
    return DetectionManifestRecord(
        sample_id=f"detection:images/source/{name}.jpg",
        dataset_name="detection",
        task="detection",
        image_path=f"images/source/{name}.jpg",
        source_split="source",
        split=split,
        width=100,
        height=50,
        sha256=f"{int(name):064x}",
        perceptual_hash="0" * 16,
        annotations=(
            DetectionAnnotation(bbox=YoloBox(center_x=0.5, center_y=0.5, width=0.2, height=0.2)),
        ),
    )


def _write_pair(dataset_root: Path, name: str) -> None:
    """Tạo cặp file ảnh-label tối thiểu vì exporter chỉ kiểm tra pairing."""
    image = dataset_root / "images" / "source" / f"{name}.jpg"
    label = dataset_root / "labels" / "source" / f"{name}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"image")
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def test_export_yolo_detection_dataset_writes_lists_and_yaml(tmp_path: Path) -> None:
    """Xác nhận exporter giữ split và tạo class map YOLO duy nhất."""
    dataset_root = tmp_path / "raw"
    records = (
        _record("1", "train"),
        _record("2", "validation"),
        _record("3", "test"),
    )
    for name in ("1", "2", "3"):
        _write_pair(dataset_root, name)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, records)

    dataset_yaml = export_yolo_detection_dataset(
        manifest_path=manifest,
        dataset_root=dataset_root,
        output_dir=tmp_path / "yolo",
    )

    decoded = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    assert decoded["names"] == {0: "license_plate"}
    train_image = tmp_path / "yolo" / "images" / "train" / f"{records[0].sha256}.jpg"
    train_label = tmp_path / "yolo" / "labels" / "train" / f"{records[0].sha256}.txt"
    assert train_image.read_bytes() == b"image"
    assert (
        (tmp_path / "yolo" / "train.txt")
        .read_text(encoding="utf-8")
        .endswith(f"/images/train/{records[0].sha256}.jpg\n")
    )
    assert train_label.read_text(encoding="utf-8") == (
        "0 0.50000000 0.50000000 0.20000000 0.20000000\n"
    )
    assert decoded["val"] == "validation.txt"


def test_export_yolo_detection_dataset_rejects_missing_label(tmp_path: Path) -> None:
    """Không bắt đầu training nếu manifest trỏ tới ảnh thiếu label."""
    dataset_root = tmp_path / "raw"
    image = dataset_root / "images" / "source" / "1.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [_record("1", "train")])

    with pytest.raises(FileNotFoundError, match="thiếu label"):
        export_yolo_detection_dataset(
            manifest_path=manifest,
            dataset_root=dataset_root,
            output_dir=tmp_path / "yolo",
        )
