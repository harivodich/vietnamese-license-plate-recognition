"""Kiểm thử điều phối tạo hai manifest từ config."""

from pathlib import Path

import pytest
from PIL import Image

from vlpr.data.manifest_io import read_manifest
from vlpr.data.prepare import main, prepare_manifests


def _write_config(root: Path) -> Path:
    """Tạo config tối thiểu trỏ tới hai raw source fixture."""
    config_path = root / "configs" / "dataset.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
datasets:
  detection:
    handle: owner/detection
    version: 1
    country: VN
    task: detection
    expected_license: MIT
    raw_dir: data/raw/detection/v1
    manifest_path: data/interim/detection.jsonl
  ocr:
    handle: owner/ocr
    version: 1
    country: VN
    task: ocr
    expected_license: MIT
    raw_dir: data/raw/ocr/v1
    manifest_path: data/interim/ocr.jsonl
validation:
  image_extensions: [".jpg"]
  near_duplicate_hamming_distance: 6
  manual_review_sample_size: 100
split:
  train: 0.75
  validation: 0.125
  test: 0.125
  seed: 42
""".lstrip(),
        encoding="utf-8",
    )
    return config_path


def _write_image(path: Path) -> None:
    """Tạo ảnh fixture nhỏ và các thư mục cha."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 10), color="white").save(path)


def _create_sources(root: Path) -> None:
    """Tạo đủ cấu trúc detection và OCR để sinh hai manifest."""
    detection = root / "data" / "raw" / "detection" / "v1" / "License Plate Detection Dataset"
    for split in ("train", "val", "test"):
        image_path = detection / "images" / split / f"{split}.jpg"
        label_path = detection / "labels" / split / f"{split}.txt"
        _write_image(image_path)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    ocr = root / "data" / "raw" / "ocr" / "v1" / "lp_ocr_dataset_vi"
    _write_image(ocr / "imgs" / "train" / "train.jpg")
    _write_image(ocr / "imgs" / "val" / "val.jpg")
    (ocr / "labels").mkdir(parents=True)
    (ocr / "labels" / "train.txt").write_text(
        "imgs/train/train.jpg\t30A 12345\n",
        encoding="utf-8",
    )
    (ocr / "labels" / "val.txt").write_text(
        "imgs/val/val.jpg\t60MĐ1 01835\n",
        encoding="utf-8",
    )


def test_prepare_manifests_writes_both_configured_outputs(tmp_path: Path) -> None:
    """Xác nhận config điều khiển đúng raw input và manifest output."""
    config_path = _write_config(tmp_path)
    _create_sources(tmp_path)

    manifests = prepare_manifests(config_path)

    assert len(read_manifest(manifests["detection"])) == 3
    assert len(read_manifest(manifests["ocr"])) == 2


def test_main_stops_before_generation_when_source_is_unready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Xác nhận CLI không tạo manifest khi receipt nguồn chưa sẵn sàng."""
    monkeypatch.setattr(
        "vlpr.data.prepare.find_unready_sources",
        lambda _: ("ocr",),
    )

    assert main([]) == 1
