"""Tests for strict project configuration."""

from pathlib import Path

from vlpr.config import SplitSettings, load_config


def test_split_ratios_accept_complete_partition() -> None:
    split = SplitSettings(train=0.75, validation=0.125, test=0.125, seed=42)

    assert split.train + split.validation + split.test == 1.0


def test_config_exposes_detection_and_ocr_datasets() -> None:
    config = load_config(Path(r"E:\PRJ\vietnamese-license-plate-recognition\configs\dataset.yaml"))

    assert config.dataset("detection").task == "detection"
    assert config.dataset("ocr").task == "ocr"
