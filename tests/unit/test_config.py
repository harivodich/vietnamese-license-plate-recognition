"""Kiểm thử validation nghiêm ngặt cho cấu hình dataset."""

from pathlib import Path

from vlpr.config import SplitSettings, load_config


def test_split_ratios_accept_complete_partition() -> None:
    """Xác nhận bộ tỷ lệ có tổng bằng 1 được Pydantic chấp nhận."""
    split = SplitSettings(train=0.75, validation=0.125, test=0.125, seed=42)

    assert split.train + split.validation + split.test == 1.0


def test_config_exposes_detection_and_ocr_datasets() -> None:
    """Xác nhận config thật có đủ detection, OCR và đường dẫn raw theo version."""
    config_path = Path(__file__).parents[2] / "configs" / "dataset.yaml"
    config = load_config(config_path)

    assert config.dataset("detection").task == "detection"
    assert config.dataset("ocr").task == "ocr"
    assert config.dataset("detection").raw_dir.as_posix().endswith("detection/v1")
    assert config.dataset("ocr").raw_dir.as_posix().endswith("ocr/v1")
    assert config.validation.report_path.as_posix() == "data/interim/dataset_audit.json"
