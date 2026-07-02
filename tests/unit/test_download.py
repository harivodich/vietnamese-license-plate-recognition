"""Kiểm thử quy trình tải dataset nguyên tử và có thể tái lập."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from vlpr.data import download
from vlpr.data.receipt import RECEIPT_NAME, fingerprint_directory, read_receipt


def _write_config(root: Path) -> Path:
    """Tạo cấu hình dataset tối thiểu trong thư mục test tạm."""
    config_dir = root / "configs"
    config_dir.mkdir()
    config_path = config_dir / "dataset.yaml"
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


def test_fingerprint_directory_is_stable_and_ignores_receipt(tmp_path: Path) -> None:
    """Xác nhận fingerprint ổn định và không tự đưa receipt vào phép hash."""
    (tmp_path / "b.txt").write_text("second", encoding="utf-8")
    (tmp_path / "a.txt").write_text("first", encoding="utf-8")
    first = fingerprint_directory(tmp_path)
    (tmp_path / RECEIPT_NAME).write_text("{}", encoding="utf-8")

    assert fingerprint_directory(tmp_path) == first
    assert first[1:] == (2, 11)


def test_download_publishes_receipt_and_skips_completed_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Xác nhận download được publish, có receipt và lần chạy sau không tải lại."""
    config_path = _write_config(tmp_path)

    def fake_download(_handle: str, staging_dir: Path) -> Path:
        """Mô phỏng KaggleHub tạo một file trong staging mà không dùng mạng."""
        destination = staging_dir
        (destination / "image.jpg").write_bytes(b"image")
        return staging_dir

    kaggle_download = Mock(side_effect=fake_download)
    monkeypatch.setattr(download, "_download_from_kaggle", kaggle_download)

    output_dir = download.download_dataset(config_path, "detection")
    second_output = download.download_dataset(config_path, "detection")
    receipt = read_receipt(output_dir)

    assert second_output == output_dir
    assert receipt is not None
    assert receipt.status == "complete"
    assert receipt.file_count == 1
    assert receipt.total_bytes == 5
    assert kaggle_download.call_count == 1


def test_incomplete_target_requires_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Xác nhận target không hoàn chỉnh không bị tự động ghi đè khi thiếu --force."""
    config_path = _write_config(tmp_path)
    target = tmp_path / "data" / "raw" / "detection" / "v1"
    target.mkdir(parents=True)
    (target / "partial.txt").write_text("partial", encoding="utf-8")

    def fake_download(_handle: str, staging_dir: Path) -> Path:
        """Mô phỏng dữ liệu mới để bảo đảm code từ chối trước bước publish."""
        destination = staging_dir
        (destination / "image.jpg").write_bytes(b"image")
        return staging_dir

    monkeypatch.setattr(download, "_download_from_kaggle", fake_download)

    with pytest.raises(FileExistsError, match="--force"):
        download.download_dataset(config_path, "detection")


def test_force_replaces_incomplete_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Xác nhận --force thay partial target bằng dataset hoàn chỉnh có receipt."""
    config_path = _write_config(tmp_path)
    target = tmp_path / "data" / "raw" / "detection" / "v1"
    target.mkdir(parents=True)
    (target / "partial.txt").write_text("partial", encoding="utf-8")

    def fake_download(_handle: str, staging_dir: Path) -> Path:
        """Mô phỏng một lần tải hoàn chỉnh dùng để kiểm tra nhánh force."""
        (staging_dir / "image.jpg").write_bytes(b"complete")
        return staging_dir

    monkeypatch.setattr(download, "_download_from_kaggle", fake_download)

    output_dir = download.download_dataset(config_path, "detection", force=True)

    assert not (output_dir / "partial.txt").exists()
    assert (output_dir / "image.jpg").read_bytes() == b"complete"
    assert read_receipt(output_dir) is not None
