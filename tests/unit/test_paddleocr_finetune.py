"""Tests for PaddleOCR fine-tune handoff config generation."""

from pathlib import Path

from PIL import Image

from vlpr.training.paddleocr_finetune import prepare_paddleocr_finetune


def _write_image(path: Path) -> None:
    """Create a tiny exported OCR line image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 24), "white").save(path)


def _write_exported_data(root: Path) -> None:
    """Create a minimal PaddleOCR-format exported dataset."""
    _write_image(root / "images" / "train" / "a.png")
    _write_image(root / "images" / "validation" / "b.png")
    _write_image(root / "images" / "test" / "c.png")
    (root / "train_list.txt").write_text("images/train/a.png\t30A12345\n", encoding="utf-8")
    (root / "val_list.txt").write_text("images/validation/b.png\t51B67890\n", encoding="utf-8")
    (root / "test_list.txt").write_text("images/test/c.png\t60MĐ101835\n", encoding="utf-8")
    (root / "dict.txt").write_text("0\n1\n2\n3\n4\n5\n6\n7\n8\n9\nA\nB\nĐ\n", encoding="utf-8")


def _write_fake_paddleocr_repo(root: Path) -> None:
    """Create the official repo files required by preflight without importing PaddleOCR."""
    (root / "tools").mkdir(parents=True)
    (root / "tools" / "train.py").write_text("print('train')\n", encoding="utf-8")
    config_path = root / "configs" / "rec" / "PP-OCRv5"
    config_path.mkdir(parents=True)
    (config_path / "PP-OCRv5_mobile_rec.yml").write_text(
        """
Global:
  use_gpu: true
  epoch_num: 75
  character_dict_path: ./dict.txt
  save_model_dir: ./output
  pretrained_model: ''
Optimizer:
  lr:
    learning_rate: 0.0005
    warmup_epoch: 5
Train:
  dataset:
    data_dir: ./train_data
    label_file_list:
      - ./train_data/train_list.txt
  loader:
    batch_size_per_card: 128
    num_workers: 8
Eval:
  dataset:
    data_dir: ./train_data
    label_file_list:
      - ./train_data/val_list.txt
  loader:
    batch_size_per_card: 128
    num_workers: 4
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_prepare_paddleocr_finetune_writes_train_config(tmp_path: Path) -> None:
    """Preflight validates exported data and writes a PaddleOCR train YAML."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    repo = tmp_path / "external" / "PaddleOCR"
    export_dir = tmp_path / "data" / "processed" / "ocr_finetune_paddleocr"
    _write_fake_paddleocr_repo(repo)
    _write_exported_data(export_dir)
    config = config_dir / "ocr-paddleocr-finetune.yaml"
    config.write_text(
        "\n".join(
            [
                "paddleocr:",
                f"  repo_dir: {repo.relative_to(tmp_path).as_posix()}",
                "  base_config: configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml",
                "  train_script: tools/train.py",
                "data:",
                f"  export_dir: {export_dir.relative_to(tmp_path).as_posix()}",
                "  train_list: train_list.txt",
                "  validation_list: val_list.txt",
                "  test_list: test_list.txt",
                "  character_dict: dict.txt",
                "output:",
                "  project: artifacts/ocr",
                "  name: paddleocr-test",
                "  train_config: train_config.yml",
                "train:",
                "  use_gpu: false",
                "  epochs: 3",
                "  learning_rate: 0.0001",
                "  warmup_epoch: 1",
                "  batch_size_per_card: 2",
                "  eval_batch_step: [0, 10]",
                "  save_epoch_step: 1",
                "  num_workers: 0",
                "  pretrained_model: ''",
                "  resume_from: ''",
                "  use_space_char: false",
                "",
            ]
        ),
        encoding="utf-8",
    )

    prepared = prepare_paddleocr_finetune(config)
    generated = prepared.train_config.read_text(encoding="utf-8")

    assert prepared.train_samples == 1
    assert prepared.validation_samples == 1
    assert prepared.test_samples == 1
    assert prepared.characters == 13
    assert prepared.working_dir == repo.resolve()
    assert "learning_rate: 0.0001" in generated
    assert "batch_size_per_card: 2" in generated
    assert "first_bs: 2" in generated
    assert "checkpoints: ''" in generated
    assert "use_space_char: false" in generated
    assert "shuffle: false" in generated
    assert str((export_dir / "dict.txt").resolve()) in generated
