"""Export PaddleOCR fine-tuning assets into a Colab-ready zip."""

import argparse
import json
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from vlpr.config import project_root, resolve_project_path
from vlpr.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


class ColabExportConfig(BaseModel):
    """Validated settings for a reproducible Colab training package."""

    model_config = ConfigDict(extra="forbid")

    data_dir: Path = Path("data/processed/ocr_finetune_paddleocr")
    train_config: Path = Path("artifacts/ocr/paddleocr-v5-mobile-finetune/train_config.yml")
    pretrained_model: Path = Path(
        "artifacts/ocr/paddleocr-v5-mobile-finetune/pretrained/"
        "en_PP-OCRv5_mobile_rec_pretrained.pdparams"
    )
    output_dir: Path = Path("colab_training_pack")
    colab_local_dir: str = "/content/colab_training_pack"
    colab_drive_dir: str = "/content/drive/MyDrive/paddleocr_checkpoints"
    use_gpu: bool = True
    distributed: bool = False
    num_workers: int = Field(default=4, ge=0)


def _load_config(path: Path | None) -> ColabExportConfig:
    """Load optional export config, using safe defaults when omitted."""
    if path is None:
        return ColabExportConfig()
    with path.open("r", encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Colab export config root must be a mapping: {path}")
    return ColabExportConfig.model_validate(raw)


def _safe_replace_dir(path: Path, root: Path) -> None:
    """Replace an output directory only when it stays inside the project root."""
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"refusing to delete outside project root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def _count_non_empty_lines(path: Path) -> int:
    """Count non-empty UTF-8 lines in a PaddleOCR list file."""
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _validate_exported_data(data_dir: Path) -> dict[str, int]:
    """Validate required OCR export files and return their row counts."""
    required = {
        "train_samples": data_dir / "train_list.txt",
        "validation_samples": data_dir / "val_list.txt",
        "test_samples": data_dir / "test_list.txt",
        "characters": data_dir / "dict.txt",
    }
    counts: dict[str, int] = {}
    for key, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"required OCR export file not found: {path}")
        counts[key] = _count_non_empty_lines(path)
        if counts[key] == 0:
            raise ValueError(f"required OCR export file is empty: {path}")
    return counts


def _rewrite_colab_config(
    config: dict[str, Any],
    *,
    settings: ColabExportConfig,
) -> dict[str, Any]:
    """Rewrite local train paths into Colab and Google Drive paths."""
    colab_local = settings.colab_local_dir.rstrip("/")
    colab_drive = settings.colab_drive_dir.rstrip("/")
    config["Global"]["use_gpu"] = settings.use_gpu
    config["Global"]["distributed"] = settings.distributed
    config["Global"]["save_model_dir"] = f"{colab_drive}/checkpoints"
    config["Global"]["pretrained_model"] = (
        f"{colab_local}/pretrained/en_PP-OCRv5_mobile_rec_pretrained.pdparams"
    )
    config["Global"]["save_inference_dir"] = f"{colab_drive}/inference"
    config["Global"]["infer_img"] = "doc/imgs_words/ch/word_1.jpg"
    config["Global"]["character_dict_path"] = f"{colab_local}/data/dict.txt"
    config["Global"]["save_res_path"] = f"{colab_drive}/predicts.txt"

    config["Train"]["dataset"]["data_dir"] = f"{colab_local}/data"
    config["Train"]["dataset"]["label_file_list"] = [f"{colab_local}/data/train_list.txt"]
    config["Train"]["loader"]["num_workers"] = settings.num_workers

    config["Eval"]["dataset"]["data_dir"] = f"{colab_local}/data"
    config["Eval"]["dataset"]["label_file_list"] = [f"{colab_local}/data/val_list.txt"]
    config["Eval"]["loader"]["num_workers"] = settings.num_workers
    config["Eval"]["loader"]["shuffle"] = False
    return config


def export_colab_zip(config_path: Path | None = None) -> Path:
    """Create a zip containing data, pretrained weights, and Colab train config."""
    settings = _load_config(config_path)
    root = project_root(config_path or Path("configs/ocr-paddleocr-finetune.yaml"))
    data_dir = resolve_project_path(root, settings.data_dir)
    train_config = resolve_project_path(root, settings.train_config)
    pretrained_model = resolve_project_path(root, settings.pretrained_model)
    output_dir = resolve_project_path(root, settings.output_dir)

    counts = _validate_exported_data(data_dir)
    if not train_config.is_file():
        raise FileNotFoundError(f"PaddleOCR train config not found: {train_config}")
    if not pretrained_model.is_file():
        raise FileNotFoundError(f"PaddleOCR pretrained model not found: {pretrained_model}")

    _safe_replace_dir(output_dir, root)
    shutil.copytree(data_dir, output_dir / "data")
    pretrained_dest = output_dir / "pretrained"
    pretrained_dest.mkdir(parents=True)
    shutil.copy(pretrained_model, pretrained_dest / pretrained_model.name)

    with train_config.open("r", encoding="utf-8") as stream:
        raw_config: Any = yaml.safe_load(stream)
    if not isinstance(raw_config, dict):
        raise ValueError(f"PaddleOCR train config root must be a mapping: {train_config}")
    colab_config = _rewrite_colab_config(raw_config, settings=settings)
    (output_dir / "train_config.yml").write_text(
        yaml.safe_dump(colab_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    zip_base = output_dir.with_suffix("")
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", output_dir))
    LOGGER.info("Colab training pack exported: %s", zip_path)
    LOGGER.info("Pack summary: %s", counts)
    return zip_path


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for Colab pack export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Colab pack export from the command line."""
    configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        export_colab_zip(args.config)
    except (OSError, ValueError) as exc:
        LOGGER.error("Colab export failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
