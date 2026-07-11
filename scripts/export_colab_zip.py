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
    train_config: Path = Path("artifacts/ocr/paddleocr-v5-mobile-finetune-v2/train_config.yml")
    pretrained_model: Path = Path(
        "artifacts/ocr/paddleocr-v5-mobile-finetune/pretrained/"
        "en_PP-OCRv5_mobile_rec_pretrained.pdparams"
    )
    output_dir: Path = Path("colab_training_pack")
    colab_local_dir: str = "/content/colab_training_pack"
    colab_drive_dir: str = "/content/drive/MyDrive/paddleocr_checkpoints_v3"
    use_gpu: bool = True
    distributed: bool = False
    num_workers: int = Field(default=4, ge=0)
    reset_optimizer: bool = True
    colab_epoch_num: int = Field(default=20, ge=1)
    colab_learning_rate: float = Field(default=1.0e-5, gt=0)
    recconaug_prob: float = Field(default=0.0, ge=0.0, le=1.0)


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


def _copy_checkpoint_files(checkpoint_stem: Path | None, destination: Path) -> str | None:
    """Copy PaddleOCR checkpoint files and return checkpoint stem name for resume."""
    if checkpoint_stem is None:
        return None
    checkpoint_stem = checkpoint_stem.with_suffix("")
    required_files = [
        checkpoint_stem.with_suffix(suffix) for suffix in (".pdparams", ".pdopt", ".states")
    ]
    if not all(path.is_file() for path in required_files):
        return None

    destination.mkdir(parents=True)
    for path in required_files:
        shutil.copy(path, destination / path.name)
    return checkpoint_stem.name


def _copy_checkpoint_weights(checkpoint_stem: Path | None, destination: Path) -> str | None:
    """Copy only model weights so Colab can reset optimizer and LR schedule."""
    if checkpoint_stem is None:
        return None
    weights_path = checkpoint_stem.with_suffix(".pdparams")
    if not weights_path.is_file():
        return None

    destination.mkdir(parents=True)
    shutil.copy(weights_path, destination / weights_path.name)
    return weights_path.name


def _set_recconaug_probability(config: dict[str, Any], probability: float) -> None:
    """Tune RecConAug strength for small license-plate crops."""
    transforms = config["Train"]["dataset"].get("transforms", [])
    for transform in transforms:
        if isinstance(transform, dict) and "RecConAug" in transform:
            recconaug = transform["RecConAug"]
            if isinstance(recconaug, dict):
                recconaug["prob"] = probability
            return


def _rewrite_colab_config(
    config: dict[str, Any],
    *,
    settings: ColabExportConfig,
    resume_checkpoint_name: str | None,
    pretrained_model_name: str | None,
) -> dict[str, Any]:
    """Rewrite local train paths into Colab and Google Drive paths."""
    colab_local = settings.colab_local_dir.rstrip("/")
    colab_drive = settings.colab_drive_dir.rstrip("/")
    global_config = config["Global"]

    global_config["use_gpu"] = settings.use_gpu
    global_config["distributed"] = settings.distributed
    global_config["epoch_num"] = settings.colab_epoch_num
    global_config["save_model_dir"] = f"{colab_drive}/checkpoints"
    global_config["checkpoints"] = (
        f"{colab_local}/resume/{resume_checkpoint_name}" if resume_checkpoint_name else ""
    )
    global_config["pretrained_model"] = (
        f"{colab_local}/pretrained/{pretrained_model_name}" if pretrained_model_name else ""
    )
    global_config["save_inference_dir"] = f"{colab_drive}/inference"
    global_config["export_with_pir"] = False
    global_config["infer_img"] = "doc/imgs_words/ch/word_1.jpg"
    global_config["character_dict_path"] = f"{colab_local}/data/dict.txt"
    global_config["save_res_path"] = f"{colab_drive}/predicts.txt"

    config["Optimizer"]["lr"]["learning_rate"] = settings.colab_learning_rate
    config["Optimizer"]["lr"]["warmup_epoch"] = 0
    _set_recconaug_probability(config, settings.recconaug_prob)

    config["Train"]["dataset"]["data_dir"] = f"{colab_local}/data"
    config["Train"]["dataset"]["label_file_list"] = [f"{colab_local}/data/train_list.txt"]
    config["Train"]["loader"]["num_workers"] = settings.num_workers

    config["Eval"]["dataset"]["data_dir"] = f"{colab_local}/data"
    config["Eval"]["dataset"]["label_file_list"] = [f"{colab_local}/data/val_list.txt"]
    config["Eval"]["loader"]["num_workers"] = settings.num_workers
    config["Eval"]["loader"]["shuffle"] = False
    return config


def _write_colab_runner(path: Path, settings: ColabExportConfig) -> None:
    """Write a Colab runner that installs PaddleOCR and trains the pack."""
    colab_local = settings.colab_local_dir.rstrip("/")
    colab_drive = settings.colab_drive_dir.rstrip("/")
    runner = f'''"""Train the VLPR PaddleOCR fine-tune pack on Google Colab.

Run this after mounting Google Drive and unzipping colab_training_pack.zip.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PACK_DIR = Path("{colab_local}")
DRIVE_DIR = Path("{colab_drive}")
PADDLEOCR_DIR = Path("/content/PaddleOCR")


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def require_gpu() -> None:
    try:
        subprocess.run(["nvidia-smi"], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Colab runtime hiện không có GPU. Bật GPU rồi chạy lại.") from exc


def main() -> int:
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError(
            "Google Drive chưa được mount. Chạy trước cell: "
            "from google.colab import drive; drive.mount('/content/drive')"
        )
    if not (PACK_DIR / "train_config.yml").is_file():
        raise FileNotFoundError(f"Không thấy train_config.yml trong {{PACK_DIR}}")

    require_gpu()
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "-q", "paddlepaddle-gpu==2.6.2"])
    if not PADDLEOCR_DIR.exists():
        run([
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/PaddlePaddle/PaddleOCR.git",
            str(PADDLEOCR_DIR),
        ])
    run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        cwd=PADDLEOCR_DIR,
    )
    run(
        [sys.executable, "tools/train.py", "-c", str(PACK_DIR / "train_config.yml")],
        cwd=PADDLEOCR_DIR,
    )
    print(f"Done. Checkpoints nằm ở: {{DRIVE_DIR / 'checkpoints'}}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    path.write_text(runner, encoding="utf-8")


def _load_train_config(train_config: Path) -> dict[str, Any]:
    """Load a generated PaddleOCR train config."""
    with train_config.open("r", encoding="utf-8") as stream:
        raw_config: Any = yaml.safe_load(stream)
    if not isinstance(raw_config, dict):
        raise ValueError(f"PaddleOCR train config root must be a mapping: {train_config}")
    return raw_config


def _resolve_resume_checkpoint(root: Path, raw_config: dict[str, Any]) -> Path | None:
    """Resolve the local PaddleOCR checkpoint stem when the config has one."""
    local_checkpoint = raw_config.get("Global", {}).get("checkpoints") or ""
    if not local_checkpoint:
        return None
    return resolve_project_path(root, Path(str(local_checkpoint)))


def export_colab_zip(config_path: Path | None = None) -> Path:
    """Create a zip containing data, model weights, and Colab train config."""
    settings = _load_config(config_path)
    root = project_root(config_path or Path("configs/ocr-paddleocr-finetune-v2.yaml"))
    data_dir = resolve_project_path(root, settings.data_dir)
    train_config = resolve_project_path(root, settings.train_config)
    output_dir = resolve_project_path(root, settings.output_dir)

    counts = _validate_exported_data(data_dir)
    if not train_config.is_file():
        raise FileNotFoundError(f"PaddleOCR train config not found: {train_config}")

    _safe_replace_dir(output_dir, root)
    shutil.copytree(data_dir, output_dir / "data")

    raw_config = _load_train_config(train_config)
    checkpoint_stem = _resolve_resume_checkpoint(root, raw_config)
    resume_name = (
        None
        if settings.reset_optimizer
        else _copy_checkpoint_files(checkpoint_stem, output_dir / "resume")
    )
    pretrained_name: str | None = None
    if settings.reset_optimizer:
        pretrained_name = _copy_checkpoint_weights(checkpoint_stem, output_dir / "pretrained")
    if resume_name is None and pretrained_name is None:
        pretrained_model = resolve_project_path(root, settings.pretrained_model)
        if not pretrained_model.is_file():
            raise FileNotFoundError(f"PaddleOCR pretrained model not found: {pretrained_model}")
        pretrained_dest = output_dir / "pretrained"
        pretrained_dest.mkdir(parents=True)
        shutil.copy(pretrained_model, pretrained_dest / pretrained_model.name)
        pretrained_name = pretrained_model.name

    colab_config = _rewrite_colab_config(
        raw_config,
        settings=settings,
        resume_checkpoint_name=resume_name,
        pretrained_model_name=pretrained_name,
    )
    (output_dir / "train_config.yml").write_text(
        yaml.safe_dump(colab_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    _write_colab_runner(output_dir / "run_train_colab.py", settings)

    summary: dict[str, int | str | None] = {
        **counts,
        "resume_checkpoint": resume_name,
        "reset_optimizer": str(settings.reset_optimizer),
        "pretrained_model": pretrained_name,
        "train_config": str(settings.train_config),
        "colab_drive_dir": settings.colab_drive_dir,
        "colab_epoch_num": str(settings.colab_epoch_num),
        "colab_learning_rate": str(settings.colab_learning_rate),
        "recconaug_prob": str(settings.recconaug_prob),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    zip_base = output_dir.with_suffix("")
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", output_dir))
    LOGGER.info("Colab training pack exported: %s", zip_path)
    LOGGER.info("Pack summary: %s", summary)
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
