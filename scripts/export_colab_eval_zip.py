"""Export a Colab-ready PaddleOCR evaluation package."""

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


class ColabEvalExportConfig(BaseModel):
    """Validated settings for packaging a fine-tuned OCR checkpoint for Colab eval."""

    model_config = ConfigDict(extra="forbid")

    data_dir: Path = Path("data/processed/ocr_finetune_paddleocr")
    train_config: Path = Path("artifacts/ocr/paddleocr-v5-mobile-finetune/train_config.yml")
    checkpoint_prefix: Path = Path(
        "artifacts/ocr/paddleocr-v5-mobile-finetune/checkpoints/best_accuracy"
    )
    output_dir: Path = Path("colab_eval_pack")
    colab_local_dir: str = "/content/colab_eval_pack"
    colab_output_dir: str = "/content/paddleocr_eval"
    use_gpu: bool = True
    batch_size: int = Field(default=64, gt=0)
    num_workers: int = Field(default=4, ge=0)


def _load_config(path: Path | None) -> ColabEvalExportConfig:
    """Load optional YAML config or use project defaults."""
    if path is None:
        return ColabEvalExportConfig()
    with path.open("r", encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Colab eval export config root must be a mapping: {path}")
    return ColabEvalExportConfig.model_validate(raw)


def _safe_replace_dir(path: Path, root: Path) -> None:
    """Replace output directory only inside the current project."""
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"refusing to delete outside project root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def _count_non_empty_lines(path: Path) -> int:
    """Count non-empty UTF-8 lines."""
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _copy_test_data(data_dir: Path, output_data_dir: Path) -> dict[str, int]:
    """Copy only test images, test labels, and the character dictionary."""
    test_list = data_dir / "test_list.txt"
    character_dict = data_dir / "dict.txt"
    if not test_list.is_file():
        raise FileNotFoundError(f"test list not found: {test_list}")
    if not character_dict.is_file():
        raise FileNotFoundError(f"character dict not found: {character_dict}")

    output_data_dir.mkdir(parents=True)
    shutil.copy(test_list, output_data_dir / "test_list.txt")
    shutil.copy(character_dict, output_data_dir / "dict.txt")

    copied_images = 0
    for line_number, line in enumerate(test_list.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            relative_image, _label = line.split("\t", maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"invalid test label row {test_list}:{line_number}") from exc
        source_image = data_dir / relative_image
        if not source_image.is_file():
            raise FileNotFoundError(
                f"test image not found {test_list}:{line_number}: {source_image}"
            )
        target_image = output_data_dir / relative_image
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_image, target_image)
        copied_images += 1

    return {
        "characters": _count_non_empty_lines(character_dict),
        "test_samples": _count_non_empty_lines(test_list),
        "test_images": copied_images,
    }


def _copy_checkpoint(checkpoint_prefix: Path, output_checkpoint_dir: Path) -> None:
    """Copy checkpoint files required for eval and optional resume traceability."""
    params = checkpoint_prefix.with_suffix(".pdparams")
    if not params.is_file():
        raise FileNotFoundError(f"checkpoint params not found: {params}")
    output_checkpoint_dir.mkdir(parents=True)
    for suffix in (".pdparams", ".pdopt", ".states"):
        source = checkpoint_prefix.with_suffix(suffix)
        if source.is_file():
            shutil.copy(source, output_checkpoint_dir / source.name)


def _rewrite_eval_config(
    config: dict[str, Any],
    *,
    settings: ColabEvalExportConfig,
) -> dict[str, Any]:
    """Rewrite train config into a Colab test-set eval config."""
    colab_local = settings.colab_local_dir.rstrip("/")
    colab_output = settings.colab_output_dir.rstrip("/")
    config["Global"]["use_gpu"] = settings.use_gpu
    config["Global"]["distributed"] = False
    config["Global"]["pretrained_model"] = ""
    config["Global"]["checkpoints"] = f"{colab_local}/checkpoints/best_accuracy"
    config["Global"]["character_dict_path"] = f"{colab_local}/data/dict.txt"
    config["Global"]["save_model_dir"] = f"{colab_output}/checkpoints"
    config["Global"]["save_inference_dir"] = f"{colab_output}/inference"
    config["Global"]["save_res_path"] = f"{colab_output}/predicts.txt"
    config["Global"]["infer_img"] = "doc/imgs_words/ch/word_1.jpg"

    config["Train"]["dataset"]["data_dir"] = f"{colab_local}/data"
    config["Train"]["dataset"]["label_file_list"] = [f"{colab_local}/data/test_list.txt"]
    config["Train"]["loader"]["shuffle"] = False
    config["Train"]["loader"]["drop_last"] = False
    config["Train"]["loader"]["batch_size_per_card"] = settings.batch_size
    config["Train"]["loader"]["num_workers"] = settings.num_workers

    config["Eval"]["dataset"]["data_dir"] = f"{colab_local}/data"
    config["Eval"]["dataset"]["label_file_list"] = [f"{colab_local}/data/test_list.txt"]
    config["Eval"]["loader"]["shuffle"] = False
    config["Eval"]["loader"]["batch_size_per_card"] = settings.batch_size
    config["Eval"]["loader"]["num_workers"] = settings.num_workers
    return config


def _write_runner(path: Path) -> None:
    """Write a small Colab runner that installs deps, clones PaddleOCR, and evaluates."""
    path.write_text(
        "\n".join(
            [
                '"""Run PaddleOCR checkpoint evaluation inside Google Colab."""',
                "",
                "from __future__ import annotations",
                "",
                "import subprocess",
                "import sys",
                "from pathlib import Path",
                "",
                "",
                "PACK_DIR = Path('/content/colab_eval_pack')",
                "PADDLEOCR_DIR = Path('/content/PaddleOCR')",
                "DEFAULT_OUTPUT_DIR = Path('/content/paddleocr_eval')",
                "GOOGLE_DRIVE_OUTPUT_DIR = Path('/content/drive/MyDrive/paddleocr_eval')",
                "",
                "",
                "def run(command: list[str], cwd: Path | None = None) -> None:",
                "    print('+', ' '.join(command), flush=True)",
                "    subprocess.run(command, cwd=cwd, check=True)",
                "",
                "",
                "def run_logged(",
                "    command: list[str],",
                "    log_path: Path,",
                "    cwd: Path | None = None,",
                ") -> None:",
                "    print('+', ' '.join(command), flush=True)",
                "    with log_path.open('w', encoding='utf-8') as stream:",
                "        process = subprocess.Popen(",
                "            command,",
                "            cwd=cwd,",
                "            stdout=subprocess.PIPE,",
                "            stderr=subprocess.STDOUT,",
                "            text=True,",
                "            bufsize=1,",
                "        )",
                "        assert process.stdout is not None",
                "        for line in process.stdout:",
                "            print(line, end='')",
                "            stream.write(line)",
                "        if process.wait() != 0:",
                "            raise subprocess.CalledProcessError(process.returncode, command)",
                "",
                "",
                "def resolve_output_dir() -> Path:",
                "    if GOOGLE_DRIVE_OUTPUT_DIR.parent.is_dir():",
                "        print('using Google Drive output:', GOOGLE_DRIVE_OUTPUT_DIR, flush=True)",
                "        return GOOGLE_DRIVE_OUTPUT_DIR",
                "    print('using local Colab output:', DEFAULT_OUTPUT_DIR, flush=True)",
                "    return DEFAULT_OUTPUT_DIR",
                "",
                "",
                "def ensure_paddle() -> None:",
                "    run([",
                "        sys.executable,",
                "        '-m',",
                "        'pip',",
                "        'install',",
                "        '-q',",
                "        'paddlepaddle-gpu==2.6.2',",
                "    ])",
                "    import paddle",
                "    print(",
                "        'paddle',",
                "        paddle.__version__,",
                "        'cuda',",
                "        paddle.device.is_compiled_with_cuda(),",
                "    )",
                "    if not paddle.device.is_compiled_with_cuda():",
                "        raise RuntimeError('PaddlePaddle GPU is not available in this runtime')",
                "",
                "",
                "def main() -> None:",
                "    if not PACK_DIR.is_dir():",
                "        raise FileNotFoundError(f'pack directory not found: {PACK_DIR}')",
                "    output_dir = resolve_output_dir()",
                "    output_dir.mkdir(parents=True, exist_ok=True)",
                "    ensure_paddle()",
                "    if not PADDLEOCR_DIR.is_dir():",
                "        run([",
                "            'git',",
                "            'clone',",
                "            '--depth',",
                "            '1',",
                "            'https://github.com/PaddlePaddle/PaddleOCR.git',",
                "            str(PADDLEOCR_DIR),",
                "        ])",
                "    run([",
                "        sys.executable,",
                "        '-m',",
                "        'pip',",
                "        'install',",
                "        '-q',",
                "        '-r',",
                "        'requirements.txt',",
                "    ], cwd=PADDLEOCR_DIR)",
                "    run_logged([",
                "        sys.executable,",
                "        'tools/eval.py',",
                "        '-c',",
                "        str(PACK_DIR / 'eval_config.yml'),",
                "    ], output_dir / 'eval_test.log', cwd=PADDLEOCR_DIR)",
                "",
                "",
                "if __name__ == '__main__':",
                "    main()",
                "",
            ]
        ),
        encoding="utf-8",
    )


def export_colab_eval_zip(config_path: Path | None = None) -> Path:
    """Create a zip containing test data, checkpoint, config, and runner script."""
    settings = _load_config(config_path)
    root = project_root(config_path or Path("configs/ocr-paddleocr-finetune.yaml"))
    data_dir = resolve_project_path(root, settings.data_dir)
    train_config = resolve_project_path(root, settings.train_config)
    checkpoint_prefix = resolve_project_path(root, settings.checkpoint_prefix)
    output_dir = resolve_project_path(root, settings.output_dir)

    if not train_config.is_file():
        raise FileNotFoundError(f"PaddleOCR train config not found: {train_config}")
    _safe_replace_dir(output_dir, root)

    summary = _copy_test_data(data_dir, output_dir / "data")
    _copy_checkpoint(checkpoint_prefix, output_dir / "checkpoints")

    with train_config.open("r", encoding="utf-8") as stream:
        raw_config: Any = yaml.safe_load(stream)
    if not isinstance(raw_config, dict):
        raise ValueError(f"PaddleOCR train config root must be a mapping: {train_config}")
    eval_config = _rewrite_eval_config(raw_config, settings=settings)
    (output_dir / "eval_config.yml").write_text(
        yaml.safe_dump(eval_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    _write_runner(output_dir / "run_eval_colab.py")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    zip_path = Path(shutil.make_archive(str(output_dir.with_suffix("")), "zip", output_dir))
    LOGGER.info("Colab eval pack exported: %s", zip_path)
    LOGGER.info("Pack summary: %s", summary)
    return zip_path


def _build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for Colab eval pack export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Colab eval pack export from CLI."""
    configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        export_colab_eval_zip(args.config)
    except (OSError, ValueError) as exc:
        LOGGER.error("Colab eval export failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
