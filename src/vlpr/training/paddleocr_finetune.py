"""Prepare PaddleOCR recognition fine-tuning config and command."""

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from vlpr.config import project_root, resolve_project_path
from vlpr.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


class PaddleOcrRepoConfig(BaseModel):
    """Location of the official PaddleOCR training repository."""

    model_config = ConfigDict(extra="forbid")

    repo_dir: Path
    base_config: Path
    train_script: Path


class PaddleOcrFinetuneDataConfig(BaseModel):
    """Exported line-level OCR dataset paths used by PaddleOCR."""

    model_config = ConfigDict(extra="forbid")

    export_dir: Path
    train_list: Path
    validation_list: Path
    test_list: Path
    character_dict: Path


class PaddleOcrFinetuneOutputConfig(BaseModel):
    """Where generated train config and checkpoints should live."""

    model_config = ConfigDict(extra="forbid")

    project: Path
    name: str = Field(min_length=1)
    train_config: Path


class PaddleOcrFinetuneTrainConfig(BaseModel):
    """Small set of train knobs project code owns before handing off to PaddleOCR."""

    model_config = ConfigDict(extra="forbid")

    use_gpu: bool
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    warmup_epoch: int = Field(ge=0)
    batch_size_per_card: int = Field(gt=0)
    eval_batch_step: tuple[int, int]
    save_epoch_step: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    pretrained_model: str = ""
    resume_from: str = ""
    use_space_char: bool = False


class PaddleOcrFinetuneConfig(BaseModel):
    """Strict project config for generating a PaddleOCR training config."""

    model_config = ConfigDict(extra="forbid")

    paddleocr: PaddleOcrRepoConfig
    data: PaddleOcrFinetuneDataConfig
    output: PaddleOcrFinetuneOutputConfig
    train: PaddleOcrFinetuneTrainConfig


class PaddleOcrFinetunePrepared(BaseModel):
    """Resolved result of a successful fine-tune preflight."""

    model_config = ConfigDict(extra="forbid")

    train_config: Path
    working_dir: Path
    command: tuple[str, ...]
    train_samples: int
    validation_samples: int
    test_samples: int
    characters: int


def load_paddleocr_finetune_config(path: Path) -> PaddleOcrFinetuneConfig:
    """Load strict YAML so bad training handoff settings fail before PaddleOCR starts."""
    with path.open("r", encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"PaddleOCR fine-tune config root must be a mapping: {path}")
    return PaddleOcrFinetuneConfig.model_validate(raw)


def _count_label_lines(path: Path) -> int:
    """Count non-empty PaddleOCR label rows."""
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _validate_label_file(path: Path, export_dir: Path) -> int:
    """Validate PaddleOCR `relative_path<TAB>label` rows and referenced images."""
    count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            relative_image, label = line.split("\t", maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"invalid PaddleOCR label row {path}:{line_number}") from exc
        if not label:
            raise ValueError(f"empty PaddleOCR label {path}:{line_number}")
        image_path = (export_dir / relative_image).resolve()
        if not image_path.is_relative_to(export_dir.resolve()):
            raise ValueError(
                f"label image escapes export dir {path}:{line_number}: {relative_image}"
            )
        if not image_path.is_file():
            raise FileNotFoundError(f"label image not found {path}:{line_number}: {image_path}")
        count += 1
    if count == 0:
        raise ValueError(f"PaddleOCR label file is empty: {path}")
    return count


def _set_nested(config: dict[str, Any], keys: Sequence[str], value: Any) -> None:
    """Set a nested dict value while keeping official config structure intact."""
    current = config
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[keys[-1]] = value


def _resolve_optional_project_path(root: Path, value: str) -> str:
    """Resolve optional project-relative paths while preserving empty settings."""
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = (root / path).resolve()
    return str(path)


def _validate_resume_checkpoint(path: str) -> str:
    """Validate PaddleOCR resume prefix or `.pdparams` path before starting training."""
    if not path:
        return ""
    checkpoint = Path(path)
    params_path = checkpoint if checkpoint.suffix == ".pdparams" else Path(f"{checkpoint}.pdparams")
    if not params_path.is_file():
        raise FileNotFoundError(f"PaddleOCR resume checkpoint not found: {params_path}")
    if checkpoint.suffix == ".pdparams":
        return str(checkpoint.with_suffix(""))
    return str(checkpoint)


def _update_paddleocr_config(
    base_config: dict[str, Any],
    *,
    root: Path,
    repo_dir: Path,
    export_dir: Path,
    output_dir: Path,
    project_config: PaddleOcrFinetuneConfig,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    """Apply project-specific dataset, charset, checkpoint, and train settings."""
    config = dict(base_config)
    train = project_config.train
    data = project_config.data

    character_dict = (export_dir / data.character_dict).resolve()
    train_list = (export_dir / data.train_list).resolve()
    validation_list = (export_dir / data.validation_list).resolve()

    _set_nested(config, ("Global", "use_gpu"), train.use_gpu)
    _set_nested(config, ("Global", "epoch_num"), train.epochs)
    _set_nested(config, ("Global", "save_epoch_step"), train.save_epoch_step)
    _set_nested(config, ("Global", "eval_batch_step"), list(train.eval_batch_step))
    _set_nested(config, ("Global", "character_dict_path"), str(character_dict))
    _set_nested(config, ("Global", "use_space_char"), train.use_space_char)
    _set_nested(config, ("Global", "save_model_dir"), str((output_dir / "checkpoints").resolve()))
    _set_nested(config, ("Global", "save_inference_dir"), str((output_dir / "inference").resolve()))
    _set_nested(config, ("Global", "save_res_path"), str((output_dir / "predicts.txt").resolve()))
    configured_resume = str(resume_from) if resume_from is not None else train.resume_from
    resume_checkpoint = _validate_resume_checkpoint(
        _resolve_optional_project_path(root, configured_resume)
    )
    _set_nested(config, ("Global", "checkpoints"), resume_checkpoint)
    pretrained_model = train.pretrained_model
    if pretrained_model:
        pretrained_path = Path(pretrained_model)
        if not pretrained_path.is_absolute():
            pretrained_path = (root / pretrained_path).resolve()
        if not pretrained_path.is_file():
            raise FileNotFoundError(f"PaddleOCR pretrained model not found: {pretrained_path}")
        pretrained_model = str(pretrained_path)
    _set_nested(config, ("Global", "pretrained_model"), pretrained_model)

    _set_nested(config, ("Optimizer", "lr", "learning_rate"), train.learning_rate)
    _set_nested(config, ("Optimizer", "lr", "warmup_epoch"), train.warmup_epoch)

    _set_nested(config, ("Train", "dataset", "data_dir"), str(export_dir.resolve()))
    _set_nested(config, ("Train", "dataset", "label_file_list"), [str(train_list)])
    _set_nested(config, ("Train", "loader", "batch_size_per_card"), train.batch_size_per_card)
    _set_nested(config, ("Train", "loader", "num_workers"), train.num_workers)
    _set_nested(config, ("Train", "sampler", "first_bs"), train.batch_size_per_card)

    _set_nested(config, ("Eval", "dataset", "data_dir"), str(export_dir.resolve()))
    _set_nested(config, ("Eval", "dataset", "label_file_list"), [str(validation_list)])
    _set_nested(config, ("Eval", "loader", "shuffle"), False)
    _set_nested(config, ("Eval", "loader", "batch_size_per_card"), train.batch_size_per_card)
    _set_nested(config, ("Eval", "loader", "num_workers"), train.num_workers)

    # Make relative paths inside official PaddleOCR config resolve from the cloned repository.
    _set_nested(
        config, ("Global", "infer_img"), str((repo_dir / "doc/imgs_words/ch/word_1.jpg").resolve())
    )
    return config


def prepare_paddleocr_finetune(
    config_path: Path,
    *,
    resume_from: Path | None = None,
) -> PaddleOcrFinetunePrepared:
    """Validate exported data and write the actual PaddleOCR train YAML."""
    project_config = load_paddleocr_finetune_config(config_path)
    root = project_root(config_path)

    repo_dir = resolve_project_path(root, project_config.paddleocr.repo_dir)
    train_script = repo_dir / project_config.paddleocr.train_script
    base_config_path = repo_dir / project_config.paddleocr.base_config
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"official PaddleOCR repo not found: {repo_dir}")
    if not train_script.is_file():
        raise FileNotFoundError(f"PaddleOCR train script not found: {train_script}")
    if not base_config_path.is_file():
        raise FileNotFoundError(f"PaddleOCR base config not found: {base_config_path}")

    export_dir = resolve_project_path(root, project_config.data.export_dir)
    if not export_dir.is_dir():
        raise FileNotFoundError(f"OCR fine-tune export dir not found: {export_dir}")
    character_dict = export_dir / project_config.data.character_dict
    if not character_dict.is_file():
        raise FileNotFoundError(f"character dict not found: {character_dict}")

    train_count = _validate_label_file(export_dir / project_config.data.train_list, export_dir)
    validation_count = _validate_label_file(
        export_dir / project_config.data.validation_list,
        export_dir,
    )
    test_count = _validate_label_file(export_dir / project_config.data.test_list, export_dir)
    characters = _count_label_lines(character_dict)
    if characters == 0:
        raise ValueError(f"character dict is empty: {character_dict}")
    if project_config.train.use_gpu:
        pass
        # import paddle

        # if not paddle.device.is_compiled_with_cuda():
        #     raise ValueError(
        #         "config train.use_gpu=true but installed PaddlePaddle is CPU-only; "
        #         "install a CUDA-enabled PaddlePaddle build or set train.use_gpu=false"
        #     )

    with base_config_path.open("r", encoding="utf-8") as stream:
        base_raw: Any = yaml.safe_load(stream)
    if not isinstance(base_raw, dict):
        raise ValueError(f"PaddleOCR base config root must be a mapping: {base_config_path}")

    output_dir = (
        resolve_project_path(root, project_config.output.project) / project_config.output.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_config_path = output_dir / project_config.output.train_config
    train_config = _update_paddleocr_config(
        base_raw,
        root=root,
        repo_dir=repo_dir,
        export_dir=export_dir,
        output_dir=output_dir,
        project_config=project_config,
        resume_from=resume_from,
    )
    train_config_path.write_text(
        yaml.safe_dump(train_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    command = (
        sys.executable,
        str(train_script.resolve()),
        "-c",
        str(train_config_path.resolve()),
    )
    summary = {
        "characters": characters,
        "command": list(command),
        "test_samples": test_count,
        "train_config": str(train_config_path.resolve()),
        "train_samples": train_count,
        "validation_samples": validation_count,
    }
    (output_dir / "preflight_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PaddleOcrFinetunePrepared(
        train_config=train_config_path,
        command=command,
        working_dir=repo_dir,
        train_samples=train_count,
        validation_samples=validation_count,
        test_samples=test_count,
        characters=characters,
    )


def run_paddleocr_finetune(config_path: Path, *, resume_from: Path | None = None) -> int:
    """Run official PaddleOCR training after the generated config passes preflight."""
    prepared = prepare_paddleocr_finetune(config_path, resume_from=resume_from)
    LOGGER.info("starting PaddleOCR fine-tune command=%s", " ".join(prepared.command))
    completed = subprocess.run(prepared.command, cwd=prepared.working_dir)
    return completed.returncode


def _build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for preflight or training handoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ocr-paddleocr-finetune.yaml"),
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from PaddleOCR checkpoint prefix or .pdparams path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare PaddleOCR train config, optionally launching the official trainer."""
    configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        if args.run:
            return run_paddleocr_finetune(args.config, resume_from=args.resume)
        prepared = prepare_paddleocr_finetune(args.config, resume_from=args.resume)
    except (OSError, ValueError) as exc:
        LOGGER.error("PaddleOCR fine-tune preflight failed: %s", exc)
        return 1
    LOGGER.info(
        "PaddleOCR fine-tune ready train=%d validation=%d test=%d chars=%d config=%s",
        prepared.train_samples,
        prepared.validation_samples,
        prepared.test_samples,
        prepared.characters,
        prepared.train_config,
    )
    LOGGER.info("Train command: %s", " ".join(prepared.command))
    return 0
