"""Train và resume CRNN+CTC trên OCR line dataset đã materialize."""

import argparse
import csv
import logging
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from vlpr.config import project_root, resolve_project_path
from vlpr.evaluation.ocr import levenshtein_distance
from vlpr.models.crnn import CrnnCtc, OcrCharset
from vlpr.training.ocr_config import (
    OcrAugmentationSettings,
    OcrTrainingExperimentConfig,
    load_ocr_training_config,
)
from vlpr.training.reproducibility import set_random_seed
from vlpr.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrLineSample:
    """Một ảnh dòng đã materialize và label ký tự tương ứng."""

    image_path: Path
    label: str


@dataclass(frozen=True)
class OcrBatch:
    """Tensor ảnh cùng targets dạng nối tiếp mà CTCLoss yêu cầu."""

    images: Tensor
    targets: Tensor
    target_lengths: Tensor
    labels: tuple[str, ...]


@dataclass(frozen=True)
class OcrEpochMetrics:
    """Loss và recognition metrics của một epoch validation."""

    loss: float
    exact_match: float
    cer: float
    character_accuracy: float


@dataclass(frozen=True)
class OcrTrainingInputs:
    """Config, dataset paths, charset và output đã qua preflight."""

    config: OcrTrainingExperimentConfig
    data_root: Path
    train_labels: Path
    validation_labels: Path
    charset_path: Path
    output_dir: Path
    charset: OcrCharset
    train_samples: tuple[OcrLineSample, ...]
    validation_samples: tuple[OcrLineSample, ...]


def _read_line_samples(
    data_root: Path,
    label_path: Path,
    charset: OcrCharset,
) -> tuple[OcrLineSample, ...]:
    """Đọc `path<TAB>label`, kiểm tra path, file, label và charset."""
    samples: list[OcrLineSample] = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"{label_path}:{line_number}: cần path<TAB>label")
        raw_path, label = fields
        image_path = (data_root / raw_path).resolve()
        if not image_path.is_relative_to(data_root.resolve()):
            raise ValueError(f"{label_path}:{line_number}: image thoát data root")
        if not image_path.is_file():
            raise FileNotFoundError(f"{label_path}:{line_number}: thiếu image {image_path}")
        if not label:
            raise ValueError(f"{label_path}:{line_number}: label rỗng")
        charset.encode(label)
        samples.append(OcrLineSample(image_path=image_path, label=label))
    if not samples:
        raise ValueError(f"OCR label file rỗng: {label_path}")
    return tuple(samples)


def validate_ocr_training_experiment(config_path: Path) -> OcrTrainingInputs:
    """Xác nhận dataset, charset, CUDA và CTC length trước khi train."""
    config = load_ocr_training_config(config_path)
    root = project_root(config_path)
    data_root = resolve_project_path(root, config.data.output_dir)
    train_labels = data_root / "train.txt"
    validation_labels = data_root / "validation.txt"
    charset_path = data_root / "charset.txt"
    for required in (train_labels, validation_labels, charset_path):
        if not required.is_file():
            raise FileNotFoundError(f"chưa chuẩn bị OCR training data: {required}")
    charset = OcrCharset.from_file(charset_path)
    train_samples = _read_line_samples(data_root, train_labels, charset)
    validation_samples = _read_line_samples(data_root, validation_labels, charset)
    if config.train.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("config yêu cầu CUDA nhưng PyTorch không thấy GPU")

    model = build_crnn(config, charset)
    with torch.no_grad():
        steps = model(torch.zeros(1, 1, config.model.image_height, config.model.image_width)).shape[
            0
        ]
    max_label_length = max(len(sample.label) for sample in (*train_samples, *validation_samples))
    if max_label_length > steps:
        raise ValueError(f"label dài nhất {max_label_length} vượt CTC timesteps {steps}")

    output_dir = resolve_project_path(root, config.output.project) / config.output.name
    return OcrTrainingInputs(
        config=config,
        data_root=data_root,
        train_labels=train_labels,
        validation_labels=validation_labels,
        charset_path=charset_path,
        output_dir=output_dir,
        charset=charset,
        train_samples=train_samples,
        validation_samples=validation_samples,
    )


def _augment(image: Image.Image, settings: OcrAugmentationSettings) -> Image.Image:
    """Áp dụng nhiễu nhẹ, giữ nguyên thứ tự và nội dung ký tự."""
    if settings.rotation_degrees:
        angle = random.uniform(-settings.rotation_degrees, settings.rotation_degrees)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=255)
    if settings.brightness:
        factor = random.uniform(1.0 - settings.brightness, 1.0 + settings.brightness)
        image = ImageEnhance.Brightness(image).enhance(factor)
    if settings.contrast:
        factor = random.uniform(1.0 - settings.contrast, 1.0 + settings.contrast)
        image = ImageEnhance.Contrast(image).enhance(factor)
    if random.random() < settings.blur_probability:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.7)))
    return image


def preprocess_ocr_image(
    image_path: Path,
    *,
    image_height: int,
    image_width: int,
    augmentation: OcrAugmentationSettings | None,
) -> Tensor:
    """Resize giữ tỉ lệ, pad trắng và chuẩn hóa grayscale về `[-1, 1]`."""
    with Image.open(image_path) as opened:
        image = opened.convert("L")
    if augmentation is not None:
        image = _augment(image, augmentation)
    scale = min(image_width / image.width, image_height / image.height)
    resized_width = max(1, min(image_width, round(image.width * scale)))
    resized_height = max(1, min(image_height, round(image.height * scale)))
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (image_width, image_height), color=255)
    vertical_offset = (image_height - resized_height) // 2
    canvas.paste(resized, (0, vertical_offset))
    array = np.asarray(canvas, dtype=np.float32).copy()
    return torch.from_numpy(array).unsqueeze(0) / 127.5 - 1.0


class OcrLineDataset(Dataset[tuple[Tensor, str]]):
    """Lazy image dataset để không giữ hàng nghìn crop trong RAM."""

    def __init__(
        self,
        samples: tuple[OcrLineSample, ...],
        *,
        image_height: int,
        image_width: int,
        augmentation: OcrAugmentationSettings | None,
    ) -> None:
        """Lưu metadata; ảnh chỉ được mở khi DataLoader yêu cầu."""
        self.samples = samples
        self.image_height = image_height
        self.image_width = image_width
        self.augmentation = augmentation

    def __len__(self) -> int:
        """Trả số line samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        """Đọc và preprocess đúng một sample."""
        sample = self.samples[index]
        return (
            preprocess_ocr_image(
                sample.image_path,
                image_height=self.image_height,
                image_width=self.image_width,
                augmentation=self.augmentation,
            ),
            sample.label,
        )


class OcrCollator:
    """Đóng gói variable-length labels theo định dạng CTCLoss."""

    def __init__(self, charset: OcrCharset) -> None:
        """Giữ charset dùng chung cho mọi batch."""
        self.charset = charset

    def __call__(self, items: list[tuple[Tensor, str]]) -> OcrBatch:
        """Stack ảnh, nối targets và lưu target lengths."""
        images, labels = zip(*items, strict=True)
        encoded = [self.charset.encode(label) for label in labels]
        targets = torch.tensor(
            [index for sequence in encoded for index in sequence],
            dtype=torch.long,
        )
        target_lengths = torch.tensor(
            [len(sequence) for sequence in encoded],
            dtype=torch.long,
        )
        return OcrBatch(
            images=torch.stack(images),
            targets=targets,
            target_lengths=target_lengths,
            labels=tuple(labels),
        )


def build_crnn(
    config: OcrTrainingExperimentConfig,
    charset: OcrCharset,
) -> CrnnCtc:
    """Khởi tạo CRNN đúng capacity đã đóng băng trong config."""
    return CrnnCtc(
        num_classes=charset.num_classes,
        hidden_size=config.model.hidden_size,
        lstm_layers=config.model.lstm_layers,
        dropout=config.model.dropout,
    )


def _build_loaders(
    inputs: OcrTrainingInputs,
) -> tuple[DataLoader[OcrBatch], DataLoader[OcrBatch]]:
    """Tạo train/validation loaders với shuffle chỉ ở train."""
    config = inputs.config
    train_dataset = OcrLineDataset(
        inputs.train_samples,
        image_height=config.model.image_height,
        image_width=config.model.image_width,
        augmentation=config.augmentation,
    )
    validation_dataset = OcrLineDataset(
        inputs.validation_samples,
        image_height=config.model.image_height,
        image_width=config.model.image_width,
        augmentation=None,
    )
    collator = OcrCollator(inputs.charset)
    generator = torch.Generator().manual_seed(config.train.seed)
    common: dict[str, Any] = {
        "batch_size": config.train.batch_size,
        "num_workers": config.train.workers,
        "pin_memory": config.train.device.startswith("cuda"),
        "persistent_workers": config.train.workers > 0,
        "collate_fn": collator,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return (
        cast(DataLoader[OcrBatch], train_loader),
        cast(DataLoader[OcrBatch], validation_loader),
    )


def _ctc_loss(
    criterion: nn.CTCLoss,
    log_probabilities: Tensor,
    batch: OcrBatch,
) -> Tensor:
    """Tính CTCLoss với cùng input length cho ảnh đã pad về width cố định."""
    input_lengths = torch.full(
        (log_probabilities.shape[1],),
        log_probabilities.shape[0],
        dtype=torch.long,
        device=batch.target_lengths.device,
    )
    return cast(
        Tensor,
        criterion(
            log_probabilities,
            batch.targets,
            input_lengths,
            batch.target_lengths,
        ),
    )


def _move_batch(batch: OcrBatch, device: torch.device) -> OcrBatch:
    """Chuyển tensor sang device, giữ labels Python phục vụ metric."""
    return OcrBatch(
        images=batch.images.to(device, non_blocking=True),
        targets=batch.targets.to(device, non_blocking=True),
        target_lengths=batch.target_lengths,
        labels=batch.labels,
    )


def _train_epoch(
    model: CrnnCtc,
    loader: DataLoader[OcrBatch],
    criterion: nn.CTCLoss,
    optimizer: AdamW,
    *,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    """Tối ưu một epoch và trả mean loss theo batch."""
    model.train()
    losses: list[float] = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = _ctc_loss(criterion, model(batch.images), batch)
        loss.backward()  # type: ignore[no-untyped-call]  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _decode_batch(log_probabilities: Tensor, charset: OcrCharset) -> tuple[str, ...]:
    """Greedy decode toàn batch từ `[time, batch, classes]`."""
    indices = log_probabilities.argmax(dim=2).transpose(0, 1).cpu().tolist()
    return tuple(charset.decode(sequence) for sequence in indices)


def _validate_epoch(
    model: CrnnCtc,
    loader: DataLoader[OcrBatch],
    criterion: nn.CTCLoss,
    charset: OcrCharset,
    *,
    device: torch.device,
) -> OcrEpochMetrics:
    """Đo loss, exact match và micro CER không cập nhật model."""
    model.eval()
    losses: list[float] = []
    exact_matches = 0
    edit_distance = 0
    character_count = 0
    sample_count = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            log_probabilities = model(batch.images)
            losses.append(float(_ctc_loss(criterion, log_probabilities, batch).cpu()))
            predictions = _decode_batch(log_probabilities, charset)
            for prediction, label in zip(predictions, batch.labels, strict=True):
                exact_matches += prediction == label
                edit_distance += levenshtein_distance(label, prediction)
                character_count += len(label)
                sample_count += 1
    cer = edit_distance / character_count
    return OcrEpochMetrics(
        loss=float(np.mean(losses)),
        exact_match=exact_matches / sample_count,
        cer=cer,
        character_accuracy=max(0.0, 1.0 - cer),
    )


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: CrnnCtc,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    best_exact_match: float,
    stale_epochs: int,
    inputs: OcrTrainingInputs,
) -> None:
    """Ghi checkpoint tạm rồi replace để tránh file dở khi máy tắt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),  # type: ignore[no-untyped-call]
            "best_exact_match": best_exact_match,
            "stale_epochs": stale_epochs,
            "charset": inputs.charset.characters,
            "model_config": inputs.config.model.model_dump(),
        },
        temporary,
    )
    temporary.replace(path)


def _append_history(
    path: Path,
    *,
    epoch: int,
    train_loss: float,
    validation: OcrEpochMetrics,
    learning_rate: float,
    epoch_seconds: float,
) -> None:
    """Append một epoch gồm loss, metric, learning rate và thời gian."""
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        if not exists:
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "validation_loss",
                    "exact_match",
                    "cer",
                    "character_accuracy",
                    "learning_rate",
                    "epoch_seconds",
                ]
            )
        writer.writerow(
            [
                epoch,
                train_loss,
                validation.loss,
                validation.exact_match,
                validation.cer,
                validation.character_accuracy,
                learning_rate,
                epoch_seconds,
            ]
        )


def _plot_history(path: Path, output_path: Path) -> None:
    """Vẽ loss, exact match và CER từ CSV đã flush."""
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], label="train")
    axes[0].plot(
        epochs,
        [float(row["validation_loss"]) for row in rows],
        label="validation",
    )
    axes[0].set_title("CTC loss")
    axes[0].legend()
    axes[1].plot(
        epochs,
        [float(row["exact_match"]) for row in rows],
        label="exact match",
    )
    axes[1].plot(epochs, [float(row["cer"]) for row in rows], label="CER")
    axes[1].set_title("Validation metrics")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _load_resume(
    checkpoint_path: Path,
    *,
    model: CrnnCtc,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    inputs: OcrTrainingInputs,
    device: torch.device,
) -> tuple[int, float, int]:
    """Khôi phục model/optimizer/scheduler và từ chối config hoặc charset khác."""
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if tuple(checkpoint["charset"]) != inputs.charset.characters:
        raise ValueError("resume checkpoint dùng charset khác")
    if checkpoint["model_config"] != inputs.config.model.model_dump():
        raise ValueError("resume checkpoint dùng model config khác")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_exact_match"]),
        int(checkpoint["stale_epochs"]),
    )


def train_ocr(
    config_path: Path,
    *,
    resume_path: Path | None = None,
) -> OcrTrainingInputs:
    """Train CRNN, lưu best/last/periodic checkpoints, history và curves."""
    inputs = validate_ocr_training_experiment(config_path)
    config = inputs.config
    set_random_seed(config.train.seed, deterministic=config.train.deterministic)
    device = torch.device(config.train.device)
    model = build_crnn(config, inputs.charset).to(device)
    criterion = nn.CTCLoss(blank=inputs.charset.blank_index, zero_infinity=True)
    optimizer = AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.train.epochs,
        eta_min=config.train.learning_rate * 0.01,
    )
    start_epoch = 1
    best_exact_match = -1.0
    stale_epochs = 0
    if resume_path is not None:
        root = project_root(config_path)
        checkpoint_path = resolve_project_path(root, resume_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"không tìm thấy OCR checkpoint: {checkpoint_path}")
        start_epoch, best_exact_match, stale_epochs = _load_resume(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            inputs=inputs,
            device=device,
        )

    train_loader, validation_loader = _build_loaders(inputs)
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = inputs.output_dir / "history.csv"
    for epoch in range(start_epoch, config.train.epochs + 1):
        started = time.perf_counter()
        train_loss = _train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device=device,
            gradient_clip_norm=config.train.gradient_clip_norm,
        )
        validation = _validate_epoch(
            model,
            validation_loader,
            criterion,
            inputs.charset,
            device=device,
        )
        epoch_seconds = time.perf_counter() - started
        learning_rate = optimizer.param_groups[0]["lr"]
        _append_history(
            history_path,
            epoch=epoch,
            train_loss=train_loss,
            validation=validation,
            learning_rate=learning_rate,
            epoch_seconds=epoch_seconds,
        )
        scheduler.step()
        improved = validation.exact_match > best_exact_match
        stale_epochs = 0 if improved else stale_epochs + 1
        if improved:
            best_exact_match = validation.exact_match
            _save_checkpoint(
                inputs.output_dir / "best.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_exact_match=best_exact_match,
                stale_epochs=stale_epochs,
                inputs=inputs,
            )
        _save_checkpoint(
            inputs.output_dir / "last.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_exact_match=best_exact_match,
            stale_epochs=stale_epochs,
            inputs=inputs,
        )
        if epoch % config.train.save_period == 0:
            _save_checkpoint(
                inputs.output_dir / f"epoch_{epoch:03d}.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_exact_match=best_exact_match,
                stale_epochs=stale_epochs,
                inputs=inputs,
            )
        _plot_history(history_path, inputs.output_dir / "training_curves.png")
        LOGGER.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f exact=%.4f CER=%.4f seconds=%.1f",
            epoch,
            train_loss,
            validation.loss,
            validation.exact_match,
            validation.cer,
            epoch_seconds,
        )
        if config.train.patience and stale_epochs >= config.train.patience:
            LOGGER.info("early stopping after %d stale epochs", stale_epochs)
            break
    return inputs


def smoke_test_ocr(config_path: Path) -> None:
    """Chạy một forward/backward batch và một validation batch, không lưu artifact."""
    inputs = validate_ocr_training_experiment(config_path)
    config = inputs.config
    set_random_seed(config.train.seed, deterministic=config.train.deterministic)
    device = torch.device(config.train.device)
    model = build_crnn(config, inputs.charset).to(device)
    criterion = nn.CTCLoss(blank=inputs.charset.blank_index, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate)
    train_loader, validation_loader = _build_loaders(inputs)
    batch = _move_batch(next(iter(train_loader)), device)
    optimizer.zero_grad(set_to_none=True)
    loss = _ctc_loss(criterion, model(batch.images), batch)
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    validation_batch = _move_batch(next(iter(validation_loader)), device)
    with torch.no_grad():
        validation_loss = _ctc_loss(
            criterion,
            model(validation_batch.images),
            validation_batch,
        )
    LOGGER.info(
        "OCR smoke test passed train_loss=%.4f validation_loss=%.4f",
        float(loss.detach().cpu()),
        float(validation_loss.cpu()),
    )


def _build_parser() -> argparse.ArgumentParser:
    """Tạo CLI với ba mode loại trừ nhau."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ocr-crnn.yaml"),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--resume", type=Path, metavar="LAST_PT")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Điều phối check, smoke, train và resume thành exit code CLI."""
    configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        if args.check_only:
            inputs = validate_ocr_training_experiment(args.config)
            LOGGER.info(
                "OCR training preflight passed train=%d validation=%d classes=%d",
                len(inputs.train_samples),
                len(inputs.validation_samples),
                inputs.charset.num_classes,
            )
        elif args.smoke_test:
            smoke_test_ocr(args.config)
        else:
            train_ocr(args.config, resume_path=args.resume)
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("OCR training failed: %s", exc)
        return 1
    return 0
