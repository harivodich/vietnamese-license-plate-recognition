"""Benchmark OCR latency for pretrained and fine-tuned recognizers on the same samples."""

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any

import cv2

from vlpr.config import project_root, resolve_project_path
from vlpr.evaluation.ocr import (
    OcrEvaluationInputs,
    _build_recognition_inputs,
    validate_ocr_evaluation,
)
from vlpr.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkResult:
    """Represent BenchmarkResult data used by this workflow."""

    name: str
    plates: int
    recognizer_inputs: int
    repeats: int
    init_seconds: float
    mean_seconds: float
    median_seconds: float
    ms_per_plate: float
    ms_per_recognizer_input: float


class DummyArgs:
    """Minimal namespace required by PaddleOCR native TextRecognizer."""

    def __getattr__(self, item: str) -> bool:
        """Run the getattr step for this workflow."""
        return False


def _limited_inputs(config_path: Path, output_dir: Path, sample_size: int) -> OcrEvaluationInputs:
    """Run the limited inputs step for this workflow."""
    inputs = validate_ocr_evaluation(config_path)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    limit = min(sample_size, len(inputs.records))
    limited = inputs.model_copy(
        update={
            "records": inputs.records[:limit],
            "image_paths": inputs.image_paths[:limit],
        }
    )
    _build_recognition_inputs(limited, output_dir)
    return limited


def _recognition_paths(inputs: OcrEvaluationInputs, output_dir: Path) -> list[str]:
    """Run the recognition paths step for this workflow."""
    return [str(item.path) for item in _build_recognition_inputs(inputs, output_dir)]


def _measure(repeats: int, fn: Callable[[], int]) -> tuple[list[float], int]:
    """Run the measure step for this workflow."""
    durations: list[float] = []
    output_count = 0
    for _ in range(repeats):
        started = time.perf_counter()
        output_count = fn()
        durations.append(time.perf_counter() - started)
    return durations, output_count


def _benchmark_pretrained(
    *,
    config_path: Path,
    sample_size: int,
    repeats: int,
    output_dir: Path,
) -> BenchmarkResult:
    """Run the benchmark pretrained step for this workflow."""
    inputs = _limited_inputs(config_path, output_dir, sample_size)
    paths = _recognition_paths(inputs, output_dir)
    config = inputs.config

    import torch  # noqa: F401
    from paddleocr import TextRecognition

    started = time.perf_counter()
    model = TextRecognition(
        model_name=config.model_name,
        device=config.device,
        enable_mkldnn=config.enable_mkldnn,
        cpu_threads=config.cpu_threads,
    )
    init_seconds = time.perf_counter() - started
    try:
        model.predict(paths, batch_size=config.batch_size)

        def run_once() -> int:
            """Run the run once step for this workflow."""
            return len(model.predict(paths, batch_size=config.batch_size))

        durations, output_count = _measure(repeats, run_once)
    finally:
        model.close()

    if output_count != len(paths):
        raise ValueError(f"pretrained recognizer returned {output_count}, expected {len(paths)}")
    return _result("pretrained_split_compact", inputs, paths, repeats, init_seconds, durations)


def _benchmark_finetuned(
    *,
    config_path: Path,
    sample_size: int,
    repeats: int,
    output_dir: Path,
) -> BenchmarkResult:
    """Run the benchmark finetuned step for this workflow."""
    inputs = _limited_inputs(config_path, output_dir, sample_size)
    paths = _recognition_paths(inputs, output_dir)
    config = inputs.config
    root = project_root(config_path)
    paddleocr_dir = str(root / "external" / "PaddleOCR")
    if paddleocr_dir not in sys.path:
        sys.path.insert(0, paddleocr_dir)

    from tools.infer.predict_rec import TextRecognizer

    model_dir = config.model_dir
    if not model_dir:
        raise ValueError("fine-tuned benchmark config must define model_dir")
    args: Any = DummyArgs()
    args.rec_model_dir = str(resolve_project_path(root, Path(model_dir)))
    args.rec_image_shape = "3,48,320"
    args.rec_batch_num = config.batch_size
    args.rec_algorithm = "SVTR_LCNet"
    args.rec_char_dict_path = str(
        resolve_project_path(root, Path("data/processed/ocr_finetune_paddleocr/dict.txt"))
    )
    args.use_space_char = False
    args.use_onnx = False
    args.benchmark = False
    args.use_gpu = config.device == "gpu"
    args.precision = "fp32"
    args.return_word_box = False
    args.max_text_length = 25
    args.use_tensorrt = False
    args.rec_image_inverse = False

    started = time.perf_counter()
    model = TextRecognizer(args)
    init_seconds = time.perf_counter() - started

    def run_once() -> int:
        """Run the run once step for this workflow."""
        images = []
        for path in paths:
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"cannot read image: {path}")
            images.append(image)
        rec_res, _ = model(images)
        return len(rec_res)

    run_once()
    durations, output_count = _measure(repeats, run_once)
    if output_count != len(paths):
        raise ValueError(f"fine-tuned recognizer returned {output_count}, expected {len(paths)}")
    return _result("finetuned_split_compact", inputs, paths, repeats, init_seconds, durations)


def _result(
    name: str,
    inputs: OcrEvaluationInputs,
    paths: Sequence[str],
    repeats: int,
    init_seconds: float,
    durations: Sequence[float],
) -> BenchmarkResult:
    """Run the result step for this workflow."""
    mean_seconds = fmean(durations)
    return BenchmarkResult(
        name=name,
        plates=len(inputs.records),
        recognizer_inputs=len(paths),
        repeats=repeats,
        init_seconds=init_seconds,
        mean_seconds=mean_seconds,
        median_seconds=median(durations),
        ms_per_plate=mean_seconds * 1000 / len(inputs.records),
        ms_per_recognizer_input=mean_seconds * 1000 / len(paths),
    )


def benchmark_ocr_latency(
    *,
    baseline_config: Path,
    finetune_config: Path,
    sample_size: int,
    repeats: int,
    output_path: Path,
) -> dict[str, Any]:
    """Run the benchmark ocr latency step for this workflow."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    output_dir = output_path.parent / "benchmark_inputs"
    baseline = _benchmark_pretrained(
        config_path=baseline_config,
        sample_size=sample_size,
        repeats=repeats,
        output_dir=output_dir / "baseline",
    )
    finetuned = _benchmark_finetuned(
        config_path=finetune_config,
        sample_size=sample_size,
        repeats=repeats,
        output_dir=output_dir / "finetuned",
    )
    summary = {
        "baseline": baseline.__dict__,
        "finetuned": finetuned.__dict__,
        "notes": [
            "Both runs use split-compact layout on the same number of plate records.",
            "Timing excludes model initialization but includes image loading and OCR inference.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    """Run the build parser step for this workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/ocr-baseline-layout.yaml"),
    )
    parser.add_argument(
        "--finetune-config",
        type=Path,
        default=Path("configs/ocr-finetune-eval.yaml"),
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ocr/paddleocr-v5-mobile-finetune/eval/latency_benchmark.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the main step for this workflow."""
    configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        root = project_root(args.finetune_config)
        summary = benchmark_ocr_latency(
            baseline_config=resolve_project_path(root, args.baseline_config),
            finetune_config=resolve_project_path(root, args.finetune_config),
            sample_size=args.sample_size,
            repeats=args.repeats,
            output_path=resolve_project_path(root, args.output),
        )
    except (ImportError, OSError, ValueError) as exc:
        LOGGER.error("OCR latency benchmark failed: %s", exc)
        return 1
    LOGGER.info("OCR latency benchmark saved: %s", args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
