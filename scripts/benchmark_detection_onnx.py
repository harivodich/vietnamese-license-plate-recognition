"""Benchmark the selected YOLO PyTorch checkpoint against its ONNX export."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


def _parse_args() -> argparse.Namespace:
    """Run the parse args step for this workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pt", type=Path, default=Path("artifacts/detection/yolo11n-baseline/weights/best.pt")
    )
    parser.add_argument(
        "--onnx", type=Path, default=Path("artifacts/detection/yolo11n-baseline/weights/best.onnx")
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/detection_yolo/dataset.yaml")
    )
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/benchmark/detection_onnx.json")
    )
    return parser.parse_args()


def _test_images(dataset_yaml: Path, samples: int) -> list[Path]:
    """Run the test images step for this workflow."""
    with dataset_yaml.open(encoding="utf-8") as stream:
        config: Any = yaml.safe_load(stream)
    test_list = dataset_yaml.parent / str(config["test"])
    paths = [
        Path(line.strip())
        for line in test_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    resolved = [
        (dataset_yaml.parent / path).resolve() if not path.is_absolute() else path for path in paths
    ]
    return resolved[:samples]


def _benchmark(model_path: Path, images: list[Path], args: argparse.Namespace) -> dict[str, float]:
    """Run the benchmark step for this workflow."""
    model = YOLO(str(model_path))
    for image in images[: args.warmup]:
        model.predict(str(image), imgsz=args.imgsz, device="cpu", verbose=False)
    timings: list[float] = []
    for image in images:
        started = time.perf_counter()
        model.predict(str(image), imgsz=args.imgsz, device="cpu", verbose=False)
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "mean_ms": statistics.fmean(timings),
        "p50_ms": statistics.median(timings),
        "p95_ms": statistics.quantiles(timings, n=20)[18],
        "throughput_images_per_second": 1000 / statistics.fmean(timings),
    }


def main() -> int:
    """Run the main step for this workflow."""
    args = _parse_args()
    images = _test_images(args.data, args.samples)
    if len(images) < 20:
        raise ValueError("benchmark requires at least 20 test images")
    result = {
        "samples": len(images),
        "imgsz": args.imgsz,
        "device": "cpu",
        "pytorch": _benchmark(args.pt, images, args),
        "onnx": _benchmark(args.onnx, images, args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
