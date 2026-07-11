"""Check PyTorch and ONNX detection outputs agree before deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO

from vlpr.evaluation.detection_analysis import intersection_over_union


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
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/benchmark/detection_equivalence.json")
    )
    return parser.parse_args()


def _images(dataset_yaml: Path, samples: int) -> list[Path]:
    """Run the images step for this workflow."""
    config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    test_list = dataset_yaml.parent / config["test"]
    return [
        (dataset_yaml.parent / line.strip()).resolve()
        for line in test_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:samples]


def _boxes(result: object) -> list[tuple[tuple[float, float, float, float], float]]:
    """Run the boxes step for this workflow."""
    boxes = result.boxes
    return [
        (tuple(map(float, box)), float(confidence))
        for box, confidence in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), strict=True)
    ]


def main() -> int:
    """Run the main step for this workflow."""
    args = _parse_args()
    images = _images(args.data, args.samples)
    pt_model, onnx_model = YOLO(str(args.pt)), YOLO(str(args.onnx))
    matched, pt_total, onnx_total = 0, 0, 0
    ious, confidence_deltas = [], []
    for image in images:
        pt_boxes = _boxes(
            pt_model.predict(
                str(image), conf=args.conf, imgsz=args.imgsz, device="cpu", verbose=False
            )[0]
        )
        onnx_boxes = _boxes(
            onnx_model.predict(
                str(image), conf=args.conf, imgsz=args.imgsz, device="cpu", verbose=False
            )[0]
        )
        pt_total += len(pt_boxes)
        onnx_total += len(onnx_boxes)
        used: set[int] = set()
        for pt_box, pt_confidence in pt_boxes:
            candidates = [
                (intersection_over_union(pt_box, onnx_box), index, onnx_confidence)
                for index, (onnx_box, onnx_confidence) in enumerate(onnx_boxes)
                if index not in used
            ]
            if candidates:
                iou, index, onnx_confidence = max(candidates)
                if iou >= 0.9:
                    used.add(index)
                    matched += 1
                    ious.append(iou)
                    confidence_deltas.append(abs(pt_confidence - onnx_confidence))
    result = {
        "images": len(images),
        "pt_detections": pt_total,
        "onnx_detections": onnx_total,
        "matched_iou_ge_0_9": matched,
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "max_confidence_delta": max(confidence_deltas, default=0.0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
