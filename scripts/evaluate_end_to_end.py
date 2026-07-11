"""Evaluate the complete detector-to-OCR pipeline on external labeled vehicle images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from predict_end_to_end import predict
from vlpr.evaluation.ocr import levenshtein_distance, normalize_plate_text


class ExternalTestRecord(BaseModel):
    """One externally sourced image with one plate text used only for final evaluation."""

    model_config = ConfigDict(extra="forbid")

    image_path: str = Field(min_length=1)
    plate_text: str = Field(min_length=1)


def _parse_args() -> argparse.Namespace:
    """Run the parse args step for this workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/external/manifest.jsonl"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/end_to_end/external_test")
    )
    parser.add_argument(
        "--detection-checkpoint",
        type=Path,
        default=Path("artifacts/detection/yolo11n-baseline/weights/best.pt"),
    )
    parser.add_argument(
        "--ocr-model-dir",
        type=Path,
        default=Path("artifacts/ocr/paddleocr-v5-mobile-finetune/inference"),
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-ocr-confidence", type=float, default=0.8)
    return parser.parse_args()


def _read_manifest(path: Path) -> tuple[ExternalTestRecord, ...]:
    """Run the read manifest step for this workflow."""
    if not path.is_file():
        raise FileNotFoundError(f"External test manifest not found: {path}")
    records: list[ExternalTestRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(ExternalTestRecord.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"Invalid JSONL row {path}:{line_number}") from exc
    if not records:
        raise ValueError(f"External test manifest is empty: {path}")
    return tuple(records)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run the evaluate step for this workflow."""
    manifest = args.manifest.resolve()
    records = _read_manifest(manifest)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"

    exact_matches = 0
    misses = 0
    character_count = 0
    edit_distance = 0
    with predictions_path.open("w", encoding="utf-8") as stream:
        for record in records:
            image_path = (manifest.parent / record.image_path).resolve()
            if not image_path.is_relative_to(manifest.parent.resolve()):
                raise ValueError(f"Image path escapes manifest directory: {record.image_path}")
            result = predict(
                argparse.Namespace(
                    image=image_path,
                    detection_checkpoint=args.detection_checkpoint,
                    ocr_model_dir=args.ocr_model_dir,
                    device=args.device,
                    conf=args.conf,
                    compact_aspect_ratio=1.5,
                    min_detection_confidence=args.min_detection_confidence,
                    min_ocr_confidence=args.min_ocr_confidence,
                    output=None,
                )
            )
            ground_truth = normalize_plate_text(record.plate_text)
            plates = result["plates"]
            selected = max(plates, key=lambda plate: plate["detection_confidence"], default=None)
            prediction = "" if selected is None else selected["normalized_text"]
            distance = levenshtein_distance(ground_truth, prediction)
            exact_match = ground_truth == prediction
            exact_matches += exact_match
            misses += selected is None
            character_count += len(ground_truth)
            edit_distance += distance
            stream.write(
                json.dumps(
                    {
                        "image_path": record.image_path,
                        "ground_truth": ground_truth,
                        "prediction": prediction,
                        "exact_match": exact_match,
                        "edit_distance": distance,
                        "plate": selected,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics = {
        "samples": len(records),
        "exact_matches": exact_matches,
        "exact_match_rate": exact_matches / len(records),
        "no_detection_count": misses,
        "no_detection_rate": misses / len(records),
        "character_count": character_count,
        "edit_distance": edit_distance,
        "cer": edit_distance / character_count,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> int:
    """Run the main step for this workflow."""
    metrics = evaluate(_parse_args())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
