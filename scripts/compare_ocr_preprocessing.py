"""Compare safe OCR preprocessing variants for one detected plate crop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from predict_end_to_end import _recognize, _write_recognition_inputs  # noqa: E402
from vlpr.evaluation.ocr import normalize_plate_text  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Run the parse args step for this workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--bbox",
        type=int,
        nargs=4,
        required=True,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
    )
    parser.add_argument("--expected", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ocr_preprocessing"))
    parser.add_argument(
        "--ocr-model-dir",
        type=Path,
        default=Path("artifacts/ocr/paddleocr-v5-mobile-finetune/inference"),
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    return parser.parse_args()


def _padded_clahe(crop: Image.Image) -> Image.Image:
    """Add context, upscale once, then apply contrast normalization without generative changes."""
    pixels = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR)
    border_x = max(2, round(pixels.shape[1] * 0.15))
    border_y = max(2, round(pixels.shape[0] * 0.15))
    padded = cv2.copyMakeBorder(
        pixels, border_y, border_y, border_x, border_x, cv2.BORDER_REPLICATE
    )
    enlarged = cv2.resize(padded, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
    lab = cv2.cvtColor(enlarged, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def main() -> int:
    """Run the main step for this workflow."""
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    source = Image.open(args.image).convert("RGB")
    left, top, right, bottom = args.bbox
    crop = source.crop((left, top, right, bottom))
    variants = {"original": crop, "padded_lanczos_clahe": _padded_clahe(crop)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_paths: list[Path] = []
    groups: list[tuple[str, int]] = []
    for name, image in variants.items():
        variant_dir = args.output_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        paths = _write_recognition_inputs(
            image, directory=variant_dir, index=0, compact_aspect_ratio=1.5
        )
        all_paths.extend(paths)
        groups.append((name, len(paths)))
    outputs = _recognize(
        all_paths, root=root, model_dir=(root / args.ocr_model_dir), device=args.device
    )
    offset = 0
    expected = normalize_plate_text(args.expected) if args.expected else None
    results = []
    for name, size in groups:
        group = outputs[offset : offset + size]
        offset += size
        text = " ".join(value for value, _score in group if value)
        normalized = normalize_plate_text(text)
        results.append(
            {
                "variant": name,
                "prediction": text,
                "normalized_prediction": normalized,
                "ocr_confidence": min((score for _text, score in group), default=0.0),
                "exact_match": normalized == expected if expected is not None else None,
            }
        )
    rendered = {"bbox": args.bbox, "expected": expected, "results": results}
    (args.output_dir / "comparison.json").write_text(
        json.dumps(rendered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(rendered, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
