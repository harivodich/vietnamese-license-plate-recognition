"""Try deterministic crop-restoration variants before escalating to a generative model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from predict_end_to_end import _recognize  # noqa: E402
from vlpr.evaluation.ocr import normalize_plate_text  # noqa: E402


def main() -> int:
    """Run the main step for this workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--bbox", type=int, nargs=4, required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ocr_rescue"))
    parser.add_argument(
        "--ocr-model-dir",
        type=Path,
        default=Path("artifacts/ocr/paddleocr-v5-mobile-finetune/inference"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = cv2.imread(str(args.image))
    left, top, right, bottom = args.bbox
    pad_x, pad_y = round((right - left) * 0.35), round((bottom - top) * 0.35)
    crop = source[max(0, top - pad_y) : bottom + pad_y, max(0, left - pad_x) : right + pad_x]
    enlarged = cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_LANCZOS4)
    lab = cv2.cvtColor(enlarged, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4)).apply(lab[:, :, 0])
    clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    unsharp = cv2.addWeighted(clahe, 1.15, cv2.GaussianBlur(clahe, (0, 0), 1.0), -0.15, 0)
    variants = {"padded_lanczos": enlarged, "clahe": clahe, "unsharp": unsharp}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, image in variants.items():
        path = args.output_dir / f"{name}.png"
        cv2.imwrite(str(path), image)
        paths.append(path)
    outputs = _recognize(paths, root=root, model_dir=root / args.ocr_model_dir, device="cpu")
    expected = normalize_plate_text(args.expected)
    for name, (text, confidence) in zip(variants, outputs, strict=True):
        normalized = normalize_plate_text(text)
        print(name, normalized, round(confidence, 3), normalized == expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
