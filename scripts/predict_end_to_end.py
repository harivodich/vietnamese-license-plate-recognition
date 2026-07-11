"""Run YOLO plate detection followed by the fine-tuned PaddleOCR recognizer."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from vlpr.config import resolve_project_path
from vlpr.data.ocr_layout import enlarge_small_crop, split_compact_crop
from vlpr.postprocessing.plate import assess_plate_text

_DETECTION_MODELS: dict[str, Any] = {}
_OCR_MODELS: dict[tuple[str, str], Any] = {}


class _PaddleArgs:
    """Native PaddleOCR inference options, defaulting unused hardware flags to False."""

    def __init__(self, *, model_dir: Path, dictionary: Path, device: str) -> None:
        """Run the init step for this workflow."""
        self.rec_model_dir = str(model_dir)
        self.rec_image_shape = "3,48,320"
        self.rec_batch_num = 32
        self.rec_algorithm = "SVTR_LCNet"
        self.rec_char_dict_path = str(dictionary)
        self.use_space_char = False
        self.use_onnx = False
        self.benchmark = False
        self.use_gpu = device == "gpu"
        self.precision = "fp32"
        self.return_word_box = False
        self.max_text_length = 25
        self.use_tensorrt = False
        self.rec_image_inverse = False

    def __getattr__(self, _name: str) -> bool:
        """Run the getattr step for this workflow."""
        return False


def _parse_args() -> argparse.Namespace:
    """Run the parse args step for this workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input vehicle image")
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
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--compact-aspect-ratio", type=float, default=1.5)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-ocr-confidence", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _recognize(
    image_paths: list[Path],
    *,
    root: Path,
    model_dir: Path,
    device: str,
) -> list[tuple[str, float]]:
    """Run the recognize step for this workflow."""
    if not image_paths:
        return []
    paddleocr_dir = str(root / "external" / "PaddleOCR")
    if paddleocr_dir not in sys.path:
        sys.path.insert(0, paddleocr_dir)
    from tools.infer.predict_rec import TextRecognizer

    key = (str(model_dir), device)
    recognizer = _OCR_MODELS.get(key)
    if recognizer is None:
        recognizer = TextRecognizer(
            _PaddleArgs(
                model_dir=model_dir,
                dictionary=root / "data/processed/ocr_finetune_paddleocr/dict.txt",
                device=device,
            )
        )
        _OCR_MODELS[key] = recognizer
    images: list[Any] = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read OCR crop: {path}")
        images.append(image)
    results, _ = recognizer(images)
    return [(str(result[0]), float(result[1])) for result in results]


def _write_recognition_inputs(
    crop: Image.Image,
    *,
    directory: Path,
    index: int,
    compact_aspect_ratio: float,
) -> list[Path]:
    """Split two-line plates using the same rule as the OCR evaluator."""
    crop = enlarge_small_crop(crop)
    if crop.width / crop.height < compact_aspect_ratio and crop.height >= 4:
        try:
            parts = split_compact_crop(crop, search_start=0.35, search_end=0.65)
        except ValueError:
            parts = (crop,)
    else:
        parts = (crop,)

    paths: list[Path] = []
    for part_index, part in enumerate(parts):
        path = directory / f"crop_{index}_{part_index}.png"
        part.save(path)
        paths.append(path)
    return paths


def predict(args: argparse.Namespace) -> dict[str, Any]:
    """Run the predict step for this workflow."""
    image_path = args.image.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    root = Path(__file__).resolve().parents[1]
    checkpoint = resolve_project_path(root, args.detection_checkpoint)
    ocr_model_dir = resolve_project_path(root, args.ocr_model_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Detection checkpoint not found: {checkpoint}")
    if not (ocr_model_dir / "inference.pdiparams").is_file():
        raise FileNotFoundError(f"OCR inference files missing: {ocr_model_dir}")

    from ultralytics import YOLO

    detection = _DETECTION_MODELS.get(str(checkpoint))
    if detection is None:
        detection = YOLO(str(checkpoint))
        _DETECTION_MODELS[str(checkpoint)] = detection
    result = detection.predict(
        source=str(image_path), conf=args.conf, device=args.device, verbose=False
    )[0]
    source = Image.open(image_path).convert("RGB")
    plate_metadata: list[dict[str, Any]] = []
    grouped_paths: list[list[Path]] = []

    with tempfile.TemporaryDirectory(prefix="vlpr-e2e-") as temporary:
        temporary_dir = Path(temporary)
        for index, (box, confidence) in enumerate(
            zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), strict=True)
        ):
            left, top, right, bottom = (round(value) for value in box)
            left, top = max(0, left), max(0, top)
            right, bottom = min(source.width, right), min(source.height, bottom)
            if right <= left or bottom <= top:
                continue
            crop = source.crop((left, top, right, bottom))
            grouped_paths.append(
                _write_recognition_inputs(
                    crop,
                    directory=temporary_dir,
                    index=index,
                    compact_aspect_ratio=args.compact_aspect_ratio,
                )
            )
            plate_metadata.append(
                {
                    "bbox": [left, top, right, bottom],
                    "detection_confidence": float(confidence),
                }
            )

        paths = [path for group in grouped_paths for path in group]
        predictions = _recognize(
            paths,
            root=root,
            model_dir=ocr_model_dir,
            device=args.device,
        )

    plates: list[dict[str, Any]] = []
    offset = 0
    for metadata, paths in zip(plate_metadata, grouped_paths, strict=True):
        group = predictions[offset : offset + len(paths)]
        offset += len(paths)
        raw_text = " ".join(text for text, _score in group if text)
        ocr_confidence = min((score for _text, score in group), default=0.0)
        assessment = assess_plate_text(
            raw_text,
            detection_confidence=metadata["detection_confidence"],
            ocr_confidence=ocr_confidence,
            min_detection_confidence=args.min_detection_confidence,
            min_ocr_confidence=args.min_ocr_confidence,
        )
        metadata.update(
            {
                "raw_text": assessment.raw_text,
                "normalized_text": assessment.normalized_text,
                "ocr_confidence": ocr_confidence,
                "format_valid": assessment.format_valid,
                "status": assessment.status,
            }
        )
        plates.append(metadata)
    return {"image": str(image_path), "plates": plates}


def main() -> int:
    """Run the main step for this workflow."""
    args = _parse_args()
    payload = predict(args)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
