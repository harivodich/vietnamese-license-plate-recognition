"""Detect and track license plates in a video with ByteTrack and cached OCR."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
from PIL import Image
from ultralytics import YOLO

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from predict_end_to_end import _recognize, _write_recognition_inputs  # noqa: E402
from vlpr.postprocessing.plate import assess_plate_text  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Run the parse args step for this workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/tracked_plates.mp4"))
    parser.add_argument("--jsonl-output", type=Path, default=Path("artifacts/tracked_plates.jsonl"))
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
    parser.add_argument("--compact-aspect-ratio", type=float, default=1.5)
    return parser.parse_args()


def _draw(frame: Any, box: list[int], track_id: int, text: str, confidence: float) -> None:
    """Run the draw step for this workflow."""
    left, top, right, bottom = box
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 168, 0), 2)
    label = f"ID {track_id} | {text or 'OCR pending'} | {confidence:.2f}"
    label_top = max(22, top - 7)
    cv2.putText(
        frame,
        label,
        (left, label_top),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    """Run the main step for this workflow."""
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    video_path = args.video.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    checkpoint = (root / args.detection_checkpoint).resolve()
    ocr_model_dir = (root / args.ocr_model_dir).resolve()
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise ValueError(f"Cannot decode video: {video_path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {args.output}")

    detector = YOLO(str(checkpoint))
    track_ocr: dict[int, tuple[str, float, float]] = {}
    frame_index = 0
    with (
        tempfile.TemporaryDirectory(prefix="vlpr-video-") as temporary,
        args.jsonl_output.open("w", encoding="utf-8") as stream,
    ):
        temporary_dir = Path(temporary)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            tracked = detector.track(
                frame, persist=True, tracker="bytetrack.yaml", conf=args.conf, verbose=False
            )[0]
            plates: list[dict[str, Any]] = []
            if tracked.boxes.id is not None:
                boxes = tracked.boxes.xyxy.int().tolist()
                track_ids = tracked.boxes.id.int().tolist()
                confidences = tracked.boxes.conf.tolist()
                for index, (box, track_id, confidence) in enumerate(
                    zip(boxes, track_ids, confidences, strict=True)
                ):
                    left, top, right, bottom = box
                    left, top = max(0, left), max(0, top)
                    right, bottom = min(width, right), min(height, bottom)
                    if track_id not in track_ocr and right > left and bottom > top:
                        crop_pixels = frame[top:bottom, left:right]
                        crop = Image.fromarray(cv2.cvtColor(crop_pixels, cv2.COLOR_BGR2RGB))
                        paths = _write_recognition_inputs(
                            crop,
                            directory=temporary_dir,
                            index=frame_index * 1000 + index,
                            compact_aspect_ratio=args.compact_aspect_ratio,
                        )
                        outputs = _recognize(
                            paths, root=root, model_dir=ocr_model_dir, device=args.device
                        )
                        raw_text = " ".join(text for text, _score in outputs if text)
                        ocr_confidence = min((score for _text, score in outputs), default=0.0)
                        assessment = assess_plate_text(
                            raw_text,
                            detection_confidence=float(confidence),
                            ocr_confidence=ocr_confidence,
                        )
                        track_ocr[track_id] = (assessment.normalized_text, ocr_confidence)
                    text, ocr_confidence = track_ocr.get(track_id, ("", 0.0))
                    _draw(frame, [left, top, right, bottom], track_id, text, float(confidence))
                    plates.append(
                        {
                            "track_id": track_id,
                            "bbox": [left, top, right, bottom],
                            "detection_confidence": float(confidence),
                            "normalized_text": text,
                            "ocr_confidence": ocr_confidence,
                        }
                    )
            writer.write(frame)
            stream.write(json.dumps({"frame_index": frame_index, "plates": plates}) + "\n")
            frame_index += 1
    capture.release()
    writer.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
