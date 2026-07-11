"""Run end-to-end prediction and save an annotated image with a JSON side panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from predict_end_to_end import predict  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Run the parse args step for this workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input vehicle image")
    parser.add_argument("--output", type=Path, default=Path("artifacts/prediction_review.jpg"))
    parser.add_argument("--json-output", type=Path, default=Path("artifacts/prediction.json"))
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
    parser.add_argument("--display-min-confidence", type=float, default=0.5)
    parser.add_argument("--compact-aspect-ratio", type=float, default=1.5)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-ocr-confidence", type=float, default=0.8)
    return parser.parse_args()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Run the font step for this workflow."""
    for candidate in (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/consola.ttf")):
        if candidate.is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _wrap_json(payload: dict[str, Any], *, width: int, font: ImageFont.ImageFont) -> list[str]:
    """Run the wrap json step for this workflow."""
    lines: list[str] = []
    for source_line in json.dumps(payload, ensure_ascii=False, indent=2).splitlines():
        current = source_line
        while current:
            length = len(current)
            while length > 1 and font.getlength(current[:length]) > width:
                length -= 1
            lines.append(current[:length])
            current = current[length:]
    return lines


def render_prediction(image_path: Path, payload: dict[str, Any], output_path: Path) -> None:
    """Render annotations plus compact review cards; raw JSON is stored separately."""
    source = Image.open(image_path).convert("RGB")
    title_font = _font(22)
    detail_font = _font(16)
    panel_width = 450
    padding = 18
    card_height = 132
    panel_height = max(
        source.height,
        padding * 2 + 42 + card_height * max(1, len(payload["plates"])),
    )
    canvas = Image.new("RGB", (source.width + panel_width, panel_height), "#171717")
    canvas.paste(source, (0, 0))
    draw = ImageDraw.Draw(canvas)

    for index, plate in enumerate(payload["plates"], start=1):
        left, top, right, bottom = plate["bbox"]
        candidate = plate["detection_confidence"] < 0.5
        color = "#e53935" if candidate else "#00a8ff"
        kind = "false-positive candidate" if candidate else "license_plate"
        draw.rectangle((left, top, right, bottom), outline=color, width=4)
        label = "{} {} | det {:.2f}".format(kind, index, plate["detection_confidence"])
        label_top = max(0, top - 28)
        draw.text(
            (left, label_top),
            label,
            font=detail_font,
            fill="white",
            stroke_width=2,
            stroke_fill="#005b96",
        )

    panel_left = source.width
    draw.rectangle((panel_left, 0, canvas.width, canvas.height), fill="#f7f7f7")
    draw.text((panel_left + padding, padding), "Detected plates", font=title_font, fill="#111111")
    for index, plate in enumerate(payload["plates"]):
        y = padding + 48 + index * card_height
        draw.rounded_rectangle(
            (panel_left + padding, y, canvas.width - padding, y + card_height - 10),
            radius=8,
            fill="white",
            outline="#d0d0d0",
        )
        left, top, right, bottom = plate["bbox"]
        crop = source.crop((left, top, right, bottom)).resize((110, 70))
        canvas.paste(crop, (panel_left + padding + 10, y + 12))
        x = panel_left + padding + 135
        candidate = plate["detection_confidence"] < 0.5
        card_status = "false-positive candidate" if candidate else plate["status"]
        lines = [
            f"Plate {index + 1} | {card_status}",
            "OCR: {}".format(plate["normalized_text"] or "-"),
            "det {:.2f} | ocr {:.2f}".format(
                plate["detection_confidence"], plate["ocr_confidence"]
            ),
            "bbox: {}".format(plate["bbox"]),
        ]
        for line_index, line in enumerate(lines):
            draw.text((x, y + 12 + line_index * 25), line, font=detail_font, fill="#222222")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def main() -> int:
    """Run the main step for this workflow."""
    args = _parse_args()
    payload = predict(args)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    rendered_json = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    args.json_output.write_text(rendered_json, encoding="utf-8")
    render_prediction(args.image.resolve(), payload, args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
