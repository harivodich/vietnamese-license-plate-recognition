"""HTTP API for Vietnamese license plate recognition."""

from __future__ import annotations

import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from predict_end_to_end import predict  # noqa: E402

MODEL_VERSION = "1.0.0"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
DETECTION_CHECKPOINT = Path("artifacts/detection/yolo11n-baseline/weights/best.onnx")
OCR_MODEL_DIR = Path("artifacts/ocr/paddleocr-v5-mobile-finetune/inference")


class PlateResponse(BaseModel):
    """One detected plate returned by the end-to-end model."""

    model_config = ConfigDict(extra="forbid")

    bbox: list[int] = Field(min_length=4, max_length=4)
    detection_confidence: float
    raw_text: str
    normalized_text: str
    ocr_confidence: float
    format_valid: bool
    status: Literal["valid", "low_confidence", "invalid_format", "empty"]


class PredictResponse(BaseModel):
    """Stable API response for one vehicle image."""

    model_config = ConfigDict(extra="forbid")

    model_version: str
    processing_time_ms: float = Field(ge=0.0)
    plates: list[PlateResponse]


app = FastAPI(title="VLPR API", version=MODEL_VERSION)


@app.get("/health")
def health() -> dict[str, str]:
    """Return service liveness without loading either ML model."""
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, str]:
    """Expose the selected model paths for deployment auditing."""
    return {
        "model_version": MODEL_VERSION,
        "detection_checkpoint": str(DETECTION_CHECKPOINT),
        "ocr_model_dir": str(OCR_MODEL_DIR),
    }


@app.post("/predict", response_model=PredictResponse)
def predict_image(image: UploadFile = File(...)) -> PredictResponse:  # noqa: B008
    """Detect and recognize all plates in one uploaded image."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="image must have an image/* content type")
    payload = image.file.read(MAX_IMAGE_BYTES + 1)
    if len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image exceeds 10 MiB")
    if not payload:
        raise HTTPException(status_code=400, detail="image is empty")

    started = time.perf_counter()

    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        temporary.write(payload)
        temporary.flush()
        try:
            result = predict(
                Namespace(
                    image=Path(temporary.name),
                    detection_checkpoint=DETECTION_CHECKPOINT,
                    ocr_model_dir=OCR_MODEL_DIR,
                    device="cpu",
                    conf=0.25,
                    compact_aspect_ratio=1.5,
                    min_detection_confidence=0.5,
                    min_ocr_confidence=0.8,
                )
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="unable to process image") from exc
    Path(temporary.name).unlink(missing_ok=True)
    return PredictResponse(
        model_version=MODEL_VERSION,
        processing_time_ms=(time.perf_counter() - started) * 1000,
        plates=result["plates"],
    )
