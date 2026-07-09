"""Đánh giá OCR độc lập cho mô hình PaddleOCR fine-tuned sử dụng native inference."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from statistics import fmean

import cv2

from vlpr.config import project_root, resolve_project_path
from vlpr.evaluation.ocr import (
    OcrEvaluationResult,
    _build_recognition_inputs,
    _group_by_geometry,
    _match_results_to_recognition_inputs,
    _metric_group,
    _write_predictions,
    validate_ocr_evaluation,
)
from vlpr.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


class DummyArgs:
    """Args mô phỏng cho tools.infer.predict_rec.TextRecognizer"""

    def __getattr__(self, item):
        return False


def evaluate_finetune_ocr(config_path: Path) -> OcrEvaluationResult:
    inputs = validate_ocr_evaluation(config_path)
    config = inputs.config
    root = project_root(config_path)
    output_dir = resolve_project_path(root, config.project) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    recognition_inputs = _build_recognition_inputs(inputs, output_dir)

    # Đưa PaddleOCR vào sys.path để import code native
    paddleocr_dir = str(root / "external" / "PaddleOCR")
    if paddleocr_dir not in sys.path:
        sys.path.insert(0, paddleocr_dir)

    from tools.infer.predict_rec import TextRecognizer

    args = DummyArgs()
    args.rec_model_dir = str(resolve_project_path(root, config.model_dir))
    args.rec_image_shape = "3,48,320"
    args.rec_batch_num = config.batch_size
    args.rec_algorithm = "SVTR_LCNet"

    dict_path = str(resolve_project_path(root, "data/processed/ocr_finetune_paddleocr/dict.txt"))
    args.rec_char_dict_path = dict_path
    args.use_space_char = False
    args.use_onnx = False
    args.benchmark = False
    args.use_gpu = config.device == "gpu"
    args.precision = "fp32"
    args.return_word_box = False
    args.max_text_length = 25
    args.use_tensorrt = False
    args.rec_image_inverse = False

    initialization_started = time.perf_counter()
    model = TextRecognizer(args)
    initialization_seconds = time.perf_counter() - initialization_started

    inference_started = time.perf_counter()

    # Chuẩn bị danh sách ảnh numpy cho batch predict
    image_list = []
    path_list = [str(r.path) for r in recognition_inputs]
    for p in path_list:
        img = cv2.imread(p)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh: {p}")
        image_list.append(img)

    rec_res, _ = model(image_list)
    inference_seconds = time.perf_counter() - inference_started

    # Chuyển đổi kết quả về dạng tương đương raw_results
    raw_results = []
    for path, res in zip(path_list, rec_res, strict=False):
        text, score = res[0], res[1]
        raw_results.append({"input_path": path, "rec_text": text, "rec_score": score})

    predictions = _match_results_to_recognition_inputs(
        inputs,
        recognition_inputs,
        raw_results,
    )

    ranked_failures = tuple(
        sorted(
            (prediction for prediction in predictions if not prediction.exact_match),
            key=lambda prediction: (
                -prediction.edit_distance,
                prediction.confidence,
                prediction.sample_id,
            ),
        )[: config.failure_examples]
    )

    result = OcrEvaluationResult(
        model_name="finetuned",
        split="test",
        geometry=config.geometry,
        layout=config.layout,
        metrics=_metric_group(predictions),
        metrics_by_geometry=_group_by_geometry(predictions, config.compact_aspect_ratio),
        mean_confidence=fmean(prediction.confidence for prediction in predictions),
        model_initialization_seconds=initialization_seconds,
        inference_ms_per_image=inference_seconds * 1000 / len(predictions),
        failure_examples=ranked_failures,
    )

    _write_predictions(output_dir / "predictions.jsonl", predictions)
    (output_dir / "metrics.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("OCR finetune metrics saved to %s", output_dir / "metrics.json")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ocr-finetune-eval.yaml"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        result = evaluate_finetune_ocr(args.config)
        LOGGER.info(
            "OCR finetune eval completed exact=%.4f CER=%.4f char_acc=%.4f",
            result.metrics.exact_match_rate,
            result.metrics.cer,
            result.metrics.character_accuracy,
        )
    except (AttributeError, ImportError, OSError, ValueError) as exc:
        LOGGER.error("OCR evaluation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
