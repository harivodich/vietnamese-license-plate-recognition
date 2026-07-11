# Vietnamese License Plate Recognition

End-to-end recognition for one or more Vietnamese license plates in a vehicle image:

```text
image -> plate detection -> crop/rectify -> OCR -> normalize/validate -> structured JSON
```

## Status

Project initialization and dataset audit are in progress. No model metrics are reported until
they are measured on the frozen test set.

## Environment setup

This repository uses the existing Conda environment `HariAI`:

```powershell
conda activate HariAI
python --version
python -m pip install -e ".[data,tracking,dev]"
pre-commit install
```

`pip install -e` performs an editable install: imports resolve to the source files in this
repository, so code changes take effect without reinstalling the package. Extras add only the
dependencies needed by the current project gate.

Dependency groups are intentionally separated:

- `data`: Kaggle download and image loading;
- `detection`: PyTorch and Ultralytics;
- `ocr`: PaddlePaddle and PaddleOCR;
- `tracking`: Weights & Biases;
- `api`: FastAPI and the localhost web UI;
- `export`: ONNX export and inference;
- `dev`: lint, type checking, tests, and pre-commit.

Install only the groups needed for the current gate. This avoids installing both ML frameworks
before they are needed.

## Environment check

```powershell
python scripts/check_environment.py --output reports/environment_report.md
```

This command distinguishes a physical NVIDIA GPU from a CUDA-enabled PyTorch runtime. A machine
can have an NVIDIA GPU while the installed PyTorch build still runs on CPU.

## Scope v1

Included: still images, multiple vehicles/plates, detection, OCR, post-processing, HTTP inference,
ONNX export, and CPU/GPU benchmarking.

Excluded: video tracking, camera streaming, vehicle recognition, microservices, and Kubernetes.

## Dataset workflow

```powershell
python scripts/download_data.py --config configs/dataset.yaml --dataset detection
python scripts/download_data.py --config configs/dataset.yaml --dataset ocr
python scripts/prepare_data.py --config configs/dataset.yaml
python scripts/validate_data.py --config configs/dataset.yaml
python scripts/create_review_samples.py --config configs/dataset.yaml
python scripts/split_data.py --config configs/dataset.yaml
python scripts/analyze_detection_data.py --config configs/dataset.yaml
python scripts/analyze_ocr_data.py --config configs/dataset.yaml
python scripts/export_detection_yolo.py --config configs/dataset.yaml
```

The baseline data strategy uses two Kaggle datasets:

- detection: [`miahuynh04/vietnamese-license-plate-detection`](https://www.kaggle.com/datasets/miahuynh04/vietnamese-license-plate-detection)
- OCR: [`wirqhuy/vietnamese-license-plate-ocr`](https://www.kaggle.com/datasets/wirqhuy/vietnamese-license-plate-ocr)

This is deliberate. The detection source provides full-scene vehicle images with plate bounding
boxes, while the OCR source provides cropped plate images with text labels. We will evaluate them
independently before measuring the combined pipeline.

Raw sources are versioned under `data/raw/kaggle/<task>/v<version>`. Every completed source has a
`download_receipt.json` containing its identity, file count, byte count, and
deterministic content fingerprint. Re-running a completed download is a no-op unless `--force` is
explicitly supplied.

## Detection baseline

Install the detection dependencies, validate the frozen experiment contract, then start training:

```powershell
python -m pip install -e ".[data,detection,tracking,dev]"
python scripts/train_detection.py --config configs/detection-baseline.yaml --check-only
python scripts/train_detection.py --config configs/detection-baseline.yaml
python scripts/evaluate_detection.py --config configs/detection-evaluation.yaml --check-only
python scripts/evaluate_detection.py --config configs/detection-evaluation.yaml
python scripts/analyze_detection_errors.py --config configs/detection-evaluation.yaml
```

`--check-only` validates the strict training config, dataset YAML, and all three non-empty image
lists without loading model weights or starting a training run. The baseline config records image
size, batch size, optimizer, learning rate, epochs, augmentation, seed, and deterministic mode.
The evaluation config locks the checkpoint and `test` split, while the error-analysis command
reports recall by plate size at a fixed confidence and IoU operating point.

Detailed explanation of the training code, metrics, checkpoints, resume flow, and model tradeoffs:
[Detection training guide](docs/detection-training-guide.md).

## OCR baseline

The primary OCR path uses a pretrained recognizer first, the same principle as starting detection
from a pretrained YOLO checkpoint instead of training a detector from scratch. The baseline runs
recognition only on fixed ground-truth plate crops, so OCR errors are measured independently from
detection and crop errors.

```powershell
python scripts/evaluate_ocr.py --config configs/ocr-baseline.yaml --check-only
python scripts/evaluate_ocr.py --config configs/ocr-baseline.yaml
python scripts/evaluate_ocr.py --config configs/ocr-baseline-wide.yaml
python scripts/evaluate_ocr.py --config configs/ocr-baseline-layout.yaml
```

`configs/ocr-baseline.yaml` scores the full OCR test split using original crops.
`configs/ocr-baseline-wide.yaml` scores only one-line wide crops.
`configs/ocr-baseline-layout.yaml` keeps wide crops unchanged, splits compact crops into ordered
line crops, then merges line predictions back into full-plate metrics.

Results and error analysis are documented in the
[OCR baseline report](reports/ocr_baseline/report.md).

## OCR training

The CRNN+CTC trainer is kept as a controlled scratch baseline and teaching experiment, not as the
main accuracy route. It is still useful for validating preprocessing, CTC labels, checkpoints,
resume behavior, and metric code before fine-tuning a stronger pretrained OCR model.

```powershell
python scripts/prepare_ocr_training.py --config configs/ocr-crnn.yaml
python scripts/train_ocr.py --config configs/ocr-crnn.yaml --check-only
python scripts/train_ocr.py --config configs/ocr-crnn.yaml --smoke-test
python scripts/train_ocr.py --config configs/ocr-crnn.yaml --tiny-overfit
python scripts/train_ocr.py --config configs/ocr-crnn.yaml
```

If training is interrupted, resume from the latest checkpoint:

```powershell
python scripts/train_ocr.py --config configs/ocr-crnn.yaml --resume artifacts/ocr/crnn-ctc-wide-baseline/last.pt
```

The trainer selects `best.pt` by validation CER first and uses a conservative cosine learning-rate
schedule. The next production-oriented OCR step is pretrained OCR fine-tuning with the project
charset, using the same compact/wide subgroup reporting.

Prepare PaddleOCR fine-tune line data:

```powershell
python scripts/export_ocr_finetune_data.py --config configs/ocr-finetune-paddleocr.yaml
python scripts/prepare_paddleocr_finetune.py --config configs/ocr-paddleocr-finetune.yaml
python scripts/train_paddleocr_finetune.py --config configs/ocr-paddleocr-finetune.yaml
```

PaddleOCR saves `latest` every epoch and `iter_epoch_5/10/...` according to the configured
checkpoint interval. Resume interrupted fine-tuning with:

```powershell
python scripts/train_paddleocr_finetune.py --config configs/ocr-paddleocr-finetune.yaml --resume artifacts/ocr/paddleocr-v5-mobile-finetune/checkpoints/latest
```

PaddleOCR fine-tuning setup and command shape:
[OCR fine-tuning guide](docs/ocr-finetuning-guide.md).

Detailed explanation of CRNN, CTC loss, metrics, checkpoints, resume flow, and tunable settings:
[OCR training guide](docs/ocr-training-guide.md).

## Quality commands

```powershell
ruff check .
ruff format --check .
mypy
pytest
```

Training, evaluation, API, Docker, measured results, and limitations will be documented as their
gates are completed.

## End-to-end inference

Run detection, OCR, normalization, and status classification on one vehicle image:

```bash
python scripts/predict_end_to_end.py path/to/vehicle.jpg
```

Create an annotated review image and a separate JSON result:

```bash
python scripts/predict_visualized.py path/to/vehicle.jpg \
  --output artifacts/prediction_review.jpg \
  --json-output artifacts/prediction.json
```

Run the API locally:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` and use `POST /predict`.

## Video tracking

Run plate detection, ByteTrack IDs, and best-frame OCR updates:

```bash
python scripts/track_video.py path/to/video.mp4
```

The command writes `artifacts/tracked_plates.mp4` and a frame-level JSONL audit file.

## Docker

The image keeps model and dataset files outside Git. Checkpoints and processed OCR assets are mounted
read-only through Compose:

```bash
docker compose build
docker compose up
```

## Deployment benchmark

The selected detector was exported to ONNX and compared with PyTorch on 100 fixed test images. ONNX
Runtime achieved 79.42 ms mean latency, 78.40 ms p50, 86.53 ms p95, and 12.59 images/s on CPU. All
100 ONNX detections matched PyTorch at IoU >= 0.9, with mean IoU 0.977 and maximum confidence delta
0.0116. ONNX is therefore the selected CPU detection runtime.

## Limitations

Detection and OCR were trained and evaluated on separate datasets. A numeric end-to-end exact-match
metric is not reported because the project does not yet have an external set containing vehicle
images and verified plate text for the same records. Very small or blurred plate crops can remain
unreadable; preprocessing and super-resolution must not be treated as evidence when source pixels
are missing. Video OCR improves opportunities by revisiting sharper frames under the same track ID.
