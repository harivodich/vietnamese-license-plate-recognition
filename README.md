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

Run recognition-only PaddleOCR on the fixed ground-truth test crops:

```powershell
python scripts/evaluate_ocr.py --config configs/ocr-baseline.yaml --check-only
python scripts/evaluate_ocr.py --config configs/ocr-baseline.yaml
```

The baseline deliberately excludes the detector so OCR errors can be measured independently.
Results and error analysis are documented in the
[OCR baseline report](reports/ocr_baseline/report.md).
## Quality commands

```powershell
ruff check .
ruff format --check .
mypy
pytest
```

Training, evaluation, API, Docker, measured results, and limitations will be documented as their
gates are completed.
