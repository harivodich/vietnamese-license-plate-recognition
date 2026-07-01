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
python scripts/split_data.py --config configs/dataset.yaml
```

The baseline data strategy uses two Kaggle datasets:

- detection: [`miahuynh04/vietnamese-license-plate-detection`](https://www.kaggle.com/datasets/miahuynh04/vietnamese-license-plate-detection)
- OCR: [`wirqhuy/vietnamese-license-plate-ocr`](https://www.kaggle.com/datasets/wirqhuy/vietnamese-license-plate-ocr)

This is deliberate. The detection source provides full-scene vehicle images with plate bounding
boxes, while the OCR source provides cropped plate images with text labels. We will evaluate them
independently before measuring the combined pipeline.

## Quality commands

```powershell
ruff check .
ruff format --check .
mypy
pytest
```

Training, evaluation, API, Docker, measured results, and limitations will be documented as their
gates are completed.
