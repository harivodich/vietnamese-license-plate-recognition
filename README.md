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
python -m pip install -e ".[dev]"
pre-commit install
```

`pip install -e` performs an editable install: imports resolve to the source files in this
repository, so code changes take effect without reinstalling the package. The `[dev]` extra adds
quality tools such as Ruff, mypy, pytest, and pre-commit; it does not add training frameworks yet.
Model dependencies will be selected when the detection baseline is designed.

## Scope v1

Included: still images, multiple vehicles/plates, detection, OCR, post-processing, HTTP inference,
ONNX export, and CPU/GPU benchmarking.

Excluded: video tracking, camera streaming, vehicle recognition, microservices, and Kubernetes.

## Dataset workflow

```powershell
python scripts/download_data.py --config configs/dataset.yaml
python scripts/prepare_data.py --config configs/dataset.yaml
python scripts/validate_data.py --config configs/dataset.yaml
python scripts/split_data.py --config configs/dataset.yaml
```

The initial Kaggle candidate is
[`miahuynh04/vietnamese-license-plate-detection`](https://www.kaggle.com/datasets/miahuynh04/vietnamese-license-plate-detection).
It is not considered approved until the downloaded files, annotations, provenance, and license
have been audited.

## Quality commands

```powershell
ruff check .
ruff format --check .
mypy
pytest
```

Training, evaluation, API, Docker, measured results, and limitations will be documented as their
gates are completed.
