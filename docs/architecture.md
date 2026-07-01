# Architecture

## Training path

```text
Kaggle sources
  -> immutable raw data for detection
  -> immutable raw data for OCR
  -> normalized JSONL manifest
  -> integrity/label/duplicate audit
  -> group-aware frozen splits
  -> detector and OCR training
  -> independent component evaluation
  -> end-to-end evaluation
```

## Inference path

```text
input validation
  -> detector
  -> per-plate crop and optional rectification
  -> OCR
  -> Vietnamese plate normalization and validation
  -> typed response
```

Detection and OCR remain separate components. This permits two controlled OCR evaluations:
ground-truth crop to OCR and predicted crop to OCR. Their difference measures error propagation
from detection instead of incorrectly attributing every end-to-end failure to OCR.
