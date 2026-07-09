# Experiment report

## Detection baseline

The reproducible baseline contract is defined in `configs/detection-baseline.yaml`:

- pretrained model: `yolo11n.pt`;
- input size: 640;
- batch size: 8;
- optimizer: AdamW;
- initial learning rate: 0.001;
- maximum epochs: 50;
- seed: 42;
- deterministic mode: enabled;
- project split: 6,191 train, 1,032 validation, and 1,032 test images.

The baseline completed all 50 epochs. The best validation result was recorded at epoch 47. The
frozen `best.pt` checkpoint was then evaluated once on the fixed test split:

| Metric | Validation | Test |
| --- | ---: | ---: |
| Images | 1,032 | 1,032 |
| Instances | 1,071 | 1,049 |
| Precision | 0.9924 | 0.9943 |
| Recall | 0.9754 | 0.9931 |
| mAP@0.5 | 0.9930 | 0.9944 |
| mAP@0.5:0.95 | 0.7173 | 0.7246 |
| Inference latency | 7.3 ms/image | 11.8 ms/image |

Training took 4.859 hours on an NVIDIA Quadro M2200. The local `best.pt` checkpoint has SHA-256
`70289013711AB5DA541C2A3B6EB44052756C0DFD6FF3EF16039AD83A5856CA3C`. Model checkpoints and full
run artifacts are intentionally excluded from Git.

Test performance is slightly higher than validation performance, so there is no measured
generalization drop between the two project splits. This is not evidence of cross-source
generalization because both splits come from the same source dataset. An external test set remains
required.

### Fixed operating point

In addition to threshold-independent mAP, predictions were measured at confidence `0.25` and
matched one-to-one with ground truth at IoU `0.5`:

| Metric | Value |
| --- | ---: |
| Predictions | 1,055 |
| Matched ground truths | 1,044 / 1,049 |
| False negatives | 5 |
| False positives | 11 |
| Precision | 0.9896 |
| Recall | 0.9952 |
| F1 | 0.9924 |

The confidence threshold was fixed before this analysis. It was not selected on the test set.
Threshold tuning for deployment must use validation data.

### Recall by plate size

Size groups use the same COCO-inspired thresholds as the dataset audit after letterboxing to
`640 × 640`:

| Size | Instances | Matched | Recall |
| --- | ---: | ---: | ---: |
| Small, area below `32²` | 81 | 76 | 0.9383 |
| Medium, area from `32²` to below `96²` | 601 | 601 | 1.0000 |
| Large, area at least `96²` | 367 | 367 | 1.0000 |

All five false negatives occur in one `600 × 400` image containing seven very small plates. Two
plates are matched at confidence `0.25`; additional low-confidence candidates appear below the fixed threshold.
The main measured weakness is therefore dense scenes with very small plates. Medium and large
plates have no false negatives at this operating point.

False positives occur across ten images: one image contains two false positives and nine images
contain one each. All eleven rendered failure images are retained for qualitative review.

### Latency

Ultralytics reports the following GPU processing time per test image:

| Stage | ms/image |
| --- | ---: |
| Preprocess | 3.59 |
| Inference | 11.79 |
| Postprocess | 5.19 |
| Total measured pipeline | 20.57 |

This corresponds to approximately `48.6 images/second` for the measured preprocess, inference, and
postprocess stages on the NVIDIA Quadro M2200. It excludes image upload, disk I/O, OCR, API
serialization, and network latency, so it is not yet the end-to-end application benchmark.

### Interpretation and decision

- The detector reliably finds plates at IoU `0.5`; mAP@0.5 is `0.9944`.
- mAP@0.5:0.95 is lower at `0.7246`, so tight localization remains harder than basic detection.
- Increasing model size is not justified by the current same-source test results.
- The dataset has no structured night, tilt, or partial-occlusion tags, so reproducible metrics
  for those subgroups are not available yet.
- The next detector work should target small-plate cases or an external test set, not tune against
  this test split.
- Detection is sufficient to start an independent OCR baseline while preserving these results as
  the frozen detection baseline.

Tracked evidence:

- [epoch metrics](detection_baseline/results.csv);
- [training curves](detection_baseline/results.png);
- [test metric summary](detection_baseline/test/metrics.json);
- [fixed operating-point analysis](detection_baseline/test/operating_point.json);
- [test precision-recall curve](detection_baseline/test/BoxPR_curve.png);
- [test F1-confidence curve](detection_baseline/test/BoxF1_curve.png);
- [test confusion matrix](detection_baseline/test/confusion_matrix.png);
- [normalized test confusion matrix](detection_baseline/test/confusion_matrix_normalized.png);
- [rendered failure examples](detection_baseline/test/failure_examples/).
## OCR pretrained baseline

Recognition-only PaddleOCR was evaluated on 818 fixed ground-truth test crops. The detector was not
used in this experiment.

| Metric | Value |
| --- | ---: |
| Full-plate exact match | 0.1944 |
| Character error rate | 0.5938 |
| Character accuracy | 0.4062 |
| CPU inference latency | 62.86 ms/image |

Compact crops have zero exact matches and CER `0.7883`, compared with exact match `0.4251` and CER
`0.3477` for wide crops. Layout handling must therefore be tested on validation data before OCR
fine-tuning.

Detailed protocol, subgroup results, limitations, and sample-level evidence are in the
[OCR baseline report](ocr_baseline/report.md).
## OCR training preparation

The trainable OCR baseline is prepared as a lightweight CRNN + CTC experiment. This keeps training local and reproducible on the same PyTorch stack used by the detector.

Prepared line-level dataset:

| Split | Line samples |
| --- | ---: |
| Train | 7,595 |
| Validation | 1,249 |

Compact two-line plates are split into top and bottom line crops before training. Wide one-line plates remain single line crops. This is intentional because the pretrained OCR baseline showed compact crops are the main failure group; line-level training makes the first trainable baseline simpler and easier to debug.

Experiment contract:

- config: `configs/ocr-crnn.yaml`;
- preparation script: `scripts/prepare_ocr_training.py`;
- training script: `scripts/train_ocr.py`;
- training data output: `data/processed/ocr_training/`;
- artifact output: `artifacts/ocr/crnn-ctc-baseline/`;
- model: CRNN with CNN feature extractor, BiLSTM sequence encoder, and CTC loss;
- input: grayscale OCR line image resized/padded to `32 x 160`;
- epochs: `50` maximum;
- batch size: `64`;
- optimizer: AdamW;
- learning rate: `0.001`;
- early stopping: validation exact match patience `10`;
- checkpointing: `last.pt`, `best.pt`, and periodic epoch checkpoints.

Preflight and smoke-test completed successfully on the prepared dataset. Full training has not been run in this report yet; the user will run it locally and use the saved `history.csv`, `training_curves.png`, `best.pt`, and `last.pt` for the next evaluation step.

Tracked guidance:

- [OCR training guide](../docs/ocr-training-guide.md).

## OCR PaddleOCR fine-tuning

The project then switches the trainable OCR path from the local CRNN experiment to the pretrained
PaddleOCR recognition model. This is the stronger production-oriented direction because it starts
from an OCR model that already understands text-line features instead of learning OCR from scratch.

The fine-tuning data is exported in PaddleOCR recognition format:

| Split | Line samples |
| --- | ---: |
| Train | 7,595 |
| Validation | 1,249 |
| Test | 1,262 |

The evaluated checkpoint is `best_accuracy` from the PaddleOCR fine-tuning run. It was first evaluated on
the fixed held-out line-level test split using PaddleOCR's official recognition metric:

| Metric | Value |
| --- | ---: |
| PaddleOCR recognition accuracy | 0.7639 |
| Normalized edit distance | 0.8997 |
| GPU evaluation throughput | 167.65 FPS |

### Full-plate Evaluation

The fine-tuned recognizer was exported and integrated into the project evaluation path, measuring full-plate exact match, CER, and geometric subgroup metrics under the exact same strict protocol as the pretrained OCR baseline.

| Metric | Pretrained Baseline | Fine-tuned Model | Improvement |
| --- | ---: | ---: | ---: |
| Full-plate exact match | 19.44% | 70.42% | **+ 50.98%** |
| Character error rate (CER) | 59.38% | 9.25% | **- 50.13%** |
| Character accuracy | 40.62% | 90.75% | **+ 50.13%** |

**Metrics by Geometry:**
- **Compact plates (2 lines):** Exact match 66.89%, CER 10.49%
- **Wide plates (1 line):** Exact match 74.60%, CER 7.69%

The fine-tuning process was a massive success, boosting the strict full-plate exact match rate from an unusable 19% to over 70%, and pushing character accuracy past 90%.

### Error Analysis and Next Steps

Despite the massive improvement, the 70.4% exact match indicates room for optimization. The top 25 failure cases were extracted for manual visual review using the `scripts/analyze_ocr_errors.py` script.

Initial findings from the worst failure cases (e.g. `Ground truth: 68HC 00047` vs `Prediction: 29A 1923`) strongly indicate the presence of **Noisy Labels / Annotation Errors** in the ground-truth test set. When the model outputs a completely different but valid license plate format with a high edit distance, it is usually because the dataset annotation is mismatched with the image crop.

**Recommended Next Steps before further training:**
1. **Clean the Test Set:** Visually review the images in `artifacts/ocr/paddleocr-v5-mobile-finetune/eval/failures` and correct the ground-truth labels. Evaluating on noisy labels artificially caps the maximum possible exact match score.
2. **Address Compact Plates:** Wide plates perform significantly better than compact plates (74.6% vs 66.9% exact match). Data augmentation targeting compact plates (blur, perspective transform) or reviewing the crop splitting logic should be considered.
