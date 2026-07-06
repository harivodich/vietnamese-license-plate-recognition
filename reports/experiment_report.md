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
