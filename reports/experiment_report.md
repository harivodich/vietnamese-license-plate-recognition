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
`640 Ã— 640`:

| Size | Instances | Matched | Recall |
| --- | ---: | ---: | ---: |
| Small, area below `32Â²` | 81 | 76 | 0.9383 |
| Medium, area from `32Â²` to below `96Â²` | 601 | 601 | 1.0000 |
| Large, area at least `96Â²` | 367 | 367 | 1.0000 |

All five false negatives occur in one `600 Ã— 400` image containing seven very small plates. Two
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

### Full-plate evaluation

The fine-tuned recognizer was exported and integrated into the project evaluation path. The table
separates two effects: layout handling improves the pretrained model, and fine-tuning improves the
same split-compact protocol further.

| Metric | Pretrained original | Pretrained split-compact | Fine-tuned split-compact |
| --- | ---: | ---: | ---: |
| Full-plate exact match | 17.85% | 36.55% | 81.05% |
| Character error rate (CER) | 60.45% | 33.86% | 5.37% |
| Character accuracy | 39.55% | 66.14% | 94.63% |

Fine-tuning improves exact match by `+44.50` percentage points over the layout-aware pretrained
baseline, and by `+63.20` percentage points over the direct pretrained baseline on the cleaned
manifest.

Fine-tuned subgroup metrics:

| Geometry | Samples | Exact match | CER | Character accuracy |
| --- | ---: | ---: | ---: | ---: |
| Compact plates | 436 | 80.50% | 5.43% | 94.57% |
| Wide plates | 382 | 81.68% | 5.30% | 94.70% |

### Error analysis and next steps

The model is now clearly learning the task, but `81.05%` exact match still leaves enough failures to
inspect before further training. The top failure examples are saved in
`artifacts/ocr/paddleocr-v5-mobile-finetune/eval/metrics.json` and the full prediction table is saved
as `artifacts/ocr/paddleocr-v5-mobile-finetune/eval/predictions.jsonl`.

The largest errors are concentrated in very small or hard crops, especially compact plates. Some
examples may also be annotation issues, but that should be verified visually before changing labels.
The next step is failure review by geometry, crop size, and common character confusions before any
new fine-tuning run.

Failure analysis artifacts:

| Evidence | Value |
| --- | ---: |
| Failed full-plate samples | 155 |
| Likely label issues | 3 |
| Likely quality issues | 93 |
| Uncertain failures | 59 |
| Compact failures | 85 |
| Wide failures | 70 |
| Tiny crop failures, area `<500 px` | 55 |
| Severe failures, edit distance `>=7` | 8 |
| Mean edit distance among failures | 2.30 |
| Mean confidence among failures | 0.836 |

The new bucketed review makes the next action clear: inspect `label_review/` first, then
`quality_review/`. The most common edit operations are deletions of digits (`0`, `1`, `3`, `5`,
`7`) and digit confusions such as `7->2`, `2->0`, `6->0`, `6->5`, and `9->0`. That points to a mix
of tiny/blurred crops and a small number of possible label issues, not a need for blind retraining.

Review files:

- `artifacts/ocr/paddleocr-v5-mobile-finetune/eval/error_report.md`;
- `artifacts/ocr/paddleocr-v5-mobile-finetune/eval/error_summary.json`;
- `artifacts/ocr/paddleocr-v5-mobile-finetune/eval/failure_contact_sheet_label.jpg`;
- `artifacts/ocr/paddleocr-v5-mobile-finetune/eval/failure_contact_sheet_quality.jpg`;
- `artifacts/ocr/paddleocr-v5-mobile-finetune/eval/label_review/`;
- `artifacts/ocr/paddleocr-v5-mobile-finetune/eval/quality_review/`;
- `artifacts/ocr/paddleocr-v5-mobile-finetune/eval/failures/`.

Manual review decisions applied before the cleaned evaluation:

- `imgs/train/type3_277.jpg` corrected from `29KT 00576` to `2 003`;
- `imgs/train/type1_683.jpg`, `imgs/train/type2_261.jpg`, and `imgs/val/type2_912.jpg` excluded because the crop is too blurred, truncated, or label evidence is unsafe.

### OCR latency benchmark

A small CPU benchmark was run on the same 100 test plate records with split-compact layout and three
repeats. Timing excludes model initialization but includes image loading and OCR inference.

| Model | Plates | Recognizer inputs | ms/plate | ms/recognizer input |
| --- | ---: | ---: | ---: | ---: |
| Pretrained split-compact | 100 | 149 | 93.11 | 62.49 |
| Fine-tuned split-compact | 100 | 149 | 362.56 | 243.33 |

The fine-tuned model is much more accurate but currently slower on CPU. This should be revisited
before API optimization, especially with ONNX export or GPU inference.

### Continuation experiments

Additional PaddleOCR continuation attempts were run from the current `best_accuracy` model to check
whether more training alone improves the recognizer. They did not beat the frozen fine-tuned model.

| Attempt | Setup | Best PaddleOCR validation acc | Decision |
| --- | --- | ---: | --- |
| Resume optimizer | Continue from `best_accuracy` with optimizer/scheduler state | 0.7549 | Not useful; learning rate had decayed near zero. |
| Reset optimizer, `5e-5` LR | Load `best_accuracy.pdparams`, reset optimizer | 0.7407 | Rejected; validation accuracy dropped. |
| Reset optimizer, `1e-5` LR, no `RecConAug` | Load `best_accuracy.pdparams`, train gently, save to a separate v3 Drive folder | 0.7359 | Rejected; validation accuracy still dropped. |

The selected OCR model remains the original PaddleOCR fine-tuned `best_accuracy` checkpoint with
full-plate evaluation `81.05%` exact match, CER `5.37%`, and character accuracy `94.63%`.

The next OCR work should not be blind retraining. It should focus on reviewing the remaining OCR
failure set, removing unsafe labels or unusable crops, improving compact crop splitting where
needed, and only then launching another controlled fine-tune.

### Quality-filtered training export

A new OCR fine-tune data export now applies an objective crop-area filter to train and validation
records only. Crops with area below `500 px` are excluded from training/validation because the error
analysis shows tiny crops are a dominant failure group. The fixed test split is not filtered by this
rule, so evaluation remains comparable and is not made easier by removing hard test examples.

| Split | Line samples after filter | Skipped records |
| --- | ---: | ---: |
| Train | 6,718 | 658 |
| Validation | 1,104 | 116 |
| Test | 1,254 | 0 |

The refreshed Colab pack uses this filtered training export, starts from the selected
`best_accuracy.pdparams` weights, resets optimizer state, trains for `20` epochs at learning rate
`1e-5`, disables `RecConAug`, and writes checkpoints to
`/content/drive/MyDrive/paddleocr_checkpoints_v3`. This is a controlled data-quality experiment; it
must beat the frozen OCR checkpoint before replacing the selected model.

The filtered run reached PaddleOCR validation accuracy `0.7899` at epoch 8, compared with `0.7549`
for the previous continuation baseline. Because the validation set was changed by the quality
filter, this is only a candidate improvement. The v3 checkpoint must still be exported and evaluated
on the fixed project test split with `configs/ocr-finetune-eval-v3.yaml` before it can replace the
selected OCR model.

The fixed-test evaluation of v3 is now complete. It used the same 818-record test split as the
selected model, so the comparison is valid:

| Model | Exact match | CER | Character accuracy | Decision |
| --- | ---: | ---: | ---: | --- |
| Selected fine-tuned checkpoint | 81.05% | 5.37% | 94.63% | Keep |
| v3 quality-filtered continuation | 77.75% | 6.32% | 93.68% | Reject |

The v3 run improved its filtered validation score, but became worse on the fixed test set. This is
evidence that the validation change made the experiment look better without improving generalization.
The v3 checkpoint is therefore retained as an experiment artifact only; it must not replace the
selected checkpoint. Further progress should come from correcting confirmed labels, handling
unusable crops, and applying plate-format post-processing, followed by evaluation on the unchanged
test split.
### ONNX detection benchmark

The selected YOLO detector was exported from `best.pt` to `best.onnx` at input size 640 and
benchmarked on 20 fixed detection test images on CPU, with three warm-up images per runtime.

| Runtime | Mean latency | p50 | p95 | Throughput |
| --- | ---: | ---: | ---: | ---: |
| PyTorch | 95.09 ms/image | 91.53 ms | 128.35 ms | 10.52 images/s |
| ONNX Runtime | 87.56 ms/image | 80.98 ms | 163.56 ms | 11.42 images/s |

ONNX Runtime improved mean latency by about 8% and throughput by about 9%, but had a worse p95 on
this small CPU sample. It is therefore a deployment candidate, not an automatic replacement. A
larger benchmark and output-equivalence check are required before selecting it for production.
A larger 100-image CPU benchmark confirmed the ONNX advantage:

| Runtime | Mean latency | p50 | p95 | Throughput |
| --- | ---: | ---: | ---: | ---: |
| PyTorch | 103.30 ms/image | 96.66 ms | 168.03 ms | 9.68 images/s |
| ONNX Runtime | 79.42 ms/image | 78.40 ms | 86.53 ms | 12.59 images/s |

On this larger run, ONNX Runtime is 23% faster on mean latency, 19% faster at p50, and 49% faster
at p95. ONNX is therefore the preferred candidate for CPU detection deployment, pending an
output-equivalence check against the PyTorch checkpoint.
