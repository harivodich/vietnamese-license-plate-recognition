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

The baseline completed all 50 epochs. The best validation result was recorded at epoch 47:

| Metric | Value |
| --- | ---: |
| Precision | 0.9924 |
| Recall | 0.9754 |
| mAP@0.5 | 0.9930 |
| mAP@0.5:0.95 | 0.7173 |
| Validation inference latency | 7.3 ms/image |

Training took 4.859 hours on an NVIDIA Quadro M2200. The local `best.pt` checkpoint has SHA-256
`70289013711AB5DA541C2A3B6EB44052756C0DFD6FF3EF16039AD83A5856CA3C`. Model checkpoints and full
run artifacts are intentionally excluded from Git.

These are validation metrics, not final test-set metrics. The fixed test set remains untouched for
the subsequent evaluation step.

Tracked training evidence:

- [epoch metrics](detection_baseline/results.csv);
- [training curves](detection_baseline/results.png);
- [precision-recall curve](detection_baseline/BoxPR_curve.png);
- [F1-confidence curve](detection_baseline/BoxF1_curve.png);
- [confusion matrix](detection_baseline/confusion_matrix.png);
- [normalized confusion matrix](detection_baseline/confusion_matrix_normalized.png).
