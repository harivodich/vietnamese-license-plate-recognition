# OCR baseline report

## Protocol

The OCR baseline evaluates `en_PP-OCRv5_mobile_rec` on the fixed OCR `test` split using
ground-truth plate crops. The detector is not involved, so these results isolate recognition
quality from detection and crop errors.

Shared runtime settings:

- inference device: CPU;
- batch size: 32;
- CPU threads: 8;
- MKL-DNN: enabled;
- text detection: disabled;
- input source: ground-truth plate crops;
- normalization: Unicode NFKC, uppercase, then remove non-alphanumeric separators.

## Why layout matters

`en_PP-OCRv5_mobile_rec` is a line recognizer. A wide one-line plate crop matches that assumption.
A compact two-line crop does not. Scoring compact crops directly measures a layout mismatch as much
as recognition quality.

The project therefore keeps two OCR baselines:

1. `original`: send each plate crop directly to the recognizer.
2. `split_compact`: keep wide crops unchanged, split compact crops into top/bottom line crops,
   recognize each line, then concatenate the normalized line predictions before calculating the
   full-plate metric.

## Result summary

| Experiment | Test crops | Exact matches | Full-plate exact match | CER | Character accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original crops, all geometry | 818 | 159 | 0.1944 | 0.5938 | 0.4062 |
| Original crops, wide only | 374 | 159 | 0.4251 | 0.3477 | 0.6523 |
| Split compact layout, all geometry | 818 | 308 | 0.3765 | 0.3241 | 0.6759 |

The layout-aware baseline roughly doubles full-plate exact matches compared with direct OCR on all
crops: 159 -> 308. CER improves from 0.5938 to 0.3241.

## Geometry breakdown for split-compact layout

| Geometry | Crops | Exact matches | Exact match | CER | Character accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compact, split into two lines | 444 | 149 | 0.3356 | 0.3054 | 0.6946 |
| Wide, unchanged | 374 | 159 | 0.4251 | 0.3477 | 0.6523 |

The compact group changes from zero exact matches under direct recognition to 149 exact matches
after deterministic row splitting. This confirms that the main compact failure was input layout,
not just lack of OCR capacity.

## Remaining failure modes

- The pretrained English recognizer still does not model the full Vietnamese license-plate charset
  perfectly, especially project-specific characters such as `Đ`.
- Some compact crops are very small, so row splitting creates tiny line images with limited detail.
- Confidence is useful for ranking failures but is not a correctness guarantee.
- The split threshold is deterministic and simple; it has not been tuned with a dedicated validation
  search.

## Artifacts

- Original full-split metrics: [metrics.json](metrics.json)
- Original full-split predictions: [predictions.jsonl](predictions.jsonl)
- Split-compact metrics: [layout_metrics.json](layout_metrics.json)

The current decision is to use pretrained OCR plus layout handling as the main OCR path. The
CRNN+CTC trainer remains a scratch baseline for teaching, sanity checks, checkpointing, and metric
validation. The next production-oriented step is pretrained OCR fine-tuning with the project
charset and the same compact/wide subgroup reporting.
