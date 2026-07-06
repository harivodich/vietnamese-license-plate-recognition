# OCR baseline report

## Protocol

The baseline evaluates `en_PP-OCRv5_mobile_rec` directly on the 818 fixed test crops from the OCR
dataset. The detector is not involved, so these results isolate recognition quality from detection
and crop errors.

- inference device: CPU;
- batch size: 32;
- CPU threads: 8;
- MKL-DNN: enabled;
- text detection: disabled;
- input: ground-truth plate crops;
- normalization: Unicode NFKC, uppercase, then remove non-alphanumeric separators.

Raw predictions are retained alongside normalized strings. Normalization therefore does not hide
the model output used to calculate each edit distance.

## Overall result

| Metric | Value |
| --- | ---: |
| Test crops | 818 |
| Exact matches | 159 |
| Full-plate exact match | 0.1944 |
| Character error rate | 0.5938 |
| Character accuracy | 0.4062 |
| Mean model confidence | 0.5832 |
| Empty predictions | 47 |
| Inference latency | 62.86 ms/image |
| Throughput | 15.91 images/second |

This pretrained model is not an acceptable final recognizer. Approximately one in five plates is
fully correct, and the aggregate edit distance is 3,952 over 6,656 ground-truth characters.

## Geometry breakdown

The crop aspect ratio is used only as a reproducible geometry proxy. `Compact` does not claim that
every image is a two-line plate.

| Geometry | Crops | Exact matches | Exact match | CER |
| --- | ---: | ---: | ---: | ---: |
| Compact, width/height below 1.5 | 444 | 0 | 0.0000 | 0.7883 |
| Wide, width/height at least 1.5 | 374 | 159 | 0.4251 | 0.3477 |

The zero exact-match rate on compact crops is the strongest measured failure mode. The same
recognizer is materially better on wide crops, so layout mismatch should be tested before spending
compute on fine-tuning.

## Error analysis

- All 159 exact matches come from the wide group.
- Forty-seven crops produce an empty string.
- Eighteen labels contain `Đ`; none is an exact match because the English pretrained charset does
  not represent this project-specific character.
- Mean confidence is 0.9519 for exact predictions and 0.4942 for incorrect predictions.
- Confidence alone is not a correctness guarantee: 25 incorrect predictions have confidence at
  least 0.9, and the highest incorrect confidence is 0.9943.

The 25 predictions with the largest edit distance are embedded in
[metrics.json](metrics.json). Every sample-level raw prediction, normalized prediction,
confidence, edit distance, and exact-match flag is retained in
[predictions.jsonl](predictions.jsonl).

## Decision

Do not fine-tune immediately. The next controlled experiment should preserve direct recognition
for wide crops and transform compact crops into ordered text-line inputs before recognition.
Candidate row splitting must be designed and selected on the validation split only.

If validation results remain weak after layout handling, fine-tuning becomes justified. A
fine-tuned recognizer must include the exact project charset, including `Đ`, and must report:

- character error rate;
- character accuracy;
- full-plate exact match;
- compact and wide subgroup metrics;
- latency with the same CPU protocol.

The current test result is frozen as the pretrained OCR baseline. The test split should not be
reused repeatedly while choosing preprocessing thresholds or model variants.
