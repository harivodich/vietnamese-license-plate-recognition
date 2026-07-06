# Annotation review completed

## Scope

- 100 detection visualizations from the deterministic priority queue.
- 100 OCR visualizations from the deterministic priority queue.
- OCR character-frequency and text-length outliers.
- Review method: visual inspection.

## Decisions

- All 100 sampled detection visualizations contain plausible plate boxes.
- Four byte-identical detection groups have slightly different but plausible boxes. Processed data
  keeps one deterministic canonical record per SHA-256.
- Three OCR labels are corrected in `configs/ocr-corrections.jsonl`.
- Two OCR crops with unreadable plate prefixes are excluded rather than guessed.
- Exact duplicate records are removed before project split assignment.

## Remaining limitation

This review supports internal baseline development. An external benchmark is still required before
publishing final model claims.
