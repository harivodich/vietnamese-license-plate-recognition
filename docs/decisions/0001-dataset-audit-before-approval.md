# ADR 0001: Audit the Kaggle dataset before approving it

- Status: accepted

## Context

The Kaggle data card for `miahuynh04/vietnamese-license-plate-detection` reports approximately
16,500 files and an MIT license. The public description does not establish whether OCR text,
vehicle/video group identifiers, and complete provenance are present.

## Decision

Treat this dataset as a candidate, not an approved training source. Download into immutable raw
storage and audit file integrity, annotation format, bounding boxes, text labels, duplicates,
near-duplicates, source grouping, and license evidence before defining the split.

## Consequences

Detection work may proceed only after bounding-box labels pass review. OCR work requires verified
text labels or a separately licensed OCR dataset. A random image-level split is forbidden when
frames or vehicle identities can cross subsets.
