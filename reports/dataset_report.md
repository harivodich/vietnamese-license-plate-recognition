# Dataset report

## Status

Detection and OCR are treated as separate source datasets. Both selected sources have been
downloaded into versioned immutable directories with typed completion receipts and deterministic
content fingerprints. Both sources have passed the initial structural, pairing, label parsing, and
full-image decode checks. Automated statistics and duplicate detection are complete. Near-duplicate
candidate review and manual annotation review are not yet complete.

This split is intentional. For this project, "end-to-end" refers to the pipeline output, not to a
requirement that one public dataset must contain every annotation type. Separate component datasets
are acceptable as long as the final benchmark set is frozen and human-verified.

## Detection source

- Kaggle handle: `miahuynh04/vietnamese-license-plate-detection`
- Dataset version: `1`
- Retrieved at: `2026-07-02T03:00:16Z`
- Data-card license claim: MIT
- Versioned raw path: `data/raw/kaggle/detection/v1`
- Receipt file count: 16,519
- Receipt content size: 366,308,679 bytes
- Content SHA-256: `51de1ea6a00699bffd8c9c8cd58fbeefa3eccdcb9404b8a276152ca31a1a0956`

The license value above is not yet treated as verified provenance. A copy of the license or other
primary evidence distributed with the dataset has not been found in the downloaded files.

## Detection structural inventory

| Split | Images | Label files | Bounding boxes | Multi-plate images |
|---|---:|---:|---:|---:|
| Train | 6,607 | 6,607 | 6,785 | 117 |
| Validation | 814 | 814 | 818 | 3 |
| Test | 838 | 838 | 849 | 9 |
| **Total** | **8,259** | **8,259** | **8,452** | **129** |

All images currently use the `.jpg` extension. Labels use YOLO detection format with one class:

```text
class_id center_x center_y width height
```

The initial structural scan found:

- zero missing image-label pairs;
- zero orphan label files;
- zero empty label files;
- no rows with a field count other than five;
- no normalized coordinate values outside `[0, 1]`;
- all 8,259 images decode successfully;
- all 8,452 bounding boxes pass the typed YOLO schema;
- no OCR text labels.

Seventeen boxes touch an image boundary and exceed the mathematical edge by approximately
`5e-7` after their source values are rounded to six decimal places. The schema accepts boundary
error up to `1e-6`, while retaining the original values and rejecting material overflow. Raw
annotations are not clamped or edited.

Measured detection statistics:

| Property | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Image width (pixels) | 216 | 472 | 4,653 |
| Image height (pixels) | 159 | 303 | 2,910 |
| Normalized bbox width | 0.015625 | 0.190678 | 0.703333 |
| Normalized bbox height | 0.011662 | 0.115512 | 0.668750 |
| Normalized bbox area | 0.000210 | 0.023270 | 0.470354 |

## OCR source decision

Selected OCR baseline source:

- Kaggle handle: `wirqhuy/vietnamese-license-plate-ocr`
- Dataset version: `1`
- Data-card license claim: MIT
- Retrieved at: `2026-07-02T03:00:54Z`
- Versioned raw path: `data/raw/kaggle/ocr/v1`
- Receipt file count: 6,646
- Receipt content size: 10,968,384 bytes
- Content SHA-256: `a0042b3f9af82f38fbb1894f869e303cbbd6a25b8690ed834f1aeafeb676ede9`
- Structure observed during inspection:
  - `imgs/train`: 5,314 images
  - `imgs/val`: 1,329 images
  - `labels/train.txt` and `labels/val.txt` with `image_path<TAB>plate_text`

Sample labels observed:

```text
imgs/train/type5_258.jpg    BT 5581
imgs/train/type7_527.jpg    51G 46455
imgs/val/type4_628.jpg      73B 0040
```

The initial OCR structural scan found:

- 5,314 train image-label pairs;
- 1,329 validation image-label pairs;
- zero missing images;
- zero unreferenced images;
- zero empty paths or labels;
- exactly one tab separator in every label row;
- all 6,643 images decode successfully;
- UTF-8 labels include spaces and the Vietnamese character `Đ`.

Measured OCR statistics:

| Property | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Crop width (pixels) | 9 | 46 | 556 |
| Crop height (pixels) | 6 | 31 | 350 |
| Raw text length | 5 | 9 | 11 |

The observed character set is:

```text
 0123456789ABCDEFGHIJKLMNOPQRSTUVXYZĐ
```

## Duplicate audit

The audit uses SHA-256 for exact duplicates and 64-bit difference hashes for near-duplicate
candidates. Near candidates use a Hamming-distance threshold of 6.

| Task | Exact groups | Exact groups crossing source splits | Near candidate pairs | Near pairs crossing source splits |
|---|---:|---:|---:|---:|
| Detection | 4 | 1 | 9,888 | 3,332 |
| OCR | 93 | 31 | 629 | 208 |

Exact duplicates are byte-identical evidence. The source-provided splits therefore contain known
leakage: one exact detection group and 31 exact OCR groups cross source split boundaries. The
project split must keep each duplicate group together instead of trusting the source split.

All four exact detection groups have slightly different bounding boxes for byte-identical images.
Two exact OCR groups have conflicting text:

```text
imgs/train/car_338.jpg -> 30A 34588
imgs/train/car_352.jpg -> 30A 31588

imgs/train/car_553.jpg -> 30G 31553
imgs/train/car_522.jpg -> 80G 31553
```

These conflicts require manual review and a separate correction manifest. Raw labels are not
modified in place.

Near-duplicate counts are candidate counts, not confirmed duplicate counts. The high detection
count shows that the current dHash threshold also retrieves visually similar layouts. Candidates
must be visualized and sampled before they are used to create final duplicate groups.

## Manual-review queue

A deterministic review queue has been generated from the manifest and audit fingerprints:

```text
data/interim/manual_review/86998f01-20bc6900-20260701/
```

It contains:

- 100 detection visualizations with bounding boxes overlaid;
- 100 OCR visualizations with the raw label below each crop;
- `review_queue.jsonl` with the sample id, source path, priority reasons, and `pending` status.

The queue prioritizes annotation conflicts, exact duplicates crossing source splits, and the closest
near-duplicate candidates before filling the remaining capacity with a seeded random sample.
Generating the queue does not count as completing manual review: no sample is marked approved,
rejected, or corrected until a person records that decision separately.

Secondary OCR source kept for possible augmentation only:

- Kaggle handle: `topkek69/vietnamese-license-plate-ocr`
- Structure observed during inspection:
  - `cropped`: 6,643 images
  - `generated`: 5,547 images
  - `labels/crop_labels.csv`
  - `labels/gen_labels.csv`

This source mixes real cropped plates with generated images. It may help OCR robustness later, but
it should not be treated as the primary benchmark dataset until we quantify the effect of synthetic
data.

Rejected as OCR source:

- `johnkhanhnguyen/vietnamese-license-plate`

This dataset contains YOLO detection labels only and does not provide OCR strings.

## Current limitations

- The source-provided split has not been accepted as the project split.
- Completion receipts prove source identity and byte-level contents, not annotation correctness.
- No video, vehicle, capture-session, or duplicate group identifiers are present.
- The selected OCR dataset has text labels, but it is not paired with full-scene plate bounding
  boxes from the detection dataset.
- Bounding boxes have not yet been checked visually.
- Near-duplicate candidates have not yet been visually confirmed.
- Exact-duplicate annotation conflicts have not yet been manually resolved.
- The 100 detection and 100 OCR review queues exist, but human decisions have not yet been recorded.
- A final end-to-end test set with human-verified plate text still needs to be built.
- A legacy unversioned detection download remains under `data/raw/kaggle`; it is ignored by the
  configured pipeline and has not been deleted automatically.
