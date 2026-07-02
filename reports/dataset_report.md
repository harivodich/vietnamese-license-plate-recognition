# Dataset report

## Status

Detection and OCR are treated as separate source datasets. Both selected sources have been
downloaded into versioned immutable directories with typed completion receipts and deterministic
content fingerprints. Detection has an earlier structural inventory; OCR has not yet been fully
audited for duplicates, image integrity, or label quality.

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
- no OCR text labels.

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
- Images have not yet been decoded to detect corruption or measure resolution.
- Bounding boxes have not yet been checked visually.
- Exact and near-duplicate leakage has not yet been measured.
- A final end-to-end test set with human-verified plate text still needs to be built.
- A legacy unversioned detection download remains under `data/raw/kaggle`; it is ignored by the
  configured pipeline and has not been deleted automatically.
