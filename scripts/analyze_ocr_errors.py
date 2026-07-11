"""Analyze OCR failures and prepare evidence for manual review."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageDraw, ImageFont

ReviewBucket = Literal["likely_label_issue", "likely_quality_issue", "uncertain"]


@dataclass(frozen=True)
class OcrFailure:
    """Represent OcrFailure data used by this workflow."""

    sample_id: str
    image_path: str
    ground_truth: str
    prediction: str
    normalized_ground_truth: str
    normalized_prediction: str
    width: int
    height: int
    confidence: float
    edit_distance: int
    exact_match: bool

    @property
    def aspect_ratio(self) -> float:
        """Run the aspect ratio step for this workflow."""
        return self.width / self.height

    @property
    def area(self) -> int:
        """Run the area step for this workflow."""
        return self.width * self.height


@dataclass(frozen=True)
class ReviewedFailure:
    """Represent ReviewedFailure data used by this workflow."""

    failure: OcrFailure
    bucket: ReviewBucket


def _sanitize_component(text: str) -> str:
    """Run the sanitize component step for this workflow."""
    return text.replace("/", "_").replace(chr(92), "_")


def _load_predictions(path: Path) -> tuple[OcrFailure, ...]:
    """Run the load predictions step for this workflow."""
    failures: list[OcrFailure] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            if row.get("exact_match"):
                continue
            try:
                failures.append(
                    OcrFailure(
                        sample_id=str(row["sample_id"]),
                        image_path=str(row["image_path"]),
                        ground_truth=str(row["ground_truth"]),
                        prediction=str(row["prediction"]),
                        normalized_ground_truth=str(row["normalized_ground_truth"]),
                        normalized_prediction=str(row["normalized_prediction"]),
                        width=int(row["width"]),
                        height=int(row["height"]),
                        confidence=float(row["confidence"]),
                        edit_distance=int(row["edit_distance"]),
                        exact_match=bool(row["exact_match"]),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"missing OCR prediction field {path}:{line_number}") from exc
    return tuple(failures)


def _bucket_area(area: int) -> str:
    """Run the bucket area step for this workflow."""
    if area < 500:
        return "tiny_<500px"
    if area < 1000:
        return "small_500_999px"
    if area < 2500:
        return "medium_1000_2499px"
    return "large_>=2500px"


def _bucket_edit_distance(distance: int) -> str:
    """Run the bucket edit distance step for this workflow."""
    if distance <= 1:
        return "minor_1"
    if distance <= 3:
        return "moderate_2_3"
    if distance <= 6:
        return "major_4_6"
    return "severe_>=7"


def _geometry(failure: OcrFailure, compact_aspect_ratio: float) -> str:
    """Run the geometry step for this workflow."""
    return "compact" if failure.aspect_ratio < compact_aspect_ratio else "wide"


def _review_bucket(failure: OcrFailure) -> ReviewBucket:
    """Run the review bucket step for this workflow."""
    small_crop = failure.area < 1000 or min(failure.width, failure.height) < 15
    low_quality = failure.confidence < 0.55 or not failure.prediction.strip()
    if small_crop or low_quality:
        return "likely_quality_issue"
    if failure.confidence >= 0.70 and failure.edit_distance >= 4:
        return "likely_label_issue"
    return "uncertain"


def _substitution_pairs(reference: str, hypothesis: str) -> Counter[str]:
    """Run the substitution pairs step for this workflow."""
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
            )

    pairs: Counter[str] = Counter()
    i = len(reference)
    j = len(hypothesis)
    while i > 0 and j > 0:
        if reference[i - 1] == hypothesis[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
            continue
        if dp[i][j] == dp[i - 1][j - 1] + 1:
            pairs[f"{reference[i - 1]}->{hypothesis[j - 1]}"] += 1
            i -= 1
            j -= 1
            continue
        if dp[i][j] == dp[i - 1][j] + 1:
            pairs[f"{reference[i - 1]}-><del>"] += 1
            i -= 1
            continue
        pairs[f"<ins>->{hypothesis[j - 1]}"] += 1
        j -= 1
    while i > 0:
        pairs[f"{reference[i - 1]}-><del>"] += 1
        i -= 1
    while j > 0:
        pairs[f"<ins>->{hypothesis[j - 1]}"] += 1
        j -= 1
    return pairs


def _rank_failures(failures: Sequence[OcrFailure]) -> list[OcrFailure]:
    """Run the rank failures step for this workflow."""
    return sorted(
        failures,
        key=lambda item: (-item.edit_distance, item.confidence, item.sample_id),
    )


def _copy_ranked_failures(
    failures: Sequence[OcrFailure],
    *,
    dataset_root: Path,
    output_dir: Path,
    limit: int,
) -> list[Path]:
    """Run the copy ranked failures step for this workflow."""
    copied: list[Path] = []
    review_dir = output_dir / "failures"
    review_dir.mkdir(parents=True, exist_ok=True)
    for path in review_dir.iterdir():
        if path.is_file():
            path.unlink()

    for index, failure in enumerate(_rank_failures(failures)[:limit], start=1):
        source = dataset_root / failure.image_path
        if not source.is_file():
            continue
        gt = _sanitize_component(failure.normalized_ground_truth)
        pred = _sanitize_component(failure.normalized_prediction)
        filename = f"{index:02d}_ed{failure.edit_distance}_GT_{gt}_Pred_{pred}{source.suffix}"
        target = review_dir / filename
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def _copy_bucket_review(
    failures: Sequence[ReviewedFailure],
    *,
    dataset_root: Path,
    output_dir: Path,
    name: str,
    limit: int,
) -> None:
    """Run the copy bucket review step for this workflow."""
    review_dir = output_dir / name
    review_dir.mkdir(parents=True, exist_ok=True)
    for path in review_dir.iterdir():
        if path.is_file():
            path.unlink()

    ranked = _rank_failures([item.failure for item in failures])[:limit]
    for index, failure in enumerate(ranked, start=1):
        source = dataset_root / failure.image_path
        if not source.is_file():
            continue
        gt = _sanitize_component(failure.normalized_ground_truth)
        pred = _sanitize_component(failure.normalized_prediction)
        filename = f"{index:02d}_ed{failure.edit_distance}_GT_{gt}_Pred_{pred}{source.suffix}"
        shutil.copy2(source, review_dir / filename)


def _draw_contact_sheet(
    failures: Sequence[ReviewedFailure],
    *,
    dataset_root: Path,
    output_path: Path,
    limit: int,
    title: str,
) -> None:
    """Run the draw contact sheet step for this workflow."""
    ranked = failures[:limit]
    if not ranked:
        return

    columns = 4
    cell_width = 300
    cell_height = 200
    rows = (len(ranked) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, reviewed in enumerate(ranked):
        failure = reviewed.failure
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        image_path = dataset_root / failure.image_path
        if image_path.is_file():
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            image.thumbnail((cell_width - 20, 84), Image.Resampling.LANCZOS)
            sheet.paste(image, (x + 10, y + 10))
        text_y = y + 100
        draw.text((x + 10, y + 86), title[:38], fill="#444444", font=font)
        lines = [
            f"#{index + 1} {reviewed.bucket} ed={failure.edit_distance}",
            f"conf={failure.confidence:.2f} size={failure.width}x{failure.height}",
            f"GT: {failure.normalized_ground_truth}",
            f"PR: {failure.normalized_prediction}",
        ]
        for line in lines:
            draw.text((x + 10, text_y), line[:42], fill="black", font=font)
            text_y += 18
        draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline="#dddddd")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def _write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    """Run the write markdown report step for this workflow."""
    label_count = summary["review_counts"]["label_review"]
    quality_count = summary["review_counts"]["quality_review"]
    uncertain_count = summary["review_counts"]["uncertain"]
    lines = [
        "# OCR error analysis",
        "",
        "This report analyzes incorrect full-plate OCR predictions from the fine-tuned ",
        "PaddleOCR recognizer.",
        "Manual review is still required before changing any labels.",
        "",
        "## Summary",
        "",
        f"- Failed samples: {summary['failed_samples']}",
        f"- Likely label issues: {label_count}",
        f"- Likely quality issues: {quality_count}",
        f"- Uncertain failures: {uncertain_count}",
        f"- Mean failure edit distance: {summary['mean_failure_edit_distance']:.2f}",
        f"- Mean failure confidence: {summary['mean_failure_confidence']:.3f}",
        "",
        "## Review priority",
        "",
        "1. Check `label_review/` first: high-confidence but wrong-looking reads are the ",
        "best label-audit candidates.",
        "2. Check `quality_review/` next: tiny, blurry, or empty-output crops are ",
        "usually data-quality problems.",
        "3. Leave `uncertain` for last: these need manual judgment or cross-checking ",
        "against the source image.",
        "",
        "## Failures by geometry",
        "",
        "| Geometry | Failures |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in summary["by_geometry"].items())
    lines.extend(
        [
            "",
            "## Failures by crop area",
            "",
            "| Area bucket | Failures |",
            "| --- | ---: |",
        ]
    )
    lines.extend(f"| {name} | {count} |" for name, count in summary["by_area"].items())
    lines.extend(
        [
            "",
            "## Most common edit operations",
            "",
            "| Operation | Count |",
            "| --- | ---: |",
        ]
    )
    lines.extend(f"| `{pair}` | {count} |" for pair, count in summary["top_confusions"][:15])
    lines.extend(
        [
            "",
            "## Review artifacts",
            "",
            "- `label_review/`",
            "- `quality_review/`",
            "- `failure_contact_sheet_label.jpg`",
            "- `failure_contact_sheet_quality.jpg`",
            "- `error_summary.json`",
            "",
        ]
    )
    newline = chr(10)
    path.write_text(newline.join(lines), encoding="utf-8", newline=newline)


def analyze_ocr_errors(
    *,
    predictions_path: Path,
    dataset_root: Path,
    output_dir: Path,
    compact_aspect_ratio: float,
    review_limit: int,
) -> dict[str, Any]:
    """Run the analyze ocr errors step for this workflow."""
    failures = _load_predictions(predictions_path)
    if not failures:
        raise ValueError(f"no failed OCR predictions found: {predictions_path}")

    reviewed: list[ReviewedFailure] = [
        ReviewedFailure(failure=failure, bucket=_review_bucket(failure)) for failure in failures
    ]
    by_geometry = Counter(_geometry(item.failure, compact_aspect_ratio) for item in reviewed)
    by_area = Counter(_bucket_area(item.failure.area) for item in reviewed)
    by_edit_distance = Counter(
        _bucket_edit_distance(item.failure.edit_distance) for item in reviewed
    )
    by_bucket = Counter(item.bucket for item in reviewed)
    confusions: Counter[str] = Counter()
    for item in reviewed:
        confusions.update(
            _substitution_pairs(
                item.failure.normalized_ground_truth,
                item.failure.normalized_prediction,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    copied = _copy_ranked_failures(
        [item.failure for item in reviewed],
        dataset_root=dataset_root,
        output_dir=output_dir,
        limit=review_limit,
    )

    label_review = [item for item in reviewed if item.bucket == "likely_label_issue"]
    quality_review = [item for item in reviewed if item.bucket == "likely_quality_issue"]
    uncertain_review = [item for item in reviewed if item.bucket == "uncertain"]

    _copy_bucket_review(
        label_review,
        dataset_root=dataset_root,
        output_dir=output_dir,
        name="label_review",
        limit=review_limit,
    )
    _copy_bucket_review(
        quality_review,
        dataset_root=dataset_root,
        output_dir=output_dir,
        name="quality_review",
        limit=review_limit,
    )
    _draw_contact_sheet(
        label_review,
        dataset_root=dataset_root,
        output_path=output_dir / "failure_contact_sheet_label.jpg",
        limit=review_limit,
        title="Likely label issue",
    )
    _draw_contact_sheet(
        quality_review,
        dataset_root=dataset_root,
        output_path=output_dir / "failure_contact_sheet_quality.jpg",
        limit=review_limit,
        title="Likely quality issue",
    )

    summary: dict[str, Any] = {
        "failed_samples": len(reviewed),
        "review_images": len(copied),
        "mean_failure_confidence": (
            sum(item.failure.confidence for item in reviewed) / len(reviewed)
        ),
        "mean_failure_edit_distance": (
            sum(item.failure.edit_distance for item in reviewed) / len(reviewed)
        ),
        "by_geometry": dict(sorted(by_geometry.items())),
        "by_area": dict(sorted(by_area.items())),
        "by_edit_distance": dict(sorted(by_edit_distance.items())),
        "by_bucket": dict(sorted(by_bucket.items())),
        "top_confusions": confusions.most_common(30),
        "review_counts": {
            "label_review": len(label_review),
            "quality_review": len(quality_review),
            "uncertain": len(uncertain_review),
        },
    }
    newline = chr(10)
    (output_dir / "error_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + newline,
        encoding="utf-8",
    )
    _write_markdown_report(summary, output_dir / "error_report.md")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    """Run the build parser step for this workflow."""
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=root
        / "artifacts"
        / "ocr"
        / "paddleocr-v5-mobile-finetune"
        / "eval"
        / "predictions.jsonl",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=root / "data" / "raw" / "kaggle" / "ocr" / "v1" / "lp_ocr_dataset_vi",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts" / "ocr" / "paddleocr-v5-mobile-finetune" / "eval",
    )
    parser.add_argument("--compact-aspect-ratio", type=float, default=1.5)
    parser.add_argument("--review-limit", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the main step for this workflow."""
    args = _build_parser().parse_args(argv)
    try:
        summary = analyze_ocr_errors(
            predictions_path=args.predictions,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            compact_aspect_ratio=args.compact_aspect_ratio,
            review_limit=args.review_limit,
        )
    except (OSError, ValueError) as exc:
        print(f"OCR error analysis failed: {exc}")
        return 1
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
