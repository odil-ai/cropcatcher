#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reproduce CropCatcher's SIFT/AKAZE/ORB evaluation charts and results table.

Runs :func:`cropcatcher.evaluation.evaluate` and
:func:`cropcatcher.evaluation.evaluate_transforms` against the packaged
Mandragore dataset, prints results tables, and writes the comparison charts
and example retrieval galleries embedded in the README to
``docs/evaluation/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cropcatcher.evaluation import (
    DEFAULT_METHODS,
    DEFAULT_TRANSFORMS,
    EvaluationCase,
    MethodEvaluation,
    TransformEvaluation,
    evaluate,
    evaluate_transforms,
    load_dataset,
)
from cropcatcher.features import get_extractor
from cropcatcher.matching.pipeline import match_pair
from cropcatcher.sources.iiif import load_iiif_images
from cropcatcher.sources.images import load_image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "tests" / "fixtures" / "mandragore" / "dataset.json"
OUTPUT_DIR = REPO_ROOT / "docs" / "evaluation"

METHOD_COLORS = {"sift": "#4C72B0", "akaze": "#DD8452", "orb": "#55A868"}
METHOD_LABELS = {"sift": "SIFT", "akaze": "AKAZE", "orb": "ORB"}
TRANSFORM_LABELS = {
    "original": "original",
    "rotate_15": "rotate 15°",
    "rotate_45": "rotate 45°",
    "scale_50": "scale 50%",
    "scale_25": "scale 25%",
    "jpeg_q10": "JPEG q10",
    "perspective_heavy": "perspective",
}

EXAMPLE_GALLERIES = [
    ("gaston_phebus_f2v_veneurs", "rotate_45"),
    ("latin757_f24_creation", "perspective_heavy"),
]

CONCEPT_EXAMPLE_CASE = "gaston_phebus_f2v_veneurs"

CONCEPT_EXAMPLE_DECOYS = [
    ("16v", "https://gallica.bnf.fr/iiif/ark:/12148/btv1b525064305/f42"),
    ("48", "https://gallica.bnf.fr/iiif/ark:/12148/btv1b525064305/f105"),
    ("72", "https://gallica.bnf.fr/iiif/ark:/12148/btv1b525064305/f153"),
]


def print_table(results: list[MethodEvaluation]) -> None:
    """Print the per-method results table to stdout.

    :param results: Evaluation results, one per method.
    :type results: list[MethodEvaluation]
    """
    headers = [
        "method",
        "recall@1",
        "recall@3",
        "precision",
        "recall",
        "f1",
        "mean_inliers",
        "inlier_ratio",
        "ms/candidate",
    ]
    rows = [
        [
            r.method,
            f"{r.recall_at_1:.2f}",
            f"{r.recall_at_3:.2f}",
            f"{r.precision:.2f}",
            f"{r.recall:.2f}",
            f"{r.f1:.2f}",
            f"{r.mean_inliers_positive:.1f}",
            f"{r.mean_inlier_ratio_positive:.2f}",
            f"{r.mean_time_per_candidate * 1000:.1f}",
        ]
        for r in results
    ]
    _print_rows(headers, rows)


def print_robustness_table(robustness: dict[str, list[TransformEvaluation]]) -> None:
    """Print the per-method, per-transform robustness table to stdout.

    :param robustness: Mapping of method name to its list of
        :class:`~cropcatcher.evaluation.TransformEvaluation`.
    :type robustness: dict[str, list[TransformEvaluation]]
    """
    headers = [
        "method",
        "transform",
        "recall",
        "mean_inliers",
        "inlier_ratio",
        "mean_score",
    ]
    rows = [
        [
            method,
            t.transform,
            f"{t.recall:.2f}",
            f"{t.mean_inliers:.1f}",
            f"{t.mean_inlier_ratio:.2f}",
            f"{t.mean_score:.2f}",
        ]
        for method, transforms in robustness.items()
        for t in transforms
    ]
    _print_rows(headers, rows)


def _print_rows(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)
    ]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)))


def plot_time(results: list[MethodEvaluation], output_dir: Path) -> None:
    """Plot mean per-candidate time, split into detection and matching.

    :param results: Evaluation results, one per method.
    :type results: list[MethodEvaluation]
    :param output_dir: Directory the chart is written to.
    :type output_dir: Path
    """
    methods = [r.method for r in results]
    detect_ms = [r.mean_candidate_detect_time * 1000 for r in results]
    match_ms = [r.mean_match_time * 1000 for r in results]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(methods))
    ax.bar(x, detect_ms, label="feature detection", color="#4C72B0")
    ax.bar(
        x,
        match_ms,
        bottom=detect_ms,
        label="descriptor matching + RANSAC",
        color="#C44E52",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
    ax.set_ylabel("mean time per candidate (ms)")
    ax.set_title("CropCatcher — time per candidate, by method")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "time_per_candidate.png", dpi=150)
    plt.close(fig)


def plot_quality(results: list[MethodEvaluation], output_dir: Path) -> None:
    """Plot retrieval/detection quality metrics grouped by method.

    :param results: Evaluation results, one per method.
    :type results: list[MethodEvaluation]
    :param output_dir: Directory the chart is written to.
    :type output_dir: Path
    """
    metrics = ["recall_at_1", "recall_at_3", "precision", "recall", "f1"]
    metric_labels = ["Recall@1", "Recall@3", "Precision", "Recall", "F1"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    n_methods = len(results)
    width = 0.8 / n_methods
    x = range(len(metrics))
    for i, r in enumerate(results):
        values = [getattr(r, m) for m in metrics]
        offsets = [xi + (i - (n_methods - 1) / 2) * width for xi in x]
        ax.bar(
            offsets,
            values,
            width=width,
            label=METHOD_LABELS[r.method],
            color=METHOD_COLORS[r.method],
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("CropCatcher — match quality, by method")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "quality.png", dpi=150)
    plt.close(fig)


def plot_tradeoff(results: list[MethodEvaluation], output_dir: Path) -> None:
    """Plot the speed/quality tradeoff: time per candidate vs. Recall@1.

    :param results: Evaluation results, one per method.
    :type results: list[MethodEvaluation]
    :param output_dir: Directory the chart is written to.
    :type output_dir: Path
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for r in results:
        ax.scatter(
            r.mean_time_per_candidate * 1000,
            r.recall_at_1,
            s=140,
            color=METHOD_COLORS[r.method],
            label=METHOD_LABELS[r.method],
        )
        ax.annotate(
            METHOD_LABELS[r.method],
            (r.mean_time_per_candidate * 1000, r.recall_at_1),
            textcoords="offset points",
            xytext=(8, 6),
        )
    ax.set_xlabel("mean time per candidate (ms)")
    ax.set_ylabel("Recall@1")
    ax.set_ylim(0, 1.05)
    ax.set_title("CropCatcher — speed vs. retrieval accuracy")
    fig.tight_layout()
    fig.savefig(output_dir / "tradeoff.png", dpi=150)
    plt.close(fig)


def plot_robustness(
    robustness: dict[str, list[TransformEvaluation]], output_dir: Path
) -> None:
    """Plot recall and mean score under each transformation, by method.

    :param robustness: Mapping of method name to its list of
        :class:`~cropcatcher.evaluation.TransformEvaluation`.
    :type robustness: dict[str, list[TransformEvaluation]]
    :param output_dir: Directory the chart is written to.
    :type output_dir: Path
    """
    transform_names = [t.transform for t in next(iter(robustness.values()))]
    methods = list(robustness.keys())

    fig, (ax_recall, ax_score) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    n_methods = len(methods)
    width = 0.8 / n_methods
    x = range(len(transform_names))

    for i, method in enumerate(methods):
        offsets = [xi + (i - (n_methods - 1) / 2) * width for xi in x]
        recalls = [t.recall for t in robustness[method]]
        scores = [t.mean_score for t in robustness[method]]
        ax_recall.bar(
            offsets,
            recalls,
            width=width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        ax_score.bar(
            offsets,
            scores,
            width=width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )

    ax_recall.set_ylabel("recall\n(match still detected)")
    ax_recall.set_ylim(0, 1.05)
    ax_recall.set_title("CropCatcher — robustness to query-side transformations")
    ax_recall.legend()

    ax_score.set_ylabel("mean score")
    ax_score.set_ylim(0, 1.05)
    ax_score.set_xticks(list(x))
    ax_score.set_xticklabels([TRANSFORM_LABELS.get(t, t) for t in transform_names])

    fig.tight_layout()
    fig.savefig(output_dir / "robustness.png", dpi=150)
    plt.close(fig)


def build_example_gallery(
    cases: list[EvaluationCase],
    case_id: str,
    transform_name: str,
    output_path: Path,
    method: str = "sift",
    thumbnail_size: str = "!320,320",
    top_k: int = 6,
) -> None:
    """Illustrate retrieval on one transformed query among all candidates.

    Transforms one case's query crop, matches it against every candidate
    canvas in the dataset, and renders the query alongside its top-``top_k``
    scoring candidates. The true source highlighted in green, decoys in
    red, so a reader can see the score gap that separates a genuine match
    from unrelated pages even after distortion.

    :param cases: Known query-crop to source-canvas pairs.
    :type cases: list[EvaluationCase]
    :param case_id: Identifier of the case to illustrate.
    :type case_id: str
    :param transform_name: Name of a transform in
        :data:`~cropcatcher.evaluation.DEFAULT_TRANSFORMS` to apply to the
        query crop.
    :type transform_name: str
    :param output_path: File the chart is written to.
    :type output_path: Path
    :param method: Feature extraction method used for illustration.
    :type method: str
    :param thumbnail_size: IIIF Image API size requested for candidate
        thumbnails.
    :type thumbnail_size: str
    :param top_k: Number of top-scoring candidates to display.
    :type top_k: int
    """
    case = next(c for c in cases if c.case_id == case_id)
    transform = DEFAULT_TRANSFORMS[transform_name]

    extractor = get_extractor(method)
    candidate_images = load_iiif_images(
        [c.service_id for c in cases], size=thumbnail_size
    )

    base_image = load_image(case.crop_path)
    transformed = transform(base_image)
    query_kp, query_desc = extractor.detect_and_compute(transformed)

    scored = []
    for other_case, candidate_image in zip(cases, candidate_images, strict=True):
        candidate_kp, candidate_desc = extractor.detect_and_compute(candidate_image)
        result, _matches, _mask = match_pair(
            query_kp,
            query_desc,
            transformed.shape,
            candidate_kp,
            candidate_desc,
            other_case.case_id,
            extractor.norm_type,
            0.75,
            5.0,
            8,
        )
        scored.append((result.score, other_case.case_id, candidate_image, result))
    scored.sort(key=lambda item: item[0], reverse=True)

    top = scored[:top_k]
    if not any(entry[1] == case_id for entry in top):
        true_entry = next(entry for entry in scored if entry[1] == case_id)
        top[-1] = true_entry
        top.sort(key=lambda item: item[0], reverse=True)

    n_cols = 3
    n_rows = -(-len(top) // n_cols)
    fig = plt.figure(figsize=(3.2 * (n_cols + 1), 3.2 * n_rows))
    grid = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=[1.4] + [1] * n_cols)

    ax_query = fig.add_subplot(grid[:, 0])
    ax_query.imshow(transformed, cmap="gray")
    ax_query.set_title(
        f"query crop\n({TRANSFORM_LABELS.get(transform_name, transform_name)})"
    )
    ax_query.set_xticks([])
    ax_query.set_yticks([])

    for k, (score, other_id, candidate_image, result) in enumerate(top):
        row, col = divmod(k, n_cols)
        ax = fig.add_subplot(grid[row, 1 + col])
        ax.imshow(candidate_image, cmap="gray")
        is_true = other_id == case_id
        color = "#2ca02c" if is_true else "#d62728"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
        ax.set_xticks([])
        ax.set_yticks([])
        label = "true source" if is_true else "decoy"
        ax.set_title(
            f"{label}\nscore={score:.2f}  inliers={result.inliers}",
            fontsize=9,
            color=color,
        )

    fig.suptitle(f"Retrieving '{case_id}' with {METHOD_LABELS[method]}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_concept_illustration(
    cases: list[EvaluationCase],
    case_id: str,
    decoy_canvases: list[tuple[str, str]],
    output_path: Path,
    method: str = "sift",
    thumbnail_size: str = "!320,320",
) -> None:
    """Illustrate CropCatcher's core idea: is the query contained in a candidate?

    Renders one real, undistorted query crop next to its true source page
    and a handful of *other real folios from the same manuscript*, a much
    harder set of decoys than pages from a different manuscript, since they
    share the same script, style and parchment. Meant as a small,
    self-explanatory hero image for newcomers, independent of the fuller
    method-comparison and robustness galleries elsewhere in this script.

    :param cases: Known query-crop to source-canvas pairs.
    :type cases: list[EvaluationCase]
    :param case_id: Identifier of the case to illustrate.
    :type case_id: str
    :param decoy_canvases: ``(canvas_label, service_id)`` pairs for other
        canvases of the same manuscript to use as decoys.
    :type decoy_canvases: list[tuple[str, str]]
    :param output_path: File the chart is written to.
    :type output_path: Path
    :param method: Feature extraction method used for illustration.
    :type method: str
    :param thumbnail_size: IIIF Image API size requested for candidate
        thumbnails.
    :type thumbnail_size: str
    """
    case = next(c for c in cases if c.case_id == case_id)
    panels_meta = [
        (True, "true source", case.service_id),
        *((False, f"f. {label}", service_id) for label, service_id in decoy_canvases),
    ]

    extractor = get_extractor(method)
    query_image = load_image(case.crop_path)
    query_kp, query_desc = extractor.detect_and_compute(query_image)

    candidate_images = load_iiif_images(
        [service_id for _, _, service_id in panels_meta], size=thumbnail_size
    )

    panels = []
    for (is_true, label, service_id), candidate_image in zip(
        panels_meta, candidate_images, strict=True
    ):
        candidate_kp, candidate_desc = extractor.detect_and_compute(candidate_image)
        result, _matches, _mask = match_pair(
            query_kp,
            query_desc,
            query_image.shape,
            candidate_kp,
            candidate_desc,
            service_id,
            extractor.norm_type,
            0.75,
            5.0,
            8,
        )
        panels.append((is_true, label, candidate_image, result))
    panels.sort(key=lambda item: item[3].score, reverse=True)

    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(3.1 * (len(panels) + 1), 3.6))

    axes[0].imshow(query_image, cmap="gray")
    axes[0].set_title("query crop", fontsize=11)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#333333")
        spine.set_linewidth(2)

    for ax, (is_true, label, candidate_image, result) in zip(
        axes[1:], panels, strict=True
    ):
        ax.imshow(candidate_image, cmap="gray")
        color = "#2ca02c" if is_true else "#d62728"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
        ax.set_xticks([])
        ax.set_yticks([])
        status = "MATCH" if is_true else "no match"
        ax.set_title(
            f"{status}\n{label}  score={result.score:.2f}", fontsize=10, color=color
        )

    fig.suptitle(
        "Is the query crop contained in a candidate folio? (same manuscript)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the evaluations, print the results tables, and write the charts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = load_dataset(DATASET_PATH)

    build_concept_illustration(
        cases,
        CONCEPT_EXAMPLE_CASE,
        CONCEPT_EXAMPLE_DECOYS,
        OUTPUT_DIR / "concept.png",
    )

    results = evaluate(DATASET_PATH, methods=DEFAULT_METHODS)
    print_table(results)
    plot_time(results, OUTPUT_DIR)
    plot_quality(results, OUTPUT_DIR)
    plot_tradeoff(results, OUTPUT_DIR)

    print()
    robustness = evaluate_transforms(DATASET_PATH, methods=DEFAULT_METHODS)
    print_robustness_table(robustness)
    plot_robustness(robustness, OUTPUT_DIR)

    for case_id, transform_name in EXAMPLE_GALLERIES:
        build_example_gallery(
            cases, case_id, transform_name, OUTPUT_DIR / f"example_{case_id}.png"
        )

    summary_path = OUTPUT_DIR / "results.json"
    summary_path.write_text(
        json.dumps(
            {
                "methods": [
                    {
                        "method": r.method,
                        "recall_at_1": r.recall_at_1,
                        "recall_at_3": r.recall_at_3,
                        "precision": r.precision,
                        "recall": r.recall,
                        "f1": r.f1,
                        "mean_inliers_positive": r.mean_inliers_positive,
                        "mean_inlier_ratio_positive": r.mean_inlier_ratio_positive,
                        "mean_score_positive": r.mean_score_positive,
                        "mean_score_negative": r.mean_score_negative,
                        "mean_candidate_detect_time_ms": r.mean_candidate_detect_time
                        * 1000,
                        "mean_query_detect_time_ms": r.mean_query_detect_time * 1000,
                        "mean_match_time_ms": r.mean_match_time * 1000,
                        "mean_time_per_candidate_ms": r.mean_time_per_candidate * 1000,
                    }
                    for r in results
                ],
                "robustness": {
                    method: [
                        {
                            "transform": t.transform,
                            "recall": t.recall,
                            "mean_inliers": t.mean_inliers,
                            "mean_inlier_ratio": t.mean_inlier_ratio,
                            "mean_score": t.mean_score,
                        }
                        for t in transforms
                    ]
                    for method, transforms in robustness.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nCharts and results.json written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
