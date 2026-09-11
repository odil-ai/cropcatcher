#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation utilities for CropCatcher.

This module benchmarks CropCatcher's feature-matching methods (SIFT, AKAZE,
ORB) against a dataset of known query-crop to source-canvas pairs backed by
real IIIF resources. Every query crop is matched against every candidate
canvas in the dataset (a full similarity matrix, not only its own true
source), from which retrieval accuracy, match-detection precision/recall,
RANSAC inlier statistics and processing time are derived for each method.

The dataset is described by a JSON file (see
``tests/fixtures/mandragore/dataset.json`` for the packaged example built
from the BnF "Mandragore" dataset): a list of cases, each pointing to a
locally cached query crop image and the IIIF Image API service id of its
true source canvas.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .features import get_extractor
from .matching.pipeline import match_pair
from .result import MatchResult
from .sources.iiif import load_iiif_images
from .sources.images import load_image

DEFAULT_METHODS = ("sift", "akaze", "orb")

Transform = Callable[[np.ndarray], np.ndarray]


def _rotate(angle_degrees: float) -> Transform:
    """Build a transform that rotates an image, expanding the canvas so
    nothing is cropped out."""

    def transform(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        center = (width / 2.0, height / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_width = int(height * sin + width * cos)
        new_height = int(height * cos + width * sin)
        rotation_matrix[0, 2] += (new_width / 2.0) - center[0]
        rotation_matrix[1, 2] += (new_height / 2.0) - center[1]
        return cv2.warpAffine(
            image, rotation_matrix, (new_width, new_height), borderValue=255
        )

    return transform


def _scale(factor: float) -> Transform:
    """Build a transform that resizes an image by ``factor``."""

    def transform(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        interpolation = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC
        new_size = (max(1, int(width * factor)), max(1, int(height * factor)))
        return cv2.resize(image, new_size, interpolation=interpolation)

    return transform


def _jpeg_recompress(quality: int) -> Transform:
    """Build a transform that re-encodes an image through lossy JPEG."""

    def transform(image: np.ndarray) -> np.ndarray:
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("Could not JPEG-encode image for recompression")
        return cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)

    return transform


def _perspective_warp(magnitude: float, seed: int = 0) -> Transform:
    """Build a transform applying a mild, deterministic perspective warp.

    Each corner of the image is jittered by up to ``magnitude`` times the
    shortest side, simulating a page photographed at a slight angle.
    """

    def transform(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        src = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        jitter = magnitude * min(height, width)
        rng = np.random.default_rng(seed)
        dst = src + rng.uniform(-jitter, jitter, size=src.shape).astype(np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, matrix, (width, height), borderValue=255)

    return transform


DEFAULT_TRANSFORMS: dict[str, Transform] = {
    "original": lambda image: image,
    "rotate_15": _rotate(15.0),
    "rotate_45": _rotate(45.0),
    "scale_50": _scale(0.5),
    "scale_25": _scale(0.25),
    "jpeg_q10": _jpeg_recompress(10),
    "perspective_heavy": _perspective_warp(0.15),
}


@dataclass
class EvaluationCase:
    """A single known query-crop to source-canvas pair.

    :param case_id: Identifier of the case, as given in the dataset file.
    :type case_id: str
    :param crop_path: Local path to the query crop image.
    :type crop_path: Path
    :param service_id: IIIF Image API service id of the true source canvas.
    :type service_id: str
    :param manifest_url: IIIF Presentation manifest the true source canvas
        belongs to, if given in the dataset file.
    :type manifest_url: str | None
    """

    case_id: str
    crop_path: Path
    service_id: str
    manifest_url: str | None = None


@dataclass
class MethodEvaluation:
    """Aggregated evaluation metrics for a single feature-matching method.

    Retrieval and detection metrics are computed over the full
    ``len(cases) x len(cases)`` similarity matrix: each case's query crop is
    matched against every candidate canvas, not only its own true source.

    :param method: Feature extraction method that was evaluated.
    :type method: str
    :param recall_at_1: Fraction of queries whose true source canvas ranks
        first by score among all candidates.
    :type recall_at_1: float
    :param recall_at_3: Fraction of queries whose true source canvas ranks
        in the top 3 by score among all candidates.
    :type recall_at_3: float
    :param precision: Precision of the match/no-match decision (thresholding
        RANSAC inliers), over every query/candidate pair.
    :type precision: float
    :param recall: Recall of the match/no-match decision, over every
        query/candidate pair.
    :type recall: float
    :param f1: Harmonic mean of ``precision`` and ``recall``.
    :type f1: float
    :param mean_inliers_positive: Mean RANSAC inlier count on true
        query/source pairs.
    :type mean_inliers_positive: float
    :param mean_inlier_ratio_positive: Mean inlier ratio on true
        query/source pairs.
    :type mean_inlier_ratio_positive: float
    :param mean_score_positive: Mean match score on true query/source pairs.
    :type mean_score_positive: float
    :param mean_score_negative: Mean match score on unrelated query/candidate
        pairs.
    :type mean_score_negative: float
    :param mean_candidate_detect_time: Mean time to detect and describe
        features on one candidate canvas, in seconds.
    :type mean_candidate_detect_time: float
    :param mean_query_detect_time: Mean time to detect and describe features
        on one query crop, in seconds.
    :type mean_query_detect_time: float
    :param mean_match_time: Mean time to match one query/candidate pair
        (descriptor matching + RANSAC), in seconds.
    :type mean_match_time: float
    :param mean_time_per_candidate: ``mean_candidate_detect_time +
        mean_match_time`` — the marginal cost of adding one more candidate to
        a search, as paid by :meth:`Matcher.search_iiif`.
    :type mean_time_per_candidate: float
    :param matrix: Raw ``len(cases) x len(cases)`` matrix of match results,
        ``matrix[i][j]`` being case ``i``'s query matched against case
        ``j``'s candidate canvas.
    :type matrix: list[list[MatchResult]]
    """

    method: str
    recall_at_1: float
    recall_at_3: float
    precision: float
    recall: float
    f1: float
    mean_inliers_positive: float
    mean_inlier_ratio_positive: float
    mean_score_positive: float
    mean_score_negative: float
    mean_candidate_detect_time: float
    mean_query_detect_time: float
    mean_match_time: float
    mean_time_per_candidate: float
    matrix: list[list[MatchResult]] = field(repr=False)


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def load_dataset(dataset_path: str | Path) -> list[EvaluationCase]:
    """Load an evaluation dataset description.

    :param dataset_path: Path to a CropCatcher evaluation ``dataset.json``
        file. ``crop_path`` entries are resolved relative to its directory.
    :type dataset_path: str | Path
    :return: The known query-crop to source-canvas pairs described by the
        dataset.
    :rtype: list[EvaluationCase]
    """
    dataset_path = Path(dataset_path)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    base_dir = dataset_path.parent
    return [
        EvaluationCase(
            case_id=case["id"],
            crop_path=base_dir / case["crop_path"],
            service_id=case["service_id"],
            manifest_url=case.get("manifest_url"),
        )
        for case in payload["cases"]
    ]


def evaluate_method(
    cases: list[EvaluationCase],
    method: str,
    size: str = "!1024,1024",
    ratio: float = 0.75,
    ransac_reproj_threshold: float = 5.0,
    min_inliers: int = 8,
    extractor_options: dict | None = None,
    mutual_check: bool = False,
) -> MethodEvaluation:
    """Benchmark a single feature-matching method over a full similarity matrix.

    Candidate canvases are downloaded once and their features computed once,
    then reused for every query, so the reported per-candidate time reflects
    the cost of matching one candidate rather than of downloading it once per
    query (as :class:`~cropcatcher.index.Index` would when reused).

    :param cases: Known query-crop to source-canvas pairs to evaluate.
    :type cases: list[EvaluationCase]
    :param method: Feature extraction method (``"sift"``, ``"akaze"`` or
        ``"orb"``).
    :type method: str
    :param size: IIIF Image API size requested for each candidate canvas.
    :type size: str
    :param ratio: Lowe's ratio test threshold for descriptor matching.
    :type ratio: float
    :param ransac_reproj_threshold: RANSAC reprojection threshold, in pixels.
    :type ransac_reproj_threshold: float
    :param min_inliers: Minimum RANSAC inliers required to accept a match.
    :type min_inliers: int
    :param extractor_options: Extra keyword arguments for the feature
        extractor, e.g. ``{"nfeatures": 2000}`` to cap SIFT keypoints.
    :type extractor_options: dict | None
    :param mutual_check: Require descriptor matches to be symmetric, which
        greatly improves precision at the cost of a second matching pass.
    :type mutual_check: bool
    :return: Aggregated metrics for this method.
    :rtype: MethodEvaluation
    """
    extractor = get_extractor(method, **(extractor_options or {}))

    candidate_images = load_iiif_images([case.service_id for case in cases], size=size)

    candidate_features = []
    candidate_detect_times = []
    for image in candidate_images:
        start = time.perf_counter()
        keypoints, descriptors = extractor.detect_and_compute(image)
        candidate_detect_times.append(time.perf_counter() - start)
        candidate_features.append((keypoints, descriptors, image.shape))

    query_features = []
    query_detect_times = []
    for case in cases:
        query_image = load_image(case.crop_path)
        start = time.perf_counter()
        keypoints, descriptors = extractor.detect_and_compute(query_image)
        query_detect_times.append(time.perf_counter() - start)
        query_features.append((keypoints, descriptors, query_image.shape))

    matrix: list[list[MatchResult]] = []
    match_times: list[float] = []
    for query_kp, query_desc, query_shape in query_features:
        row = []
        for j, (candidate_kp, candidate_desc, _shape) in enumerate(candidate_features):
            start = time.perf_counter()
            result, _matches, _mask = match_pair(
                query_kp,
                query_desc,
                query_shape,
                candidate_kp,
                candidate_desc,
                cases[j].service_id,
                extractor.norm_type,
                ratio,
                ransac_reproj_threshold,
                min_inliers,
                mutual_check,
            )
            match_times.append(time.perf_counter() - start)
            row.append(result)
        matrix.append(row)

    return _aggregate(
        method,
        matrix,
        min_inliers,
        candidate_detect_times,
        query_detect_times,
        match_times,
    )


def _aggregate(
    method: str,
    matrix: list[list[MatchResult]],
    min_inliers: int,
    candidate_detect_times: list[float],
    query_detect_times: list[float],
    match_times: list[float],
) -> MethodEvaluation:
    n = len(matrix)
    recall_at_1_hits = 0
    recall_at_3_hits = 0
    true_positives = false_positives = false_negatives = 0
    positive_inliers: list[int] = []
    positive_inlier_ratios: list[float] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []

    for i, row in enumerate(matrix):
        ranked = sorted(range(n), key=lambda j: row[j].score, reverse=True)
        if ranked[0] == i:
            recall_at_1_hits += 1
        if i in ranked[:3]:
            recall_at_3_hits += 1

        for j, result in enumerate(row):
            predicted_match = result.inliers >= min_inliers
            if i == j:
                positive_inliers.append(result.inliers)
                positive_inlier_ratios.append(result.inlier_ratio)
                positive_scores.append(result.score)
                if predicted_match:
                    true_positives += 1
                else:
                    false_negatives += 1
            else:
                negative_scores.append(result.score)
                if predicted_match:
                    false_positives += 1

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives)
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return MethodEvaluation(
        method=method,
        recall_at_1=recall_at_1_hits / n if n else 0.0,
        recall_at_3=recall_at_3_hits / n if n else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_inliers_positive=_mean(positive_inliers),
        mean_inlier_ratio_positive=_mean(positive_inlier_ratios),
        mean_score_positive=_mean(positive_scores),
        mean_score_negative=_mean(negative_scores),
        mean_candidate_detect_time=_mean(candidate_detect_times),
        mean_query_detect_time=_mean(query_detect_times),
        mean_match_time=_mean(match_times),
        mean_time_per_candidate=_mean(candidate_detect_times) + _mean(match_times),
        matrix=matrix,
    )


def evaluate(
    dataset_path: str | Path,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    **kwargs,
) -> list[MethodEvaluation]:
    """Benchmark several feature-matching methods against one dataset.

    :param dataset_path: Path to a CropCatcher evaluation ``dataset.json`` file.
    :type dataset_path: str | Path
    :param methods: Feature extraction methods to evaluate.
    :type methods: tuple[str, ...]
    :param kwargs: Extra keyword arguments forwarded to :func:`evaluate_method`.
    :return: One :class:`MethodEvaluation` per method, in the given order.
    :rtype: list[MethodEvaluation]
    """
    cases = load_dataset(dataset_path)
    return [evaluate_method(cases, method, **kwargs) for method in methods]


@dataclass
class TransformEvaluation:
    """Robustness of one method to one query-side image transformation.

    Each case's query crop is transformed and matched only against its own
    true source canvas (not the full corpus), measuring how much a
    transformation degrades an otherwise-correct match rather than retrieval
    accuracy across a corpus.

    :param method: Feature extraction method that was evaluated.
    :type method: str
    :param transform: Name of the applied transformation.
    :type transform: str
    :param recall: Fraction of cases still detected as a match
        (``inliers >= min_inliers``) after the transformation.
    :type recall: float
    :param mean_inliers: Mean RANSAC inlier count after the transformation.
    :type mean_inliers: float
    :param mean_inlier_ratio: Mean inlier ratio after the transformation.
    :type mean_inlier_ratio: float
    :param mean_score: Mean match score after the transformation.
    :type mean_score: float
    """

    method: str
    transform: str
    recall: float
    mean_inliers: float
    mean_inlier_ratio: float
    mean_score: float


def evaluate_transform_robustness(
    cases: list[EvaluationCase],
    method: str,
    transforms: dict[str, Transform] = DEFAULT_TRANSFORMS,
    size: str = "!1024,1024",
    ratio: float = 0.75,
    ransac_reproj_threshold: float = 5.0,
    min_inliers: int = 8,
    extractor_options: dict | None = None,
) -> list[TransformEvaluation]:
    """Benchmark one method's robustness to query-side image transformations.

    Unlike :func:`evaluate_method`, each transformed query is matched only
    against its own true source canvas: the purpose is to measure how much a
    transformation degrades an already-correct match, not retrieval accuracy
    across a corpus.

    :param cases: Known query-crop to source-canvas pairs to evaluate.
    :type cases: list[EvaluationCase]
    :param method: Feature extraction method (``"sift"``, ``"akaze"`` or
        ``"orb"``).
    :type method: str
    :param transforms: Mapping of transform name to a function applied to
        each grayscale query crop before matching.
    :type transforms: dict[str, Transform]
    :param size: IIIF Image API size requested for each candidate canvas.
    :type size: str
    :param ratio: Lowe's ratio test threshold for descriptor matching.
    :type ratio: float
    :param ransac_reproj_threshold: RANSAC reprojection threshold, in pixels.
    :type ransac_reproj_threshold: float
    :param min_inliers: Minimum RANSAC inliers required to accept a match.
    :type min_inliers: int
    :param extractor_options: Extra keyword arguments for the feature
        extractor, e.g. ``{"nfeatures": 2000}`` to cap SIFT keypoints.
    :type extractor_options: dict | None
    :return: One :class:`TransformEvaluation` per transform, in the given order.
    :rtype: list[TransformEvaluation]
    """
    extractor = get_extractor(method, **(extractor_options or {}))

    candidate_images = load_iiif_images([case.service_id for case in cases], size=size)
    candidate_features = [
        (*extractor.detect_and_compute(image), image.shape)
        for image in candidate_images
    ]
    base_images = [load_image(case.crop_path) for case in cases]

    evaluations = []
    for transform_name, transform_fn in transforms.items():
        case_results: list[MatchResult] = []
        for i, base_image in enumerate(base_images):
            transformed = transform_fn(base_image)
            query_kp, query_desc = extractor.detect_and_compute(transformed)
            candidate_kp, candidate_desc, _shape = candidate_features[i]
            result, _matches, _mask = match_pair(
                query_kp,
                query_desc,
                transformed.shape,
                candidate_kp,
                candidate_desc,
                cases[i].service_id,
                extractor.norm_type,
                ratio,
                ransac_reproj_threshold,
                min_inliers,
            )
            case_results.append(result)

        evaluations.append(
            TransformEvaluation(
                method=method,
                transform=transform_name,
                recall=_mean(
                    [1.0 if r.inliers >= min_inliers else 0.0 for r in case_results]
                ),
                mean_inliers=_mean([r.inliers for r in case_results]),
                mean_inlier_ratio=_mean([r.inlier_ratio for r in case_results]),
                mean_score=_mean([r.score for r in case_results]),
            )
        )

    return evaluations


def evaluate_transforms(
    dataset_path: str | Path,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    transforms: dict[str, Transform] = DEFAULT_TRANSFORMS,
    **kwargs,
) -> dict[str, list[TransformEvaluation]]:
    """Benchmark several methods' robustness to query-side transformations.

    :param dataset_path: Path to a CropCatcher evaluation ``dataset.json`` file.
    :type dataset_path: str | Path
    :param methods: Feature extraction methods to evaluate.
    :type methods: tuple[str, ...]
    :param transforms: Mapping of transform name to a function applied to
        each grayscale query crop before matching.
    :type transforms: dict[str, Transform]
    :param kwargs: Extra keyword arguments forwarded to
        :func:`evaluate_transform_robustness`.
    :return: Mapping of method name to its list of :class:`TransformEvaluation`.
    :rtype: dict[str, list[TransformEvaluation]]
    """
    cases = load_dataset(dataset_path)
    return {
        method: evaluate_transform_robustness(
            cases, method, transforms=transforms, **kwargs
        )
        for method in methods
    }
