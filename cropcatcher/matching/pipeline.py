#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end pair matching for CropCatcher.

This module chains feature extraction, descriptor matching, RANSAC homography
estimation and scoring into the single operation that compares one query
image with one candidate image.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..result import MatchResult
from .descriptors import match_descriptors
from .ransac import estimate_homography, project_bbox
from .scoring import compute_score


def match_pair(
    query_kp: list[cv2.KeyPoint],
    query_desc: np.ndarray | None,
    query_shape: tuple[int, int],
    candidate_kp: list[cv2.KeyPoint],
    candidate_desc: np.ndarray | None,
    source: str,
    norm_type: int,
    ratio: float,
    ransac_reproj_threshold: float,
    min_inliers: int,
    mutual_check: bool = False,
) -> tuple[MatchResult, list[cv2.DMatch], np.ndarray | None]:
    """Match precomputed query and candidate features into a result.

    Shared by :class:`~cropcatcher.matcher.Matcher`, which computes features
    on the fly, and :class:`~cropcatcher.index.Index`, which stores them, so
    both stay consistent without duplicating the matching, RANSAC and
    scoring steps.

    :param query_kp: Keypoints detected in the query image.
    :type query_kp: list[cv2.KeyPoint]
    :param query_desc: Descriptors of the query keypoints, or ``None``.
    :type query_desc: numpy.ndarray | None
    :param query_shape: Shape of the query image as ``(height, width)``.
    :type query_shape: tuple[int, int]
    :param candidate_kp: Keypoints detected in the candidate image.
    :type candidate_kp: list[cv2.KeyPoint]
    :param candidate_desc: Descriptors of the candidate keypoints, or
        ``None``.
    :type candidate_desc: numpy.ndarray | None
    :param source: Identifier recorded on the result: a file path, or a IIIF
        Image API service id.
    :type source: str
    :param norm_type: OpenCV norm identifier matching the descriptor type.
    :type norm_type: int
    :param ratio: Lowe's ratio threshold applied when filtering matches.
    :type ratio: float
    :param ransac_reproj_threshold: Reprojection error, in pixels, tolerated
        when deciding whether a match is an inlier.
    :type ransac_reproj_threshold: float
    :param min_inliers: Minimum inlier count below which no bounding box is
        reported.
    :type min_inliers: int
    :param mutual_check: Require descriptor matches to be symmetric.
    :type mutual_check: bool
    :return: The match result, the filtered descriptor matches and the
        RANSAC inlier mask. The last two are needed for visualization.
    :rtype: tuple[MatchResult, list[cv2.DMatch], numpy.ndarray | None]
    """
    good_matches = match_descriptors(
        query_desc, candidate_desc, norm_type, ratio=ratio, mutual_check=mutual_check
    )

    homography, mask = estimate_homography(
        query_kp, candidate_kp, good_matches, ransac_reproj_threshold
    )

    num_matches = len(good_matches)
    num_inliers = int(mask.sum()) if mask is not None else 0
    inlier_ratio = num_inliers / num_matches if num_matches else 0.0
    score = compute_score(num_matches, num_inliers, inlier_ratio)

    bbox = None
    if homography is not None and num_inliers >= min_inliers:
        bbox = project_bbox(query_shape, homography)

    result = MatchResult(
        source=source,
        score=score,
        matches=num_matches,
        inliers=num_inliers,
        inlier_ratio=inlier_ratio,
        bbox=bbox,
        homography=homography,
    )
    return result, good_matches, mask
