#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Descriptor matching and filtering for CropCatcher.

This module pairs query descriptors with candidate descriptors and discards
ambiguous pairs using Lowe's ratio test and an optional mutual consistency
check.
"""

from __future__ import annotations

import cv2
import numpy as np

_FLANN_INDEX_KDTREE = 1


def _knn_match(
    desc_query: np.ndarray, desc_train: np.ndarray, norm_type: int, k: int
) -> list[list[cv2.DMatch]]:
    """Find the ``k`` nearest ``desc_train`` entries for each query entry.

    Uses FLANN for L2 (SIFT) descriptors and a brute-force Hamming matcher
    for binary (ORB/AKAZE) descriptors.

    :param desc_query: Descriptors to find neighbours for.
    :type desc_query: numpy.ndarray
    :param desc_train: Descriptors to search among.
    :type desc_train: numpy.ndarray
    :param norm_type: OpenCV norm identifier, e.g. ``cv2.NORM_L2``.
    :type norm_type: int
    :param k: Number of neighbours to return per query descriptor.
    :type k: int
    :return: For each query descriptor, its ``k`` nearest matches.
    :rtype: list[list[cv2.DMatch]]
    """
    if norm_type == cv2.NORM_L2:
        index_params = {"algorithm": _FLANN_INDEX_KDTREE, "trees": 5}
        search_params = {"checks": 50}
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        return matcher.knnMatch(
            desc_query.astype(np.float32), desc_train.astype(np.float32), k=k
        )

    matcher = cv2.BFMatcher(norm_type)
    return matcher.knnMatch(desc_query, desc_train, k=k)


def match_descriptors(
    desc1: np.ndarray | None,
    desc2: np.ndarray | None,
    norm_type: int,
    ratio: float = 0.75,
    mutual_check: bool = False,
) -> list[cv2.DMatch]:
    """Match descriptors and keep only those passing Lowe's ratio test.

    :param desc1: Query descriptors, or ``None`` when none were extracted.
    :type desc1: numpy.ndarray | None
    :param desc2: Candidate descriptors, or ``None`` when none were
        extracted.
    :type desc2: numpy.ndarray | None
    :param norm_type: OpenCV norm identifier matching the descriptor type.
    :type norm_type: int
    :param ratio: Lowe's ratio threshold; lower keeps fewer, cleaner matches.
    :type ratio: float
    :param mutual_check: Additionally require matches to be symmetric, so
        that the candidate descriptor's own nearest neighbour is the query
        descriptor it was matched from. This discards one-sided matches at
        the cost of a second matching pass in the reverse direction.
    :type mutual_check: bool
    :return: The surviving descriptor matches.
    :rtype: list[cv2.DMatch]
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []

    knn_matches = _knn_match(desc1, desc2, norm_type, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            good_matches.append(best)

    if not mutual_check or not good_matches:
        return good_matches

    # For each desc2 entry, which desc1 entry is its own nearest neighbour?
    reverse = _knn_match(desc2, desc1, norm_type, k=1)
    nearest_back = {
        index: pair[0].trainIdx for index, pair in enumerate(reverse) if pair
    }
    return [m for m in good_matches if nearest_back.get(m.trainIdx) == m.queryIdx]
