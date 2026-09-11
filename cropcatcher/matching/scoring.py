#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Match scoring for CropCatcher.

This module condenses the descriptor and geometric evidence of a match into a
single confidence score in the ``[0, 1]`` range.
"""

from __future__ import annotations


def compute_score(
    num_matches: int,
    num_inliers: int,
    inlier_ratio: float,
    inlier_norm: float = 30.0,
    inlier_ratio_weight: float = 0.6,
    inlier_count_weight: float = 0.4,
) -> float:
    """Combine descriptor matches and RANSAC inliers into a single score.

    The score is weighted mostly toward geometrically consistent inliers
    rather than the raw number of descriptor matches, so that a large but
    incoherent set of matches cannot pass for a match.

    :param num_matches: Descriptor matches surviving the ratio test.
    :type num_matches: int
    :param num_inliers: Matches consistent with the estimated homography.
    :type num_inliers: int
    :param inlier_ratio: ``num_inliers / num_matches``.
    :type inlier_ratio: float
    :param inlier_norm: Inlier count considered saturated, i.e. scoring 1.0
        on the count component.
    :type inlier_norm: float
    :param inlier_ratio_weight: Weight given to the inlier ratio.
    :type inlier_ratio_weight: float
    :param inlier_count_weight: Weight given to the saturated inlier count.
    :type inlier_count_weight: float
    :return: Confidence score in the ``[0, 1]`` range.
    :rtype: float
    """
    if num_matches == 0:
        return 0.0

    inlier_count_score = min(num_inliers / inlier_norm, 1.0)
    return float(
        inlier_ratio_weight * inlier_ratio + inlier_count_weight * inlier_count_score
    )
