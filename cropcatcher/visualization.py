#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization helpers for CropCatcher.

This module renders keypoints, descriptor matches and the localized query
region so that a match, or the absence of one, can be inspected visually.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .result import MatchResult


@dataclass
class MatchDebugInfo:
    """Everything needed to visualize why a candidate matched, or did not.

    :param result: The match result being explained.
    :type result: MatchResult
    :param query_image: The query image, as matched.
    :type query_image: numpy.ndarray
    :param candidate_image: The candidate image, as matched.
    :type candidate_image: numpy.ndarray
    :param query_keypoints: Keypoints detected in the query image.
    :type query_keypoints: list[cv2.KeyPoint]
    :param candidate_keypoints: Keypoints detected in the candidate image.
    :type candidate_keypoints: list[cv2.KeyPoint]
    :param matches: Descriptor matches surviving the ratio test.
    :type matches: list[cv2.DMatch]
    :param mask: RANSAC inlier mask over ``matches``, or ``None`` when no
        homography could be estimated.
    :type mask: numpy.ndarray | None
    """

    result: MatchResult
    query_image: np.ndarray
    candidate_image: np.ndarray
    query_keypoints: list[cv2.KeyPoint]
    candidate_keypoints: list[cv2.KeyPoint]
    matches: list[cv2.DMatch]
    mask: np.ndarray | None


def draw_keypoints(image: np.ndarray, keypoints: list[cv2.KeyPoint]) -> np.ndarray:
    """Draw detected keypoints with their scale and orientation.

    :param image: Image to draw on.
    :type image: numpy.ndarray
    :param keypoints: Keypoints to render.
    :type keypoints: list[cv2.KeyPoint]
    :return: A colour image with the keypoints drawn.
    :rtype: numpy.ndarray
    """
    return cv2.drawKeypoints(
        image, keypoints, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    )


def draw_matches(debug: MatchDebugInfo, max_matches: int | None = 100) -> np.ndarray:
    """Draw query and candidate side by side, connecting matched keypoints.

    RANSAC inliers are drawn in green and outliers in red, so it is easy to
    see at a glance how much of the match is geometrically consistent.

    :param debug: Match data to render.
    :type debug: MatchDebugInfo
    :param max_matches: Draw only the ``max_matches`` closest matches, to
        keep the figure readable. ``None`` draws all of them.
    :type max_matches: int | None
    :return: The rendered side-by-side image.
    :rtype: numpy.ndarray
    """
    matches = debug.matches
    mask_flat = (
        debug.mask.ravel().tolist() if debug.mask is not None else [1] * len(matches)
    )

    order = sorted(range(len(matches)), key=lambda i: matches[i].distance)
    if max_matches is not None:
        order = order[:max_matches]

    kept_matches = [matches[i] for i in order]
    inlier_mask = [mask_flat[i] for i in order]
    outlier_mask = [1 - v for v in inlier_mask]

    common_kwargs = {
        "img1": debug.query_image,
        "keypoints1": debug.query_keypoints,
        "img2": debug.candidate_image,
        "keypoints2": debug.candidate_keypoints,
        "matches1to2": kept_matches,
        "singlePointColor": (120, 120, 120),
        "flags": cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    }

    canvas = cv2.drawMatches(
        outImg=None, matchColor=(0, 0, 220), matchesMask=outlier_mask, **common_kwargs
    )
    canvas = cv2.drawMatches(
        outImg=canvas,
        matchColor=(0, 200, 0),
        matchesMask=inlier_mask,
        flags=common_kwargs["flags"] | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG,
        **{k: v for k, v in common_kwargs.items() if k != "flags"},
    )
    return canvas


def draw_localization(
    candidate_image: np.ndarray,
    query_shape: tuple[int, int],
    homography: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    """Draw the query outline projected into the candidate image.

    :param candidate_image: Image to draw the outline on.
    :type candidate_image: numpy.ndarray
    :param query_shape: Shape of the query image as ``(height, width)``.
    :type query_shape: tuple[int, int]
    :param homography: The 3x3 projective transform mapping query
        coordinates onto candidate coordinates.
    :type homography: numpy.ndarray
    :param color: Outline colour, as a BGR triple.
    :type color: tuple[int, int, int]
    :param thickness: Outline thickness in pixels.
    :type thickness: int
    :return: A colour copy of the candidate with the outline drawn.
    :rtype: numpy.ndarray
    """
    height, width = query_shape
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(
        -1, 1, 2
    )
    projected = cv2.perspectiveTransform(corners, homography)

    canvas = (
        cv2.cvtColor(candidate_image, cv2.COLOR_GRAY2BGR)
        if candidate_image.ndim == 2
        else candidate_image.copy()
    )
    cv2.polylines(
        canvas, [np.int32(projected)], isClosed=True, color=color, thickness=thickness
    )
    return canvas


__all__ = ["MatchDebugInfo", "draw_keypoints", "draw_matches", "draw_localization"]
