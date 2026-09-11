#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Geometric verification for CropCatcher.

This module estimates the homography relating a query to a candidate using
RANSAC, and projects the query bounding box into the candidate's coordinate
system.
"""

from __future__ import annotations

import cv2
import numpy as np


def estimate_homography(
    query_keypoints: list[cv2.KeyPoint],
    candidate_keypoints: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    ransac_reproj_threshold: float = 5.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate a homography mapping query points onto candidate points.

    Fitting a homography requires at least four point correspondences; below
    that, no estimate is attempted.

    :param query_keypoints: Keypoints detected in the query image.
    :type query_keypoints: list[cv2.KeyPoint]
    :param candidate_keypoints: Keypoints detected in the candidate image.
    :type candidate_keypoints: list[cv2.KeyPoint]
    :param matches: Descriptor matches linking the two keypoint sets.
    :type matches: list[cv2.DMatch]
    :param ransac_reproj_threshold: Reprojection error, in pixels, tolerated
        when deciding whether a match is an inlier.
    :type ransac_reproj_threshold: float
    :return: The 3x3 homography and the per-match inlier mask, or
        ``(None, None)`` when there are too few matches.
    :rtype: tuple[numpy.ndarray | None, numpy.ndarray | None]
    """
    if len(matches) < 4:
        return None, None

    src_pts = np.float32([query_keypoints[m.queryIdx].pt for m in matches]).reshape(
        -1, 1, 2
    )
    dst_pts = np.float32([candidate_keypoints[m.trainIdx].pt for m in matches]).reshape(
        -1, 1, 2
    )

    homography, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold
    )
    return homography, mask


def project_bbox(
    query_shape: tuple[int, int], homography: np.ndarray
) -> tuple[int, int, int, int]:
    """Project the query image corners into candidate space.

    :param query_shape: Shape of the query image as ``(height, width)``.
    :type query_shape: tuple[int, int]
    :param homography: The 3x3 projective transform to apply.
    :type homography: numpy.ndarray
    :return: Axis-aligned bounding box ``(x0, y0, x1, y1)`` of the projected
        corners, in candidate coordinates.
    :rtype: tuple[int, int, int, int]
    """
    height, width = query_shape
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(
        -1, 1, 2
    )
    projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    x_min, y_min = projected.min(axis=0)
    x_max, y_max = projected.max(axis=0)
    return (int(x_min), int(y_min), int(x_max), int(y_max))
