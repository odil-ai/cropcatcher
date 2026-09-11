#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ORB feature extraction for CropCatcher.

This module wraps the OpenCV ORB implementation, the fastest binary
alternative available, behind the common extractor interface.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import FeatureExtractor


class ORBExtractor(FeatureExtractor):
    """ORB: fast binary alternative to SIFT/AKAZE."""

    norm_type = cv2.NORM_HAMMING

    def __init__(self, nfeatures: int = 2000, **kwargs) -> None:
        """Create the extractor.

        :param nfeatures: Maximum number of keypoints to retain.
        :type nfeatures: int
        :param kwargs: Keyword arguments forwarded verbatim to
            :func:`cv2.ORB_create`.
        :type kwargs: typing.Any
        """
        self._orb = cv2.ORB_create(nfeatures=nfeatures, **kwargs)

    def detect_and_compute(
        self, image: np.ndarray
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        """Detect keypoints and compute descriptors for a grayscale image.

        :param image: Grayscale image to describe.
        :type image: numpy.ndarray
        :return: The detected keypoints and their descriptors; the
            descriptors are ``None`` when no keypoint was found.
        :rtype: tuple[list[cv2.KeyPoint], numpy.ndarray | None]
        """
        return self._orb.detectAndCompute(image, None)
