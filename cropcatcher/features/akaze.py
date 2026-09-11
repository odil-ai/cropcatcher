#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AKAZE feature extraction for CropCatcher.

This module wraps the OpenCV AKAZE implementation, a binary-descriptor
alternative to SIFT, behind the common extractor interface.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import FeatureExtractor


class AKAZEExtractor(FeatureExtractor):
    """AKAZE: binary descriptor, good compromise between robustness and speed."""

    norm_type = cv2.NORM_HAMMING

    def __init__(self, **kwargs) -> None:
        """Create the extractor.

        :param kwargs: Keyword arguments forwarded verbatim to
            :func:`cv2.AKAZE_create`, e.g. ``nfeatures``.
        :type kwargs: typing.Any
        """
        self._akaze = cv2.AKAZE_create(**kwargs)

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
        return self._akaze.detectAndCompute(image, None)
