#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SIFT feature extraction for CropCatcher.

This module wraps the OpenCV SIFT implementation, the reference method of the
package, behind the common extractor interface.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import FeatureExtractor


class SIFTExtractor(FeatureExtractor):
    """SIFT: primary/reference method, robust to scale and rotation."""

    norm_type = cv2.NORM_L2

    def __init__(self, **kwargs) -> None:
        """Create the extractor.

        :param kwargs: Keyword arguments forwarded verbatim to
            :func:`cv2.SIFT_create`, e.g. ``nfeatures``.
        :type kwargs: typing.Any
        """
        self._sift = cv2.SIFT_create(**kwargs)

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
        return self._sift.detectAndCompute(image, None)
