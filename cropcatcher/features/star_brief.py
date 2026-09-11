#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAR/CenSurE and BRIEF feature extraction for CropCatcher.

This module pairs the OpenCV STAR detector with BRIEF descriptors behind the
common extractor interface, as an unpatented fast alternative.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import FeatureExtractor


class StarBriefExtractor(FeatureExtractor):
    """STAR/CenSurE detector + BRIEF descriptor: fast, binary, not patented."""

    norm_type = cv2.NORM_HAMMING

    def __init__(self, response_threshold: int = 10, **kwargs) -> None:
        """Create the detector/descriptor pair.

        :param response_threshold: STAR detector response threshold. OpenCV's
            own default (30) is conservative and can leave small crops with
            too few keypoints to reach a reliable homography; 10 finds
            noticeably more without flooding matching with noise.
        :type response_threshold: int
        :param kwargs: Keyword arguments forwarded verbatim to
            :func:`cv2.xfeatures2d.StarDetector_create`.
        :type kwargs: typing.Any
        :raises ImportError: If ``cv2.xfeatures2d`` is unavailable, i.e. the
            installed OpenCV is not a contrib build.
        """
        if not hasattr(cv2, "xfeatures2d"):
            raise ImportError(
                "STAR/BRIEF requires opencv-contrib-python(-headless), which "
                "provides cv2.xfeatures2d. Install it instead of "
                "opencv-python-headless."
            )
        # OpenCV's default (30) is tuned conservatively and can leave small
        # crops with too few keypoints to reach a reliable homography; 10
        # finds noticeably more without flooding matching with noise.
        self._detector = cv2.xfeatures2d.StarDetector_create(
            responseThreshold=response_threshold, **kwargs
        )
        self._descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()

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
        keypoints = self._detector.detect(image, None)
        keypoints, descriptors = self._descriptor.compute(image, keypoints)
        return keypoints, descriptors
