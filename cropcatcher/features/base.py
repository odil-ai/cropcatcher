#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Common interface for CropCatcher feature extractors.

This module defines the abstract base class every extractor implements, so
that detection methods can be swapped and benchmarked without changing the
rest of the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class FeatureExtractor(ABC):
    """Common interface for local feature detectors/descriptors.

    Every method (SIFT, AKAZE, ORB, ...) implements this interface so the
    rest of the matching pipeline stays method-agnostic.
    """

    norm_type: int

    @abstractmethod
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
        raise NotImplementedError
