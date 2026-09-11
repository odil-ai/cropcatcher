#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SURF feature extraction for CropCatcher.

This module wraps the OpenCV SURF implementation behind the common extractor
interface. SURF is patented and excluded from prebuilt OpenCV wheels, so it
requires a custom build with non-free algorithms enabled.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import FeatureExtractor


class SURFExtractor(FeatureExtractor):
    """SURF: optional/experimental, patented.

    Prebuilt opencv-contrib-python(-headless) wheels from PyPI ship with SURF
    excluded even though the `cv2.xfeatures2d.SURF_create` binding is present
    — using it requires a custom OpenCV build compiled with
    `-DOPENCV_ENABLE_NONFREE=ON`. Instantiating this class raises a clear
    error unless such a build is used.
    """

    norm_type = cv2.NORM_L2

    def __init__(self, **kwargs) -> None:
        """Create the extractor, if the OpenCV build allows it.

        :param kwargs: Keyword arguments forwarded verbatim to
            :func:`cv2.xfeatures2d.SURF_create`.
        :type kwargs: typing.Any
        :raises ImportError: If ``cv2.xfeatures2d`` is unavailable, i.e. the
            installed OpenCV is not a contrib build.
        :raises RuntimeError: If the OpenCV build was compiled without
            non-free algorithms, which is the case for the PyPI wheels.
        """
        if not hasattr(cv2, "xfeatures2d"):
            raise ImportError(
                "SURF requires opencv-contrib-python(-headless), which provides "
                "cv2.xfeatures2d. Install it instead of opencv-python-headless."
            )
        try:
            self._surf = cv2.xfeatures2d.SURF_create(**kwargs)
        except cv2.error as exc:
            raise RuntimeError(
                "SURF is patented and disabled in standard OpenCV builds. It "
                "requires a custom OpenCV build compiled with "
                "-DOPENCV_ENABLE_NONFREE=ON; the prebuilt PyPI wheels "
                "(opencv-contrib-python-headless) do not enable it."
            ) from exc

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
        return self._surf.detectAndCompute(image, None)
