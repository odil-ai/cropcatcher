#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local image loading for CropCatcher.

This module reads images from disk in grayscale and expands paths,
directories and lists of paths into a flat list of candidate files.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as grayscale.

    :param path: Path to the image file.
    :type path: str | pathlib.Path
    :return: The decoded image as a 2-D grayscale array.
    :rtype: numpy.ndarray
    :raises FileNotFoundError: If the file is missing or cannot be decoded.
    """
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def resolve_image_paths(
    images: str | Path | list[str | Path],
) -> list[Path]:
    """Resolve a directory, a single path or a list of paths to image paths.

    Directories are expanded to their image files, sorted by name; other
    inputs are passed through unchanged.

    :param images: A directory, a single image path, or a list of paths.
    :type images: str | pathlib.Path | list[str | pathlib.Path]
    :return: Flat list of candidate image paths.
    :rtype: list[pathlib.Path]
    """
    if isinstance(images, (str, Path)):
        path = Path(images)
        if path.is_dir():
            return sorted(
                p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
            )
        return [path]

    return [Path(p) for p in images]
