#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared pytest fixtures for the CropCatcher test suite.

This module builds the synthetic images, crops and temporary paths reused
across the test modules.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


def _make_textured_image(seed: int, size: int = 600) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((size, size), 255, dtype=np.uint8)

    for _ in range(60):
        shape = rng.integers(0, 3)
        pt1 = tuple(rng.integers(0, size, size=2).tolist())
        pt2 = tuple(rng.integers(0, size, size=2).tolist())
        color = int(rng.integers(0, 180))
        thickness = int(rng.integers(1, 4))
        if shape == 0:
            cv2.rectangle(image, pt1, pt2, color, thickness)
        elif shape == 1:
            radius = int(rng.integers(5, 40))
            cv2.circle(image, pt1, radius, color, thickness)
        else:
            cv2.line(image, pt1, pt2, color, thickness)

    for _ in range(15):
        org = tuple(rng.integers(0, size - 20, size=2).tolist())
        cv2.putText(
            image,
            "CropCatcher",
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            int(rng.integers(0, 120)),
            1,
            cv2.LINE_AA,
        )

    return image


@pytest.fixture()
def image_dir(tmp_path: Path) -> Path:
    """A query dir and a candidates dir (the query is not itself a candidate)."""
    folio = _make_textured_image(seed=1)
    crop = folio[150:350, 200:450].copy()
    negative = _make_textured_image(seed=2)

    query_dir = tmp_path / "query"
    candidates_dir = tmp_path / "candidates"
    query_dir.mkdir()
    candidates_dir.mkdir()

    cv2.imwrite(str(query_dir / "query_crop.png"), crop)
    cv2.imwrite(str(candidates_dir / "folio_142r.png"), folio)
    cv2.imwrite(str(candidates_dir / "unrelated.png"), negative)
    return tmp_path
