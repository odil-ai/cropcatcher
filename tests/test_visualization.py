#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for the visualization helpers."""

from __future__ import annotations

from pathlib import Path

from cropcatcher import Matcher
from cropcatcher.visualization import draw_keypoints, draw_localization, draw_matches


def test_explain_and_draw_matches(image_dir: Path) -> None:
    matcher = Matcher(method="sift")
    debug = matcher.explain(
        query=image_dir / "query" / "query_crop.png",
        candidate=image_dir / "candidates" / "folio_142r.png",
    )

    assert debug.result.inliers >= 8
    assert debug.mask is not None

    canvas = draw_matches(debug)
    assert canvas.ndim == 3
    assert canvas.shape[0] > 0 and canvas.shape[1] > 0


def test_draw_keypoints(image_dir: Path) -> None:
    matcher = Matcher(method="sift")
    debug = matcher.explain(
        query=image_dir / "query" / "query_crop.png",
        candidate=image_dir / "candidates" / "folio_142r.png",
    )

    canvas = draw_keypoints(debug.query_image, debug.query_keypoints)
    assert canvas.shape[:2] == debug.query_image.shape[:2]


def test_draw_localization(image_dir: Path) -> None:
    matcher = Matcher(method="sift")
    debug = matcher.explain(
        query=image_dir / "query" / "query_crop.png",
        candidate=image_dir / "candidates" / "folio_142r.png",
    )
    assert debug.result.homography is not None

    canvas = draw_localization(
        debug.candidate_image, debug.query_image.shape, debug.result.homography
    )
    assert canvas.ndim == 3
    assert canvas.shape[:2] == debug.candidate_image.shape[:2]
