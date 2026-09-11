#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for the high-level matching entry points."""

from __future__ import annotations

from pathlib import Path

import pytest

from cropcatcher import Matcher, SearchResults

METHODS = ["sift", "akaze", "orb", "star_brief"]


@pytest.mark.parametrize("method", METHODS)
def test_search_ranks_true_source_first(image_dir: Path, method: str) -> None:
    matcher = Matcher(method=method)

    results = matcher.search(
        query=image_dir / "query" / "query_crop.png",
        images=image_dir / "candidates",
    )

    assert isinstance(results, SearchResults)
    assert len(results) == 2

    best = results.best
    assert best is not None
    assert Path(best.source).name == "folio_142r.png"
    assert best.inliers >= 8
    assert best.inlier_ratio > 0.5
    assert best.bbox is not None

    negative = next(r for r in results if Path(r.source).name == "unrelated.png")
    assert best.score > negative.score


@pytest.mark.parametrize("method", METHODS)
def test_bbox_localizes_crop_region(image_dir: Path, method: str) -> None:
    matcher = Matcher(method=method)

    results = matcher.search(
        query=image_dir / "query" / "query_crop.png",
        images=[image_dir / "candidates" / "folio_142r.png"],
    )

    best = results.best
    assert best is not None
    x_min, y_min, x_max, y_max = best.bbox

    # crop = folio[150:350, 200:450] -> x in [200, 450], y in [150, 350]
    assert abs(x_min - 200) < 25
    assert abs(y_min - 150) < 25
    assert abs(x_max - 450) < 25
    assert abs(y_max - 350) < 25


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError):
        Matcher(method="not-a-real-method")
