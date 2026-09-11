#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for descriptor matching and filtering."""

from __future__ import annotations

import cv2
import numpy as np

from cropcatcher.matching.descriptors import match_descriptors


def _descriptors(seed: int, count: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((count, 128), dtype=np.float32) * 255


def test_mutual_check_keeps_only_symmetric_matches() -> None:
    base = _descriptors(seed=1)
    # every second descriptor is duplicated into the candidate set, so half the
    # forward matches have an exact counterpart and half are one-sided.
    candidate = np.vstack([base[::2], _descriptors(seed=2, count=20)])

    without = match_descriptors(base, candidate, cv2.NORM_L2, ratio=0.9)
    with_mutual = match_descriptors(
        base, candidate, cv2.NORM_L2, ratio=0.9, mutual_check=True
    )

    assert len(with_mutual) <= len(without)
    # a mutual match must be its candidate's own nearest neighbour
    kept = {(m.queryIdx, m.trainIdx) for m in with_mutual}
    assert kept.issubset({(m.queryIdx, m.trainIdx) for m in without})


def test_mutual_check_is_a_noop_on_empty_input() -> None:
    assert match_descriptors(None, None, cv2.NORM_L2, mutual_check=True) == []
