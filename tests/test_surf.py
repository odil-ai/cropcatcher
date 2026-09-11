#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for the SURF extractor's behaviour on builds without non-free algorithms."""

from __future__ import annotations

import pytest

from cropcatcher import Matcher


def test_surf_raises_clear_error_without_nonfree_build() -> None:
    with pytest.raises(RuntimeError, match="OPENCV_ENABLE_NONFREE"):
        Matcher(method="surf")
