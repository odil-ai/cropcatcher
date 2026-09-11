#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CropCatcher: locate a visual query image inside candidate images.

This package exposes the public entry points of the library: the
:class:`Matcher` for one-off searches, the :class:`Index` for reusable
corpora, and the structured result types returned by both.
"""

from __future__ import annotations

from .index import Index
from .matcher import Matcher
from .result import MatchResult, SearchResults
from .visualization import MatchDebugInfo

__all__ = ["Matcher", "Index", "MatchResult", "SearchResults", "MatchDebugInfo"]
