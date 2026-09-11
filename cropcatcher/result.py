#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Structured result types for CropCatcher.

This module defines :class:`MatchResult`, the per-candidate outcome of a
match, and :class:`SearchResults`, the ranked collection returned by the
search entry points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MatchResult:
    """Outcome of matching a query image against a single candidate.

    :param source: Identifier of the candidate: a file path for local images,
        or the IIIF Image API service id for IIIF entry points.
    :type source: str
    :param score: Confidence in the ``[0, 1]`` range, combining the inlier
        ratio and the inlier count.
    :type score: float
    :param matches: Descriptor matches surviving the ratio test, before any
        geometric verification.
    :type matches: int
    :param inliers: Matches consistent with the estimated homography under
        RANSAC. This is the primary evidence of a match.
    :type inliers: int
    :param inlier_ratio: ``inliers / matches``, i.e. how clean the match is
        independently of its size.
    :type inlier_ratio: float
    :param bbox: Location of the query inside the candidate as
        ``(x0, y0, x1, y1)``, or ``None`` when too few inliers were found.
    :type bbox: tuple[int, int, int, int] | None
    :param homography: The 3x3 projective transform mapping the query onto
        the candidate, or ``None`` under the same condition.
    :type homography: numpy.ndarray | None
    """

    source: str
    score: float
    matches: int
    inliers: int
    inlier_ratio: float
    bbox: tuple[int, int, int, int] | None = None
    homography: np.ndarray | None = None

    def __repr__(self) -> str:
        """Return a compact, rounded representation of the result.

        :return: Human-readable summary of the match.
        :rtype: str
        """
        return (
            f"MatchResult(source={self.source!r}, score={self.score:.3f}, "
            f"matches={self.matches}, inliers={self.inliers}, "
            f"inlier_ratio={self.inlier_ratio:.3f}, bbox={self.bbox})"
        )


@dataclass
class SearchResults:
    """Ranked collection of :class:`MatchResult` objects.

    Iterable, indexable and sized, with results ordered by descending score.

    :param results: Candidate results, already sorted best-first.
    :type results: list[MatchResult]
    """

    results: list[MatchResult] = field(default_factory=list)

    @property
    def best(self) -> MatchResult | None:
        """Return the highest-scoring result.

        :return: The best candidate, or ``None`` when no candidate was scored.
        :rtype: MatchResult | None
        """
        return self.results[0] if self.results else None

    def top(self, k: int) -> list[MatchResult]:
        """Return the ``k`` highest-scoring results.

        :param k: Maximum number of results to return.
        :type k: int
        :return: At most ``k`` results, ordered by descending score.
        :rtype: list[MatchResult]
        """
        return self.results[:k]

    def __iter__(self):
        """Iterate over the results, best-first.

        :return: Iterator over the ranked results.
        :rtype: collections.abc.Iterator[MatchResult]
        """
        return iter(self.results)

    def __len__(self) -> int:
        """Return the number of scored candidates.

        :return: Number of results held.
        :rtype: int
        """
        return len(self.results)

    def __getitem__(self, index: int) -> MatchResult:
        """Return the result at rank ``index``.

        :param index: Zero-based rank of the result.
        :type index: int
        :return: The result at that rank.
        :rtype: MatchResult
        """
        return self.results[index]
