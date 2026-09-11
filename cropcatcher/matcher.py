#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
High-level matching entry points for CropCatcher.

This module provides the :class:`Matcher` facade, which runs the full
detection, description, matching and geometric verification pipeline against
local images, directories of images or IIIF manifests.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .features import FeatureExtractor, get_extractor
from .matching.pipeline import match_pair
from .result import MatchResult, SearchResults
from .sources.iiif import DEFAULT_CONCURRENCY, load_iiif_image, resolve_manifest
from .sources.images import load_image, resolve_image_paths
from .visualization import MatchDebugInfo


class Matcher:
    """Determine whether a query image is equal to, or contained in, candidates.

    :param method: Local feature extraction method: ``"sift"``, ``"akaze"``,
        ``"orb"``, ``"star_brief"`` or ``"surf"``.
    :type method: str
    :param ratio: Lowe's ratio threshold applied when filtering descriptor
        matches. Lower keeps fewer, cleaner matches.
    :type ratio: float
    :param ransac_reproj_threshold: Reprojection error, in pixels, tolerated
        when deciding whether a match is a RANSAC inlier. Raise it for
        distorted or low-resolution sources.
    :type ransac_reproj_threshold: float
    :param min_inliers: Minimum inlier count below which no bounding box or
        homography is reported.
    :type min_inliers: int
    :param extractor_options: Keyword arguments forwarded to the underlying
        OpenCV detector, e.g. ``{"nfeatures": 4000}``.
    :type extractor_options: dict | None
    :param mutual_check: Require descriptor matches to be symmetric. Costs a
        second matching pass but greatly improves precision.
    :type mutual_check: bool
    """

    def __init__(
        self,
        method: str = "sift",
        ratio: float = 0.75,
        ransac_reproj_threshold: float = 5.0,
        min_inliers: int = 8,
        extractor_options: dict | None = None,
        mutual_check: bool = False,
    ) -> None:
        """Configure the matching pipeline."""
        self.method = method
        self.extractor_options = dict(extractor_options or {})
        self.extractor = get_extractor(method, **self.extractor_options)
        self.ratio = ratio
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.min_inliers = min_inliers
        self.mutual_check = mutual_check
        self._thread_local = threading.local()

    def _thread_extractor(self) -> FeatureExtractor:
        """Return this thread's own extractor.

        OpenCV Feature2D instances are not documented as thread-safe for
        concurrent `detectAndCompute` calls, and constructing one is cheap,
        so each worker thread gets its own rather than sharing `self.extractor`.
        """
        extractor = getattr(self._thread_local, "extractor", None)
        if extractor is None:
            extractor = get_extractor(self.method, **self.extractor_options)
            self._thread_local.extractor = extractor
        return extractor

    def search(
        self,
        query: str | Path,
        images: str | Path | list[str | Path],
    ) -> SearchResults:
        """Search for a query image among local candidate images.

        :param query: Path to the query image.
        :type query: str | pathlib.Path
        :param images: A directory of images, a single image path, or a list
            of image paths.
        :type images: str | pathlib.Path | list[str | pathlib.Path]
        :return: Candidates ranked by descending score.
        :rtype: SearchResults
        """
        query_image, query_kp, query_desc = self._detect_query(query)

        results = []
        for candidate_path in resolve_image_paths(images):
            candidate_image = load_image(candidate_path)
            *_, result = self._compute_match(
                query_kp,
                query_desc,
                query_image.shape,
                candidate_image,
                str(candidate_path),
            )
            results.append(result)

        results.sort(key=lambda r: r.score, reverse=True)
        return SearchResults(results)

    def search_iiif(
        self,
        query: str | Path,
        manifests: str | list[str],
        size: str = "!1024,1024",
        max_workers: int = DEFAULT_CONCURRENCY,
        max_canvases: int | None = None,
    ) -> SearchResults:
        """Search for a query image among the canvases of IIIF manifests.

        Each canvas is downloaded as a IIIF Image API derivative, described
        and matched in the same work unit, so network waits overlap with
        matching and only ``max_workers`` decoded images are held in memory.

        :param query: Path to the query image.
        :type query: str | pathlib.Path
        :param manifests: One manifest URL, or a list of manifest URLs.
        :type manifests: str | list[str]
        :param size: IIIF Image API size parameter used for the derivatives,
            e.g. ``"!1024,1024"`` to fit within a 1024-pixel box.
        :type size: str
        :param max_workers: Number of worker threads used to download and
            match canvases concurrently.
        :type max_workers: int
        :param max_canvases: Stop after this many canvases per manifest.
            ``None`` searches every canvas; truncating risks excluding the
            true source.
        :type max_canvases: int | None
        :return: Candidates ranked by descending score, sourced by IIIF Image
            API service id.
        :rtype: SearchResults
        """
        if isinstance(manifests, str):
            manifests = [manifests]

        query_image, query_kp, query_desc = self._detect_query(query)

        resources = [
            resource
            for manifest_url in manifests
            for resource in resolve_manifest(manifest_url, max_canvases=max_canvases)
        ]

        def download_and_match(resource) -> MatchResult:
            """Download, describe and match one candidate.

            Doing all three per candidate (rather than downloading everything
            first) overlaps network waits with matching, spreads matching over
            several cores — OpenCV releases the GIL — and keeps only
            `max_workers` decoded images in memory instead of the whole manifest.
            """
            candidate_image = load_iiif_image(resource.service_id, size=size)
            *_, result = self._compute_match(
                query_kp,
                query_desc,
                query_image.shape,
                candidate_image,
                resource.service_id,
            )
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(download_and_match, resources))

        results.sort(key=lambda r: r.score, reverse=True)
        return SearchResults(results)

    def explain(self, query: str | Path, candidate: str | Path) -> MatchDebugInfo:
        """Match a single query/candidate pair and return visualization data.

        :param query: Path to the query image.
        :type query: str | pathlib.Path
        :param candidate: Path to the candidate image.
        :type candidate: str | pathlib.Path
        :return: The match result together with the images, keypoints and
            matches needed to render it.
        :rtype: cropcatcher.visualization.MatchDebugInfo
        """
        query_image, query_kp, query_desc = self._detect_query(query)
        candidate_image = load_image(candidate)
        return self._explain(
            query_image, query_kp, query_desc, candidate_image, str(candidate)
        )

    def explain_iiif(
        self, query: str | Path, service_id: str, size: str = "!1024,1024"
    ) -> MatchDebugInfo:
        """Match a query against one IIIF Image API service, for visualization.

        :param query: Path to the query image.
        :type query: str | pathlib.Path
        :param service_id: IIIF Image API service identifier of the
            candidate.
        :type service_id: str
        :param size: IIIF Image API size parameter for the derivative.
        :type size: str
        :return: The match result together with the images, keypoints and
            matches needed to render it.
        :rtype: cropcatcher.visualization.MatchDebugInfo
        """
        query_image, query_kp, query_desc = self._detect_query(query)
        candidate_image = load_iiif_image(service_id, size=size)
        return self._explain(
            query_image, query_kp, query_desc, candidate_image, service_id
        )

    def _explain(
        self,
        query_image: np.ndarray,
        query_kp: list[cv2.KeyPoint],
        query_desc: np.ndarray | None,
        candidate_image: np.ndarray,
        source: str,
    ) -> MatchDebugInfo:
        candidate_kp, good_matches, mask, result = self._compute_match(
            query_kp, query_desc, query_image.shape, candidate_image, source
        )
        return MatchDebugInfo(
            result=result,
            query_image=query_image,
            candidate_image=candidate_image,
            query_keypoints=query_kp,
            candidate_keypoints=candidate_kp,
            matches=good_matches,
            mask=mask,
        )

    def _detect_query(
        self, query: str | Path
    ) -> tuple[np.ndarray, list[cv2.KeyPoint], np.ndarray | None]:
        query_image = load_image(query)
        query_kp, query_desc = self.extractor.detect_and_compute(query_image)
        return query_image, query_kp, query_desc

    def _compute_match(
        self,
        query_kp: list[cv2.KeyPoint],
        query_desc: np.ndarray | None,
        query_shape: tuple[int, int],
        candidate_image: np.ndarray,
        source: str,
    ) -> tuple[list[cv2.KeyPoint], list[cv2.DMatch], np.ndarray | None, MatchResult]:
        extractor = self._thread_extractor()
        candidate_kp, candidate_desc = extractor.detect_and_compute(candidate_image)
        result, good_matches, mask = match_pair(
            query_kp,
            query_desc,
            query_shape,
            candidate_kp,
            candidate_desc,
            source,
            extractor.norm_type,
            self.ratio,
            self.ransac_reproj_threshold,
            self.min_inliers,
            self.mutual_check,
        )
        return candidate_kp, good_matches, mask, result
