#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Persistent descriptor index for CropCatcher.

This module provides the :class:`Index`, which describes a corpus of
candidate images once and reuses those descriptors across queries. Indexes
can be saved to disk and reloaded with identical matching behaviour.
"""

from __future__ import annotations

import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .features import FeatureExtractor, get_extractor
from .matching.pipeline import match_pair
from .result import MatchResult, SearchResults
from .sources.iiif import DEFAULT_CONCURRENCY, load_iiif_image, resolve_manifest
from .sources.images import load_image, resolve_image_paths


@dataclass
class _Entry:
    source: str
    shape: tuple[int, int]
    keypoints: list[tuple]
    descriptors: np.ndarray | None


def _serialize_keypoints(keypoints: list[cv2.KeyPoint]) -> list[tuple]:
    return [
        (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
        for kp in keypoints
    ]


def _deserialize_keypoints(data: list[tuple]) -> list[cv2.KeyPoint]:
    return [
        cv2.KeyPoint(x, y, size, angle, response, octave, class_id)
        for x, y, size, angle, response, octave, class_id in data
    ]


class Index:
    """Precomputed keypoints and descriptors for a corpus of candidates.

    Describing the candidates once and reusing the result pays off as soon as
    several queries hit the same corpus. Note that :meth:`search` still scans
    every entry: the index is a descriptor cache, not an approximate nearest
    neighbour structure.

    :param method: Local feature extraction method: ``"sift"``, ``"akaze"``,
        ``"orb"``, ``"star_brief"`` or ``"surf"``.
    :type method: str
    :param ratio: Lowe's ratio threshold applied when filtering descriptor
        matches.
    :type ratio: float
    :param ransac_reproj_threshold: Reprojection error, in pixels, tolerated
        when deciding whether a match is a RANSAC inlier.
    :type ransac_reproj_threshold: float
    :param min_inliers: Minimum inlier count below which no bounding box or
        homography is reported.
    :type min_inliers: int
    :param extractor_options: Keyword arguments forwarded to the underlying
        OpenCV detector, e.g. ``{"nfeatures": 4000}``.
    :type extractor_options: dict | None
    :param mutual_check: Require descriptor matches to be symmetric.
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
        """Configure the index and the matching pipeline it will use."""
        self.method = method
        self.extractor_options = dict(extractor_options or {})
        self.extractor = get_extractor(method, **self.extractor_options)
        self.ratio = ratio
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.min_inliers = min_inliers
        self.mutual_check = mutual_check
        self._entries: list[_Entry] = []
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

    def add_images(self, images: str | Path | list[str | Path]) -> None:
        """Describe local images and add them to the index.

        :param images: A directory of images, a single image path, or a list
            of image paths.
        :type images: str | pathlib.Path | list[str | pathlib.Path]
        :return: ``None``.
        :rtype: None
        """
        for path in resolve_image_paths(images):
            self._add_entry(load_image(path), str(path))

    def add_iiif(
        self,
        manifests: str | list[str],
        size: str = "!1024,1024",
        max_workers: int = DEFAULT_CONCURRENCY,
        max_canvases: int | None = None,
    ) -> None:
        """Resolve IIIF manifests and add every canvas to the index.

        :param manifests: One manifest URL, or a list of manifest URLs.
        :type manifests: str | list[str]
        :param size: IIIF Image API size parameter used for the derivatives.
        :type size: str
        :param max_workers: Number of worker threads used to download and
            describe canvases concurrently.
        :type max_workers: int
        :param max_canvases: Stop after this many canvases per manifest.
            ``None`` indexes every canvas; truncating risks excluding the
            true source.
        :type max_canvases: int | None
        :return: ``None``.
        :rtype: None
        """
        if isinstance(manifests, str):
            manifests = [manifests]

        resources = [
            resource
            for manifest_url in manifests
            for resource in resolve_manifest(manifest_url, max_canvases=max_canvases)
        ]
        self.add_iiif_services(
            [resource.service_id for resource in resources],
            size=size,
            max_workers=max_workers,
        )

    def add_iiif_services(
        self,
        service_ids: list[str],
        size: str = "!1024,1024",
        max_workers: int = DEFAULT_CONCURRENCY,
    ) -> None:
        """Index specific IIIF Image API services, skipping manifest resolution.

        Useful when only part of a manuscript is worth indexing: a window of
        canvases around an expected page, or the shortlist of a first,
        cheaper pass. Indexing cost is paid per canvas downloaded.

        :param service_ids: IIIF Image API service identifiers to index.
        :type service_ids: list[str]
        :param size: IIIF Image API size parameter used for the derivatives.
        :type size: str
        :param max_workers: Number of worker threads used to download and
            describe canvases concurrently.
        :type max_workers: int
        :return: ``None``.
        :rtype: None
        """

        def download_and_describe(service_id: str) -> _Entry:
            """Download and describe one canvas.

            Doing both per canvas (rather than downloading everything first)
            overlaps network waits with feature extraction, spreads extraction
            over several cores — OpenCV releases the GIL — and keeps only
            `max_workers` decoded images in memory instead of the whole manifest.
            """
            image = load_iiif_image(service_id, size=size)
            keypoints, descriptors = self._thread_extractor().detect_and_compute(image)
            return _Entry(
                source=service_id,
                shape=image.shape,
                keypoints=_serialize_keypoints(keypoints),
                descriptors=descriptors,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            self._entries.extend(executor.map(download_and_describe, service_ids))

    def _add_entry(self, image: np.ndarray, source: str) -> None:
        keypoints, descriptors = self._thread_extractor().detect_and_compute(image)
        self._entries.append(
            _Entry(
                source=source,
                shape=image.shape,
                keypoints=_serialize_keypoints(keypoints),
                descriptors=descriptors,
            )
        )

    def __len__(self) -> int:
        """Return the number of indexed candidates.

        :return: Number of entries held in the index.
        :rtype: int
        """
        return len(self._entries)

    def search(
        self, query: str | Path, max_workers: int = DEFAULT_CONCURRENCY
    ) -> SearchResults:
        """Match a query image against every indexed candidate.

        :param query: Path to the query image.
        :type query: str | pathlib.Path
        :param max_workers: Number of worker threads used to match entries
            concurrently. Pass ``1`` to match sequentially.
        :type max_workers: int
        :return: Candidates ranked by descending score.
        :rtype: SearchResults
        """
        query_image = load_image(query)
        query_kp, query_desc = self.extractor.detect_and_compute(query_image)

        def match_entry(entry: _Entry) -> MatchResult:
            return self._match_entry(query_kp, query_desc, query_image.shape, entry)

        # Purely CPU-bound, but OpenCV releases the GIL during descriptor
        # matching and RANSAC, so threads still spread this over cores.
        if max_workers > 1 and len(self._entries) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(match_entry, self._entries))
        else:
            results = [match_entry(entry) for entry in self._entries]

        results.sort(key=lambda r: r.score, reverse=True)
        return SearchResults(results)

    def _match_entry(
        self,
        query_kp: list[cv2.KeyPoint],
        query_desc: np.ndarray | None,
        query_shape: tuple[int, int],
        entry: _Entry,
    ) -> MatchResult:
        candidate_kp = _deserialize_keypoints(entry.keypoints)
        result, _, _ = match_pair(
            query_kp,
            query_desc,
            query_shape,
            candidate_kp,
            entry.descriptors,
            entry.source,
            self.extractor.norm_type,
            self.ratio,
            self.ransac_reproj_threshold,
            self.min_inliers,
            self.mutual_check,
        )
        return result

    def save(self, path: str | Path) -> None:
        """Persist the index and its tuning parameters to disk.

        The file is written with :mod:`pickle`, so only load indexes you
        produced yourself or otherwise trust.

        :param path: Destination file path.
        :type path: str | pathlib.Path
        :return: ``None``.
        :rtype: None
        """
        payload = {
            "method": self.method,
            "ratio": self.ratio,
            "ransac_reproj_threshold": self.ransac_reproj_threshold,
            "min_inliers": self.min_inliers,
            "extractor_options": self.extractor_options,
            "mutual_check": self.mutual_check,
            "entries": self._entries,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> Index:
        """Reload an index previously written by :meth:`save`.

        Tuning parameters are restored alongside the descriptors, so the
        reloaded index matches identically.

        :param path: Path to the saved index file.
        :type path: str | pathlib.Path
        :return: The reloaded index.
        :rtype: Index
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)

        index = cls(
            method=payload["method"],
            ratio=payload["ratio"],
            ransac_reproj_threshold=payload["ransac_reproj_threshold"],
            min_inliers=payload["min_inliers"],
            # older index files predate this field
            extractor_options=payload.get("extractor_options"),
            mutual_check=payload.get("mutual_check", False),
        )
        index._entries = payload["entries"]
        return index
