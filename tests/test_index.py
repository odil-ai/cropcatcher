#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for the persistent descriptor index."""

from __future__ import annotations

from pathlib import Path

import pytest

from cropcatcher import Index
from cropcatcher.sources import iiif


def test_index_search_ranks_true_source_first(image_dir: Path) -> None:
    index = Index(method="sift")
    index.add_images(image_dir / "candidates")
    assert len(index) == 2

    results = index.search(image_dir / "query" / "query_crop.png")

    best = results.best
    assert best is not None
    assert Path(best.source).name == "folio_142r.png"
    assert best.inliers >= 8
    assert best.bbox is not None


def test_index_save_and_load_round_trip(image_dir: Path, tmp_path: Path) -> None:
    index = Index(method="sift")
    index.add_images(image_dir / "candidates")

    index_path = tmp_path / "collection.cropcatcher"
    index.save(index_path)

    loaded = Index.load(index_path)
    assert len(loaded) == len(index)

    results = loaded.search(image_dir / "query" / "query_crop.png")
    best = results.best
    assert best is not None
    assert Path(best.source).name == "folio_142r.png"
    assert best.inliers >= 8


def test_index_add_iiif(image_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folio_path = image_dir / "candidates" / "folio_142r.png"
    folio_bytes = folio_path.read_bytes()

    manifest = {
        "sequences": [
            {
                "canvases": [
                    {
                        "label": "folio 142r",
                        "images": [
                            {
                                "resource": {
                                    "service": {
                                        "@id": "https://example.org/iiif/folio142r"
                                    }
                                }
                            }
                        ],
                    }
                ]
            }
        ],
    }

    monkeypatch.setattr(iiif, "_fetch_json", lambda url: manifest)
    monkeypatch.setattr(iiif, "_fetch_bytes", lambda url: folio_bytes)

    index = Index(method="sift")
    index.add_iiif("https://example.org/manifest.json")
    assert len(index) == 1

    results = index.search(image_dir / "query" / "query_crop.png")
    best = results.best
    assert best is not None
    assert best.source == "https://example.org/iiif/folio142r"


def test_extractor_options_cap_keypoints_and_survive_round_trip(
    image_dir: Path, tmp_path: Path
) -> None:
    capped = Index(method="sift", extractor_options={"nfeatures": 50})
    capped.add_images(image_dir / "candidates")

    uncapped = Index(method="sift")
    uncapped.add_images(image_dir / "candidates")

    capped_keypoints = max(len(e.keypoints) for e in capped._entries)
    assert capped_keypoints <= 50
    assert max(len(e.keypoints) for e in uncapped._entries) > capped_keypoints

    index_path = tmp_path / "capped.cropcatcher"
    capped.save(index_path)
    assert Index.load(index_path).extractor_options == {"nfeatures": 50}


def test_add_iiif_services_indexes_only_the_given_canvases(
    image_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folio_bytes = (image_dir / "candidates" / "folio_142r.png").read_bytes()
    monkeypatch.setattr(iiif, "_fetch_bytes", lambda url: folio_bytes)

    index = Index(method="sift")
    index.add_iiif_services(
        [
            "https://example.org/iiif/f10",
            "https://example.org/iiif/f11",
        ]
    )

    assert len(index) == 2
    assert [e.source for e in index._entries] == [
        "https://example.org/iiif/f10",
        "https://example.org/iiif/f11",
    ]
