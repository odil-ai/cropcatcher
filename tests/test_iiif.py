#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for IIIF manifest resolution and Image API URL construction."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import pytest

from cropcatcher import Matcher
from cropcatcher.sources import iiif

MANIFEST_V2 = {
    "@context": "http://iiif.io/api/presentation/2/context.json",
    "@id": "https://example.org/manifest.json",
    "@type": "sc:Manifest",
    "sequences": [
        {
            "canvases": [
                {
                    "@id": "https://example.org/canvas/1",
                    "label": "folio 142r",
                    "images": [
                        {
                            "resource": {
                                "@id": "https://example.org/iiif/folio142r/full/full/0/default.jpg",
                                "service": {
                                    "@id": "https://example.org/iiif/folio142r",
                                    "profile": "http://iiif.io/api/image/2/level2.json",
                                },
                            }
                        }
                    ],
                }
            ]
        }
    ],
}

MANIFEST_V3 = {
    "@context": "http://iiif.io/api/presentation/3/context.json",
    "id": "https://example.org/manifest.json",
    "type": "Manifest",
    "items": [
        {
            "id": "https://example.org/canvas/1",
            "type": "Canvas",
            "label": {"en": ["folio 142r"]},
            "items": [
                {
                    "id": "https://example.org/page/1",
                    "type": "AnnotationPage",
                    "items": [
                        {
                            "id": "https://example.org/annotation/1",
                            "type": "Annotation",
                            "motivation": "painting",
                            "body": {
                                "id": "https://example.org/iiif/folio142r/full/max/0/default.jpg",
                                "service": [
                                    {
                                        "id": "https://example.org/iiif/folio142r",
                                        "type": "ImageService3",
                                    }
                                ],
                            },
                        }
                    ],
                }
            ],
        }
    ],
}


def test_resolve_manifest_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(iiif, "_fetch_json", lambda url: MANIFEST_V2)

    resources = iiif.resolve_manifest("https://example.org/manifest.json")

    assert len(resources) == 1
    assert resources[0].service_id == "https://example.org/iiif/folio142r"
    assert resources[0].canvas_label == "folio 142r"


def test_resolve_manifest_v3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(iiif, "_fetch_json", lambda url: MANIFEST_V3)

    resources = iiif.resolve_manifest("https://example.org/manifest.json")

    assert len(resources) == 1
    assert resources[0].service_id == "https://example.org/iiif/folio142r"
    assert resources[0].canvas_label == "folio 142r"


def test_resolve_manifest_max_canvases(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = {
        "sequences": [
            {
                "canvases": [
                    {
                        "label": f"folio {n}",
                        "images": [
                            {
                                "resource": {
                                    "service": {"@id": f"https://example.org/iiif/f{n}"}
                                }
                            }
                        ],
                    }
                    for n in range(5)
                ]
            }
        ]
    }
    monkeypatch.setattr(iiif, "_fetch_json", lambda url: manifest)

    resources = iiif.resolve_manifest(
        "https://example.org/manifest.json", max_canvases=2
    )

    assert [r.canvas_label for r in resources] == ["folio 0", "folio 1"]


def test_image_api_url() -> None:
    url = iiif.image_api_url("https://example.org/iiif/folio142r", size="!512,512")
    assert url == "https://example.org/iiif/folio142r/full/!512,512/0/default.jpg"


def test_search_iiif_end_to_end(
    image_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folio_path = image_dir / "candidates" / "folio_142r.png"
    folio_bytes = folio_path.read_bytes()

    manifest = {
        "@context": "http://iiif.io/api/presentation/2/context.json",
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

    matcher = Matcher(method="sift")
    results = matcher.search_iiif(
        query=image_dir / "query" / "query_crop.png",
        manifests="https://example.org/manifest.json",
    )

    assert len(results) == 1
    best = results.best
    assert best is not None
    assert best.source == "https://example.org/iiif/folio142r"
    assert best.inliers >= 8


def test_explain_iiif(image_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folio_path = image_dir / "candidates" / "folio_142r.png"
    monkeypatch.setattr(iiif, "_fetch_bytes", lambda url: folio_path.read_bytes())

    matcher = Matcher(method="sift")
    debug = matcher.explain_iiif(
        query=image_dir / "query" / "query_crop.png",
        service_id="https://example.org/iiif/folio142r",
    )

    assert debug.result.source == "https://example.org/iiif/folio142r"
    assert debug.result.inliers >= 8
    assert debug.mask is not None


def test_load_iiif_images_concurrent_preserves_order_and_is_faster(
    monkeypatch: pytest.MonkeyPatch, image_dir: Path
) -> None:
    folio_bytes = (image_dir / "candidates" / "folio_142r.png").read_bytes()
    delay = 0.05
    service_ids = [f"https://example.org/iiif/folio{i}" for i in range(6)]

    def slow_fetch_bytes(url: str) -> bytes:
        time.sleep(delay)
        return folio_bytes

    monkeypatch.setattr(iiif, "_fetch_bytes", slow_fetch_bytes)

    start = time.perf_counter()
    images = iiif.load_iiif_images(service_ids, max_workers=len(service_ids))
    elapsed = time.perf_counter() - start

    assert len(images) == len(service_ids)
    # sequential would take len(service_ids) * delay; concurrent should be
    # close to a single delay, well under half the sequential time.
    assert elapsed < (len(service_ids) * delay) / 2


def test_load_iiif_image(monkeypatch: pytest.MonkeyPatch, image_dir: Path) -> None:
    folio_path = image_dir / "candidates" / "folio_142r.png"
    folio_bytes = folio_path.read_bytes()
    monkeypatch.setattr(iiif, "_fetch_bytes", lambda url: folio_bytes)

    image = iiif.load_iiif_image("https://example.org/iiif/folio142r")

    expected = cv2.imread(str(folio_path), cv2.IMREAD_GRAYSCALE)
    assert image.shape == expected.shape
