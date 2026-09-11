#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for the evaluation harness."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from cropcatcher import evaluation
from cropcatcher.sources import iiif


def _make_textured_image(seed: int, size: int = 600) -> np.ndarray:
    """Same recipe as tests/conftest.py's image_dir fixture: distinctive text
    overlays keep keypoints from spuriously matching across unrelated images."""
    rng = np.random.default_rng(seed)
    image = np.full((size, size), 255, dtype=np.uint8)

    for _ in range(60):
        shape = rng.integers(0, 3)
        pt1 = tuple(rng.integers(0, size, size=2).tolist())
        pt2 = tuple(rng.integers(0, size, size=2).tolist())
        color = int(rng.integers(0, 180))
        thickness = int(rng.integers(1, 4))
        if shape == 0:
            cv2.rectangle(image, pt1, pt2, color, thickness)
        elif shape == 1:
            radius = int(rng.integers(5, 40))
            cv2.circle(image, pt1, radius, color, thickness)
        else:
            cv2.line(image, pt1, pt2, color, thickness)

    for _ in range(15):
        org = tuple(rng.integers(0, size - 20, size=2).tolist())
        cv2.putText(
            image,
            "CropCatcher",
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            int(rng.integers(0, 120)),
            1,
            cv2.LINE_AA,
        )

    return image


def _encode(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


@pytest.fixture()
def two_case_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[evaluation.EvaluationCase]:
    """Two unrelated query/candidate pairs, served through mocked IIIF fetches."""
    folio_a = _make_textured_image(seed=1)
    folio_b = _make_textured_image(seed=2)
    crop_a = folio_a[100:300, 120:360].copy()
    crop_b = folio_b[80:260, 90:340].copy()

    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    cv2.imwrite(str(crops_dir / "case_a.png"), crop_a)
    cv2.imwrite(str(crops_dir / "case_b.png"), crop_b)

    service_bytes = {
        "https://example.org/iiif/folio_a": _encode(folio_a),
        "https://example.org/iiif/folio_b": _encode(folio_b),
    }

    def fake_fetch_bytes(url: str) -> bytes:
        for service_id, data in service_bytes.items():
            if url.startswith(service_id):
                return data
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(iiif, "_fetch_bytes", fake_fetch_bytes)

    return [
        evaluation.EvaluationCase(
            case_id="case_a",
            crop_path=crops_dir / "case_a.png",
            service_id="https://example.org/iiif/folio_a",
        ),
        evaluation.EvaluationCase(
            case_id="case_b",
            crop_path=crops_dir / "case_b.png",
            service_id="https://example.org/iiif/folio_b",
        ),
    ]


def test_evaluate_method_ranks_true_source_first(
    two_case_dataset: list[evaluation.EvaluationCase],
) -> None:
    result = evaluation.evaluate_method(two_case_dataset, method="sift")

    assert result.method == "sift"
    assert result.recall_at_1 == 1.0
    assert result.recall_at_3 == 1.0
    assert result.recall == 1.0  # both true pairs are detected as matches
    assert result.f1 > 0.0
    assert result.mean_inliers_positive >= 8
    assert result.mean_score_positive > result.mean_score_negative
    assert len(result.matrix) == 2
    assert all(len(row) == 2 for row in result.matrix)


def test_evaluate_runs_every_requested_method(
    two_case_dataset: list[evaluation.EvaluationCase],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(evaluation, "load_dataset", lambda path: two_case_dataset)

    results = evaluation.evaluate(tmp_path / "dataset.json", methods=("sift", "orb"))

    assert [r.method for r in results] == ["sift", "orb"]
    assert all(r.recall_at_1 == 1.0 for r in results)


def test_default_transforms_change_the_image() -> None:
    image = _make_textured_image(seed=3)
    for name, transform in evaluation.DEFAULT_TRANSFORMS.items():
        transformed = transform(image)
        assert transformed.size > 0, name
        if name != "original":
            assert transformed.shape != image.shape or not np.array_equal(
                transformed, image
            ), name


def test_transform_robustness_degrades_gracefully(
    two_case_dataset: list[evaluation.EvaluationCase],
) -> None:
    transforms = {
        "original": evaluation.DEFAULT_TRANSFORMS["original"],
        "rotate_15": evaluation.DEFAULT_TRANSFORMS["rotate_15"],
    }

    results = evaluation.evaluate_transform_robustness(
        two_case_dataset, method="sift", transforms=transforms
    )

    assert [r.transform for r in results] == ["original", "rotate_15"]
    original, rotated = results
    assert original.recall == 1.0
    assert original.mean_score > 0.0
    # a mild rotation should still be recoverable by a scale/rotation
    # invariant method such as SIFT.
    assert rotated.recall == 1.0
