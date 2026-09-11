#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IIIF source resolution for CropCatcher.

This module resolves IIIF Presentation API v2 and v3 manifests into image
resources, builds IIIF Image API request URLs, and fetches derivatives over
HTTP with retries on transient failures.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

_TIMEOUT = 30
_USER_AGENT = "cropcatcher/0.1"
DEFAULT_CONCURRENCY = 8

# Institutional IIIF servers throttle and occasionally drop connections; a
# long batch run hits both routinely, so transient failures are retried with
# exponential backoff rather than aborting the whole job.
MAX_RETRIES = 3
BACKOFF_SECONDS = 1.0
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


@dataclass
class IIIFImageResource:
    """A single IIIF Image API resource discovered in a manifest canvas."""

    canvas_label: str | None
    service_id: str
    manifest_url: str


def _retry_delay(error: Exception, attempt: int) -> float | None:
    """Seconds to wait before retrying, or None if the error is permanent."""
    if isinstance(error, urllib.error.HTTPError):
        if error.code not in _RETRYABLE_STATUS:
            return None
        # 429/503 may tell us how long to wait; honour it when it is a plain
        # number of seconds (the HTTP-date form is rare and not worth parsing).
        retry_after = error.headers.get("Retry-After") if error.headers else None
        if retry_after and retry_after.strip().isdigit():
            return float(retry_after.strip())
    elif not isinstance(error, urllib.error.URLError | TimeoutError | OSError):
        return None

    return BACKOFF_SECONDS * (2**attempt)


def _fetch(url: str, headers: dict[str, str]) -> bytes:
    """Fetch a URL, retrying transient failures with exponential backoff."""
    request = urllib.request.Request(url, headers=headers)

    for attempt in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(  # noqa: S310
                request, timeout=_TIMEOUT
            ) as response:
                return response.read()
        except Exception as error:  # noqa: BLE001 - re-raised unless retryable
            delay = _retry_delay(error, attempt)
            if delay is None or attempt == MAX_RETRIES:
                raise
            time.sleep(delay)

    raise RuntimeError("unreachable")  # pragma: no cover


def _fetch_json(url: str) -> dict:
    return json.loads(
        _fetch(url, {"User-Agent": _USER_AGENT, "Accept": "application/json"})
    )


def _fetch_bytes(url: str) -> bytes:
    return _fetch(url, {"User-Agent": _USER_AGENT})


def _first_service_id(service) -> str | None:
    if isinstance(service, list):
        service = service[0] if service else None
    if not isinstance(service, dict):
        return None
    return service.get("@id") or service.get("id")


def _label_v3(label) -> str | None:
    if not isinstance(label, dict):
        return label
    for values in label.values():
        if values:
            return values[0]
    return None


def _resolve_v2(manifest: dict, manifest_url: str) -> list[IIIFImageResource]:
    resources = []
    for sequence in manifest.get("sequences", []):
        for canvas in sequence.get("canvases", []):
            label = canvas.get("label")
            for image in canvas.get("images", []):
                service_id = _first_service_id(image.get("resource", {}).get("service"))
                if service_id is not None:
                    resources.append(IIIFImageResource(label, service_id, manifest_url))
    return resources


def _resolve_v3(manifest: dict, manifest_url: str) -> list[IIIFImageResource]:
    resources = []
    for canvas in manifest.get("items", []):
        label = _label_v3(canvas.get("label"))
        for page in canvas.get("items", []):
            for annotation in page.get("items", []):
                if annotation.get("motivation") != "painting":
                    continue
                service_id = _first_service_id(
                    annotation.get("body", {}).get("service")
                )
                if service_id is not None:
                    resources.append(IIIFImageResource(label, service_id, manifest_url))
    return resources


def resolve_manifest(
    manifest_url: str, max_canvases: int | None = None
) -> list[IIIFImageResource]:
    """Resolve a IIIF Presentation manifest (v2 or v3) to its image resources.

    :param manifest_url: URL of the IIIF Presentation manifest.
    :type manifest_url: str
    :param max_canvases: Resolve only the first ``max_canvases`` canvases.
        Real manifests can hold hundreds of them, so this is useful for a
        quick demo or a two-stage search; ``None`` resolves all of them.
    :type max_canvases: int | None
    :return: One image resource per canvas exposing an Image API service.
    :rtype: list[IIIFImageResource]
    """
    manifest = _fetch_json(manifest_url)
    resources = (
        _resolve_v2(manifest, manifest_url)
        if "sequences" in manifest
        else _resolve_v3(manifest, manifest_url)
    )
    return resources[:max_canvases] if max_canvases is not None else resources


def image_api_url(service_id: str, size: str = "!1024,1024") -> str:
    """Build a IIIF Image API request URL for a full-region derivative.

    :param service_id: IIIF Image API service identifier.
    :type service_id: str
    :param size: Image API size parameter, e.g. ``"!1024,1024"`` to fit the
        image within a 1024-pixel box rather than downloading it at full
        resolution.
    :type size: str
    :return: The Image API request URL.
    :rtype: str
    """
    return f"{service_id.rstrip('/')}/full/{size}/0/default.jpg"


def load_iiif_image(service_id: str, size: str = "!1024,1024") -> np.ndarray:
    """Download and decode a IIIF image resource as grayscale.

    :param service_id: IIIF Image API service identifier.
    :type service_id: str
    :param size: Image API size parameter for the derivative.
    :type size: str
    :return: The decoded image as a 2-D grayscale array.
    :rtype: numpy.ndarray
    :raises ValueError: If the downloaded bytes cannot be decoded.
    """
    url = image_api_url(service_id, size)
    data = _fetch_bytes(url)
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode IIIF image: {url}")
    return image


def load_iiif_images(
    service_ids: list[str],
    size: str = "!1024,1024",
    max_workers: int = DEFAULT_CONCURRENCY,
) -> list[np.ndarray]:
    """Download several IIIF image resources concurrently.

    The work is network-bound, so threads help even though decoding is not.

    :param service_ids: IIIF Image API service identifiers to download.
    :type service_ids: list[str]
    :param size: Image API size parameter for the derivatives.
    :type size: str
    :param max_workers: Number of concurrent download threads.
    :type max_workers: int
    :return: The decoded grayscale images, in the same order as
        ``service_ids``.
    :rtype: list[numpy.ndarray]
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            executor.map(
                lambda service_id: load_iiif_image(service_id, size), service_ids
            )
        )
