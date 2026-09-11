#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for the retry behaviour of IIIF HTTP fetches."""

from __future__ import annotations

import urllib.error

import pytest

from cropcatcher.sources import iiif


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        return None


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = {"Retry-After": retry_after} if retry_after else {}
    return urllib.error.HTTPError("https://example.org", code, "boom", headers, None)


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(iiif.time, "sleep", lambda seconds: None)


def test_retries_then_succeeds_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = []

    def flaky(request, timeout=None):
        attempts.append(request.full_url)
        if len(attempts) < 3:
            raise _http_error(429)
        return _Response(b"payload")

    monkeypatch.setattr(iiif.urllib.request, "urlopen", flaky)

    assert iiif._fetch_bytes("https://example.org/x") == b"payload"
    assert len(attempts) == 3


def test_gives_up_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = []

    def always_throttled(request, timeout=None):
        attempts.append(1)
        raise _http_error(503)

    monkeypatch.setattr(iiif.urllib.request, "urlopen", always_throttled)

    with pytest.raises(urllib.error.HTTPError):
        iiif._fetch_bytes("https://example.org/x")
    assert len(attempts) == iiif.MAX_RETRIES + 1


def test_does_not_retry_a_permanent_error(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = []

    def not_found(request, timeout=None):
        attempts.append(1)
        raise _http_error(404)

    monkeypatch.setattr(iiif.urllib.request, "urlopen", not_found)

    with pytest.raises(urllib.error.HTTPError):
        iiif._fetch_bytes("https://example.org/x")
    assert len(attempts) == 1


def test_honours_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    waits = []
    monkeypatch.setattr(iiif.time, "sleep", waits.append)

    calls = []

    def throttled_once(request, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise _http_error(429, retry_after="7")
        return _Response(b"ok")

    monkeypatch.setattr(iiif.urllib.request, "urlopen", throttled_once)

    assert iiif._fetch_bytes("https://example.org/x") == b"ok"
    assert waits == [7.0]


def test_retries_network_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def flaky_network(request, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.URLError(TimeoutError("handshake timed out"))
        return _Response(b"{}")

    monkeypatch.setattr(iiif.urllib.request, "urlopen", flaky_network)

    assert iiif._fetch_json("https://example.org/manifest.json") == {}
    assert len(calls) == 2
