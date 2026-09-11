#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature extraction methods available to CropCatcher.

This module holds the registry of local feature extractors and the factory
used to instantiate them by name.
"""

from __future__ import annotations

from .akaze import AKAZEExtractor
from .base import FeatureExtractor
from .orb import ORBExtractor
from .sift import SIFTExtractor
from .star_brief import StarBriefExtractor
from .surf import SURFExtractor

_REGISTRY: dict[str, type[FeatureExtractor]] = {
    "sift": SIFTExtractor,
    "akaze": AKAZEExtractor,
    "orb": ORBExtractor,
    "star_brief": StarBriefExtractor,
    "surf": SURFExtractor,
}


def get_extractor(method: str, **kwargs) -> FeatureExtractor:
    """Instantiate a feature extractor by name.

    :param method: Method name, case-insensitive: ``"sift"``, ``"akaze"``,
        ``"orb"``, ``"star_brief"`` or ``"surf"``.
    :type method: str
    :param kwargs: Keyword arguments forwarded to the extractor's
        constructor, and from there to the underlying OpenCV detector.
    :type kwargs: typing.Any
    :return: The instantiated extractor.
    :rtype: FeatureExtractor
    :raises ValueError: If ``method`` is not a registered method name.
    """
    try:
        extractor_cls = _REGISTRY[method.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown feature method {method!r}. Available: {available}"
        ) from exc
    return extractor_cls(**kwargs)


__all__ = [
    "FeatureExtractor",
    "SIFTExtractor",
    "AKAZEExtractor",
    "ORBExtractor",
    "StarBriefExtractor",
    "SURFExtractor",
    "get_extractor",
]
