#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Console entry point for CropCatcher.

This module only reports the installed version; the public API lives in the
:mod:`cropcatcher` package.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def main() -> None:
    """Print the installed CropCatcher version.

    :return: ``None``.
    :rtype: None
    """
    try:
        installed = version("cropcatcher")
    except PackageNotFoundError:
        installed = "unknown (not installed)"
    print(f"CropCatcher {installed}")


if __name__ == "__main__":
    main()
