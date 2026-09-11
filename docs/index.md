# CropCatcher

<p align="center">
  <img src="assets/cropcatcher_logo.png" alt="CropCatcher" width="440">
</p>

<p align="center">
  <em>retrieve the image source of any visual crop</em>
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python 3.13+" src="https://img.shields.io/badge/python-3.13%2B-blue.svg"></a>
  <a href="https://github.com/astral-sh/uv"><img alt="uv" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json"></a>
  <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
  <img alt="PyPI" src="https://img.shields.io/badge/pypi-not%20published-lightgrey.svg">
  <a href="https://iiif.io/"><img alt="IIIF: Presentation v2/v3, Image API" src="https://img.shields.io/badge/IIIF-Presentation%20v2%2Fv3%20%7C%20Image%20API-2873AB"></a>
  <a href="https://github.com/"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
</p>


CropCatcher is a Python package for determining whether two images show the same physical content, either entirely or when one image is a crop of the other.

It orchestrates classical [OpenCV](https://opencv.org/)-based local feature matching methods such as [SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform), [AKAZE](https://docs.opencv.org/4.10.0/db/d70/tutorial_akaze_matching.html) and [ORB](https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF), with descriptor matching, [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus) and [homography](https://en.wikipedia.org/wiki/Homography_(computer_vision)) estimation for geometric verification and localization and exposes with a simple API.

CropCatcher is not a semantic similarity tool: It is designed to identify the exact source image, rank candidates and, when relevant, locate the query inside the matching image.

**Typical Cultural Heritage use case**: you've found an old, low-resolution or cropped image
somewhere — a screenshot, a thumbnail, a scan on a blog or in a PDF — and you
suspect it comes from a page held in a digital library such as Gallica.
CropCatcher searches across one or more IIIF manifests,
tells you which page
it's actually contained in and where, and hands back that page's IIIF
service — so you can fetch the original at full resolution and read off its
metadata (manuscript, shelfmark, folio) straight from the manifest.

![Is the query crop contained in a candidate folio?](evaluation/concept.png)

A real query crop (left) matched with SIFT against real folios from the
*same* manuscript on Gallicathe harder: the true
source scores clearly highest (green) and is accepted, while other pages
sharing the same script, style and parchment (red) score far lower and are
rejected. See [Benchmarks](benchmarks.md) for how this holds up across
methods and distortions.


## Install

```bash
uv add cropcatcher
```

```bash
pip install cropcatcher
```

Requires Python ≥ 3.13. Runtime dependencies: `opencv-contrib-python-headless`,
`numpy`.

!!! warning "Not published yet"
    CropCatcher is not on PyPI yet. Until the first release ships, install
    from a clone instead:

    ```bash
    git clone <repository> && cd IIIF_image_matcher && uv sync
    ```

## Quickstart

```python
from cropcatcher import Matcher

matcher = Matcher(method="sift")
results = matcher.search(query="illumination.jpg", images="folios/")

best = results.best
print(best.source, best.score, best.bbox)
```

Against a digital library instead of local files:

```python
results = matcher.search_iiif(
    query="illumination.jpg",
    manifests=["https://gallica.bnf.fr/iiif/ark:/12148/btv1b8427253m/manifest.json"],
    max_canvases=50,
)
```

## How it works

```mermaid
flowchart LR
    Q["Query image"] --> FD1["Feature detection<br/>& description"]
    C["Candidate image"] --> FD2["Feature detection<br/>& description"]
    FD1 --> DM["Descriptor matching<br/>+ ratio test"]
    FD2 --> DM
    DM --> RA["RANSAC homography"]
    RA --> SC["score + bbox"]
```

1. **Detection & description** — a few thousand keypoints per image, each
   encoded as a descriptor vector that survives scaling and rotation
   ([SIFT], [AKAZE], [ORB]).
2. **Matching** — each query descriptor is paired with its
   [nearest neighbour][nns] among the candidate's, then filtered by Lowe's
   ratio test (and optionally a [mutual check](usage.md#mutual-check)).
3. **[RANSAC]** — fits the [homography] that the largest subset of matches
   agrees on. This is what separates "many descriptors happen to look alike"
   from "these are the same physical content".
4. **Score & localization** — the score weighs inlier ratio and inlier count;
   the homography projects the crop's corners into the candidate to give a
   bounding box.

## Where to go next

| If you want to… | See |
|---|---|
| Match against local images, understand `MatchResult` | [Matching local images](usage.md) |
| Search IIIF manifests, tune cost | [Searching IIIF](iiif.md) |
| Reconcile a whole CSV of folios | [Batch reconciliation](reconciliation.md) |
| Know how well it actually works | [Benchmarks](benchmarks.md) |

## Citation

If you use CropCatcher in your research, please cite it. Metadata also lives
in `CITATION.cff` at the repository root, which GitHub exposes through its
"Cite this repository" button.

```bibtex
@software{terriel_cropcatcher,
  author       = {Terriel, Lucas},
  title        = {{CropCatcher: Locate a Visual Crop Inside Candidate Images or IIIF Manifests}},
  organization = {École nationale des chartes - PSL},
  year         = {2026},
  version      = {0.1.0},
  license      = {MIT}
}
```

[OpenCV]: https://opencv.org/
[SIFT]: https://en.wikipedia.org/wiki/Scale-invariant_feature_transform
[AKAZE]: https://docs.opencv.org/4.10.0/db/d70/tutorial_akaze_matching.html
[ORB]: https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF
[RANSAC]: https://en.wikipedia.org/wiki/Random_sample_consensus
[homography]: https://en.wikipedia.org/wiki/Homography_(computer_vision)
[nns]: https://en.wikipedia.org/wiki/Nearest_neighbor_search
