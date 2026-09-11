# Matching local images

## `Matcher`

```python
from cropcatcher import Matcher

matcher = Matcher(method="sift")
results = matcher.search(
    query="illumination.jpg",
    images="folios/",   # a directory, a list of paths, or a single path
)
```

| Parameter | Default | What it does |
|---|---|---|
| `method` | `"sift"` | `"sift"`, `"akaze"`, `"orb"`, `"star_brief"`, `"surf"`. |
| `ratio` | `0.75` | Lowe's ratio test. Lower is stricter: fewer, cleaner matches. |
| `ransac_reproj_threshold` | `5.0` | Pixels of reprojection error tolerated when deciding an inlier. Raise it for distorted or low-resolution sources. |
| `min_inliers` | `8` | Below this, no `bbox`/`homography` is returned. Also the de-facto match/no-match cutoff. |
| `extractor_options` | `None` | Passed to the OpenCV detector, e.g. `{"nfeatures": 4000}` to cap SIFT keypoints. |
| `mutual_check` | `False` | Require matches to be symmetric — see [below](#mutual-check). |

`results` is a `SearchResults`: iterable, indexable, with `.best` and
`.top(k)`, sorted by descending score.

## `MatchResult`

```python
MatchResult(
    source="folio_142r.jpg",
    score=0.87,
    matches=94,
    inliers=61,
    inlier_ratio=0.65,
    bbox=(412, 580, 1034, 1280),
    homography=...,
)
```

| Field | Meaning |
|---|---|
| `source` | File path, or IIIF Image API service id for the IIIF entry points. |
| `score` | Confidence in `[0, 1]`: `0.6 × inlier_ratio + 0.4 × min(inliers/30, 1)`. Rewards a match that is both *clean* and *large*. |
| `matches` | Descriptor matches surviving the ratio test, before any geometric check. Upper bound on `inliers`. |
| `inliers` | Of those, the ones consistent with the estimated [homography] under [RANSAC]. **This is the real evidence** — a high `matches` with low `inliers` means coincidence, not a match. |
| `inlier_ratio` | `inliers / matches`. How clean the match is, independent of its size. |
| `bbox` | `(x0, y0, x1, y1)` of the query inside the candidate. `None` below `min_inliers`. |
| `homography` | The 3×3 projective transform, `None` under the same condition. |

!!! note "Which number should I threshold on?"
    `score`, not `inliers`. On the [benchmark set](benchmarks.md), thresholding
    raw inliers at the default `min_inliers=8` gave 0.26 precision for SIFT,
    because unrelated manuscript pages coincidentally clear 8 inliers. The
    score separates true from false pairs far more cleanly.

## Mutual check

mutual_check=True keeps only bidirectional matches: a query descriptor must select a candidate descriptor, and that candidate descriptor must also select the original query descriptor as its nearest neighbour.

```python
matcher = Matcher(method="sift", mutual_check=True)
``` 

This extra reverse matching pass removes many ambiguous one-sided matches. On the benchmark set, precision improved from 0.26 to 0.91, false positives dropped from 28 to 1, and the score margin between true and false pairs increased from 0.28 to 0.62, with no loss in Recall@1. Matching time increased by about 22%.

It is disabled by default for backward compatibility, but is recommended for automated batch matching where results are accepted without manual revie

## `Index`

When several queries hit the same corpus, describing the candidates once is
worth it:

```python
from cropcatcher import Index

index = Index(method="sift")
index.add_images("my_local_files/")
index.add_iiif(["https://example.org/manifest.json"])  # also downloaded concurrently

index.save("collection.cropcatcher") # saved for later

index = Index.load("collection.cropcatcher") # load you index
results = index.search("illumination.jpg") # search in your index
```

`Index` takes the same tuning parameters as `Matcher`, and persists them
alongside the descriptors so a reloaded index behaves identically.

!!! warning "It is a descriptor cache"
    `search()` scans **every** entry in the index. The cost grows linearly with corpus size.
    What it saves is re-downloading and re-describing candidates, not the
    matching itself. Expect roughly 40 ms per indexed image per query with
    SIFT. Descriptors are also bulky on disk: ~5 MB per full manuscript page.

## Visualizing a match

```python
from cropcatcher.visualization import draw_matches, draw_localization

debug = matcher.explain(query="illumination.jpg", candidate="folio_142r.jpg")
draw_matches(debug)          # inliers green, rejected red
draw_localization(debug.candidate_image, debug.query_image.shape,
                  debug.result.homography)
```

Dense, parallel green lines mean a geometrically coherent match. Lines going
in all directions mean the score is a false positive, whatever its value.

## Choosing a method

| Method | Speed | Notes |
|---|---|---|
| [`sift`][SIFT] | baseline | Most robust; the reference. 128-dim float descriptors, so bulky (~5 MB per manuscript page). |
| [`akaze`][AKAZE] | ~1.8× faster | Builds its scale space by nonlinear diffusion rather than Gaussian blur, preserving edges; binary descriptors. Same accuracy as SIFT on the benchmark, and precise even at low `min_inliers`. Good default for large corpora. |
| [`orb`][ORB] | ~11× faster | Binary descriptors on top of the [FAST] corner detector. Cheapest, but misses matches on small, low-contrast crops. Useful as a coarse first pass. |
| `star_brief` | fast | STAR/CenSurE detector + BRIEF binary descriptor. Works out of the box, not patented. |
| [`surf`][SURF] | — | Patented and absent from PyPI OpenCV wheels; raises a clear error unless you build OpenCV with `-DOPENCV_ENABLE_NONFREE=ON`. |

Binary descriptors (AKAZE, ORB, BRIEF) are compared with
[Hamming distance][hamming], SIFT's float descriptors with L2 distance
through a [k-d tree][kdtree] index.

[SIFT]: https://en.wikipedia.org/wiki/Scale-invariant_feature_transform
[AKAZE]: https://docs.opencv.org/4.10.0/db/d70/tutorial_akaze_matching.html
[ORB]: https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF
[SURF]: https://en.wikipedia.org/wiki/Speeded_up_robust_features
[FAST]: https://en.wikipedia.org/wiki/Features_from_accelerated_segment_test
[RANSAC]: https://en.wikipedia.org/wiki/Random_sample_consensus
[homography]: https://en.wikipedia.org/wiki/Homography_(computer_vision)
[hamming]: https://en.wikipedia.org/wiki/Hamming_distance
[kdtree]: https://en.wikipedia.org/wiki/K-d_tree
