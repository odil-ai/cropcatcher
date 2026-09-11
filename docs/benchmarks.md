# Benchmarks

## The dataset

`tests/fixtures/mandragore/` is built from the BnF's
[Mandragore — échantillon segmenté 2019](https://api.bnf.fr/index.php/fr/mandragore-echantillon-segmente-2019):
631 manually segmented illuminations across 8 real Gallica manuscripts.

`dataset.json` packages **10 cases from 5 manuscripts**. Each has a query crop
cached locally — the one asset a manifest cannot regenerate — plus the IIIF
service id of its true source canvas, resolved live from Gallica at evaluation
time. Manuscripts span French, Arabic and Greek traditions, from busy hunting
scenes to near-flat gold grounds.

```bash
uv sync --group notebook
uv run python scripts/run_evaluation.py
```

## Method

`evaluate_method` builds the full 10×10 query×candidate matrix — every crop
against every canvas, not just its own source — giving Recall@1/@3,
precision/recall of the match decision, inlier statistics and timing.

`evaluate_transforms` complements it with a robustness study: each crop is
rotated, downscaled, JPEG-crushed or perspective-warped, then re-matched
against its own source.

## Results

| Method | Recall@1 | Recall@3 | Precision | F1 | mean inliers | ms/candidate |
|---|---|---|---|---|---|---|
| [SIFT] | 1.00 | 1.00 | 0.24 | 0.39 | 1424 | 154 |
| [AKAZE] | 1.00 | 1.00 | 1.00 | 1.00 | 817 | 84 |
| [ORB] | 0.90 | 0.90 | 0.90 | 0.90 | 554 | 14 |

![Match quality by method](evaluation/quality.png)

![Speed vs retrieval accuracy](evaluation/tradeoff.png)

- **SIFT and AKAZE both retrieve the right page every time.** AKAZE does it at
  roughly half the cost, and stays precise even at the default
  `min_inliers=8`.
- **SIFT's low precision is an artifact of thresholding raw inliers**, not of
  the score: a few unrelated but densely written pages coincidentally clear 8
  inliers. Ranking by `score` is unaffected. Either raise `min_inliers` or
  enable [`mutual_check`](usage.md#mutual-check), which lifts precision to
  0.91 and doubles the score margin.
- **ORB is ~11× faster but misses one case** — a small, low-contrast crop
  where its default 500 features cannot produce 8 consistent inliers.

!!! note "Numbers move slightly between runs"
    FLANN builds *randomized* [k-d trees][kdtree] and searches them
    approximately, and [RANSAC] samples randomly. Two identical runs therefore differ a little,
    most visibly in precision at low thresholds. Use `BFMatcher` and
    `cv2.setRNGSeed()` if you need reproducibility.

## Robustness

![Robustness to query-side transformations](evaluation/robustness.png)

SIFT holds Recall = 1.0 through a 45° rotation, a 4× downscale and JPEG
quality 10. AKAZE matches it on recall but its inlier counts fall sharply
under aggressive downscaling. ORB degrades first.

## Retrieval in practice

A 45°-rotated crop, matched against all 10 candidates — true source in green:

![Retrieving a rotated crop](evaluation/example_gaston_phebus_f2v_veneurs.png)

And a harder one: a mostly flat gold-and-blue page, perspective-warped.

![Retrieving a warped crop](evaluation/example_latin757_f24_creation.png)

The margin is narrower here, which is why low-texture crops are the ones to
watch when tuning thresholds for a given corpus.

!!! warning "Sample size"
    Ten cases validate the pipeline end to end on real IIIF data and give an
    honest first read of the tradeoffs. They are not a statistically robust
    benchmark. Grow `dataset.json` from the full 631-segment CSV to sharpen
    these numbers.

[SIFT]: https://en.wikipedia.org/wiki/Scale-invariant_feature_transform
[AKAZE]: https://docs.opencv.org/4.10.0/db/d70/tutorial_akaze_matching.html
[ORB]: https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF
[RANSAC]: https://en.wikipedia.org/wiki/Random_sample_consensus
[homography]: https://en.wikipedia.org/wiki/Homography_(computer_vision)
[kdtree]: https://en.wikipedia.org/wiki/K-d_tree
