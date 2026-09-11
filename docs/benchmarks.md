# Evaluation

CropCatcher ships a small, real-world evaluation dataset and a benchmarking
module (`cropcatcher.evaluation`) that compares SIFT, AKAZE and ORB on actual
IIIF manifests (not synthetic images).

## Dataset

`tests/fixtures/mandragore/` is built from the BnF's dataset [Mandragore — échantillon
segmenté 2019](https://api.bnf.fr/index.php/fr/mandragore-echantillon-segmente-2019)
: 631 manually segmented illuminations (bounding box + legend) across 8
real Gallica manuscripts, released under the [Gallica reuse
terms](https://gallica.bnf.fr/html/und/conditions-dutilisation-des-contenus-de-gallica).
The full CSV (`Segments_Enluminures.csv`) is kept as-is for future, larger
evaluation runs.

`dataset.json` packages a hand-picked subset of **10 cases from 5
manuscripts**. Each case has a query crop cached locally (the one asset that
can't be regenerated from a manifest) and the IIIF Image API service id of
its true source canvas, resolved live from Gallica at evaluation time:

| manuscript | legend | canvas |
|---|---|---|
| Français 1291 — Gaston Phébus, *Livre de la chasse* | Gaston Phébus et veneurs | f. 2v |
| Français 1291 — Gaston Phébus, *Livre de la chasse* | Faune : cerf | f. 5v |
| Arabe 274 — *Barlaam et Josaphat* | Saints Barlaam et Josaphat | f. 1v |
| Arabe 274 — *Barlaam et Josaphat* | Saint Josaphat dans son palais | f. 11v |
| Grec 2736 — Oppien, *Cynégétiques* | Chasse diurne | f. 4v |
| Grec 2736 — Oppien, *Cynégétiques* | Chevaux combattant des bêtes sauvages | f. 9 |
| Français 225 — Pétrarque, *De remediis utriusque fortunae* | Allégorie : Roue de la Fortune | f. 1 |
| Français 225 — Pétrarque, *De remediis utriusque fortunae* | Allégorie : Raison et l'homme | f. 8 |
| Latin 757 — Bible historiale | Création : lumière | f. 24 |
| Latin 757 — Bible historiale | Armes de Julien Regin | f. 4 |

## Method

`cropcatcher.evaluation.evaluate_method` builds the full 10×10 query×candidate
similarity matrix (every crop matched against every candidate canvas, not
only its own true source) from which it derives:

- **Recall@1 / Recall@3**: does the true source canvas rank first / in the
  top 3 by score?
- **Precision / recall / F1**: treating `inliers >= min_inliers` (the same
  threshold `Matcher` uses to gate a `bbox`) as a binary match/no-match
  classifier over all 100 pairs.
- **Mean inliers / inlier ratio** on the 10 true pairs.
- **Processing time**, split into feature detection and descriptor
  matching + RANSAC.


`cropcatcher.evaluation.evaluate_transforms` complements the retrieval benchmark with a robustness study. A simple crop-to-source comparison is too easy, so each query crop is degraded with realistic image transformations such as perspective changes, resizing and recompression, then matched only against its true source canvas.

The goal is not to rank a corpus, but to measure how score and inlier counts degrade under realistic distortions.

| transform | what it simulates |
|---|---|
| `rotate_15` / `rotate_45` | the crop was photographed or scanned at an angle |
| `scale_50` / `scale_25` | a lower-resolution reproduction of the same crop |
| `jpeg_q10` | heavy JPEG compression artifacts |
| `perspective_heavy` | a page photographed off-axis (non-affine distortion) |

Reproduce all of it with:

```bash
uv sync --group notebook   # pulls in matplotlib
uv run python scripts/run_evaluation.py
```

## Results

| method | Recall@1 | Recall@3 | Precision | Recall | F1 | mean inliers | inlier ratio | ms/candidate |
|---|---|---|---|---|---|---|---|---|
| SIFT  | 1.00 | 1.00 | 0.24 | 1.00 | 0.39 | 1424.7 | 0.94 | 154.0 |
| AKAZE | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 817.0  | 0.96 | 83.5  |
| ORB   | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 553.5  | 0.84 | 13.9  |

![Mean time per candidate, by method](evaluation/time_per_candidate.png)

![Match quality, by method](evaluation/quality.png)

![Speed vs. retrieval accuracy tradeoff](evaluation/tradeoff.png)

## Reading the results

- **SIFT and AKAZE achieve perfect Recall@1/@3** on this sample, with true matches clearly separated from false ones by score.
- **SIFT's low precision comes from the hard `min_inliers` threshold**, not from ranking quality. Some visually similar pages occasionally pass the threshold, while true matches still score much higher. A higher `min_inliers` or score threshold improves binary decisions.
- **AKAZE matches SIFT's retrieval accuracy while being about 46% faster per candidate** and remains more precise with the default threshold.
- **ORB is about 11x faster than SIFT**, but recall drops to 0.90 because small or low-contrast crops may not produce enough consistent features.

  
## Robustness to transformations

![Robustness to query-side transformations](evaluation/robustness.png)

- **SIFT remains fully robust** across all tested transformations, including 45° rotation, 4x downscaling and heavy JPEG compression.
- **AKAZE also keeps Recall = 1.0**, but aggressive downscaling significantly reduces the number of inliers, even when matches still pass the threshold.
- **ORB is the most fragile**, with lower recall even at full resolution and further degradation under strong downscaling.

## Examples: a distorted crop among good and bad candidates

Two real examples illustrate how SIFT behaves under distortion:

![Retrieving a 45°-rotated crop](evaluation/example_gaston_phebus_f2v_veneurs.png)

- **45° rotation**: the true source still ranks first with a score of 0.72, ahead of all decoys.

![Retrieving a perspective-warped crop](evaluation/example_latin757_f24_creation.png)

- **Perspective distortion on a low-texture crop**: This crop ("Création : lumière") is one of the harder cases in the dataset (
mostly flat gold and blue fields with limited texture). The true source remains first (0.83 vs. 0.34 for the closest decoy), but the smaller margin shows that low-texture regions are more sensitive to parameter tuning.

This 10-case dataset is an **end-to-end validation sample**, not a statistically robust benchmark. Expanding `tests/fixtures/mandragore/dataset.json` with more IIIF examples will make the comparison more representative.
