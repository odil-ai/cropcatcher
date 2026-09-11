# Batch reconciliation

[`notebooks/reconciliation.ipynb`](https://github.com/) takes a CSV where
**one row = one folio image + its manuscript's IIIF manifest**, and writes a
new CSV giving, for each folio, the **top-N candidate canvases** — i.e. which
digitized page each image actually is.

## Input and output

Input columns used: a filename resolved against `IMAGES_DIR`, the folio label
(`1r`, `175v`…), and one or more manifest URLs separated by `|`. Rows whose
image is missing, or whose manifest column is `blank`, are dropped.

Output is **long/tidy** — one row per *(folio, rank)*:

| Column | Meaning |
|---|---|
| `rank` | 1 = best candidate. |
| `candidate_service_id`, `candidate_canvas_label` | The matched canvas and its label. |
| `score`, `matches`, `inliers`, `inlier_ratio`, `bbox` | The `MatchResult` fields. |
| `n_canvases_indexed` | How many canvases this folio was compared against. |
| `target_in_manifest` | Whether a canvas labelled like the expected folio was in the index at all. |
| `status`, `error` | `ok`, `error`, or `skipped_no_label`. |

Because the search is the expensive part, run it **once** with a generous
`TOP_K`; narrowing to the top 1 or 2 afterwards is a free filter
(`top_n(df, 2)`), and `to_wide(df, n)` reshapes to one row per folio.

## Why it groups by manifest

Several folios usually share one manuscript (≈ 8 on average in our corpus).
Calling `search_iiif` per folio would re-download and re-describe the whole
manifest each time. The notebook instead builds one
[`Index`](usage.md#index) per manifest and searches every folio of that
manuscript in it. On a 455-canvas manifest with 15 folios that is ~6 minutes
instead of ~33.

## Search window

The decisive cost lever. You already know the expected folio, and the manifest
exposes canvas labels — so locate the likely position and index **only the
canvases around it** instead of the whole manuscript.

```python
SEARCH_WINDOW = 20        # None = index the whole manifest
ON_MISSING_LABEL = "skip" # or "full" to fall back to the whole manuscript
```

Since indexing means downloading, this cuts network, description *and*
comparisons at once — every other lever cuts only one of them. Measured on
the demo manifest: **51.6 s → 3.3 s** for an identical result.

!!! warning "It weakens the label sanity check"
    The window is chosen *from* the expected label, so verifying afterwards
    that the winning canvas carries that label is partly circular. It still
    catches a page labelled `175r` that does not contain the expected image,
    but it no longer validates retrieval across the whole manuscript. For an
    independent check, re-run a sample with `SEARCH_WINDOW = None`.

## Resuming and failures

A full-corpus run is a batch job of many hours; it will be interrupted.

```python
RESUME = True   # skip manifests already present in OUTPUT_CSV
```

Each finished manifest is appended to the output CSV immediately, and
manifests already in that file are skipped on the next run. Delete the output
file when you change `METHOD`, `SIZE` or `SEARCH_WINDOW`, otherwise stale
results are kept.

A manifest that fails outright (404, server down) is recorded with
`status="error"` and does not stop the batch. Transient failures are retried
with backoff inside the [IIIF layer](iiif.md#retries).

## Two failures that are not matching errors

Distinguish them before reading any agreement rate:

- **The target page is not in the manifest.** Manifests often digitize only
  part of a manuscript — one in our corpus exposes 3 pages for 399 expected
  folios. No match is possible. The `target_in_manifest` column isolates this.
- **The folio label cannot be located**, so no window can be built:
  `status="skipped_no_label"`.

Only the remaining rows say anything about matching quality.

## Cost

The notebook calibrates on the run you just did and extrapolates, separating
the two costs:

```
cost ≈ manifests × canvases_indexed × t_index      (paid once per manifest)
     + folios    × canvases_searched × t_compare   (paid per folio)
```

Both terms matter: indexing a canvas requires downloading it, so it is not
negligible. On our corpus the two were comparable. **Any lever that only
reduces comparisons — a faster method, fewer keypoints — cannot cut the total
by more than half.** Only reducing the number of canvases processed attacks
both.
