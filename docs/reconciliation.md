# Batch reconciliation

[`notebooks/reconciliation.ipynb`](https://github.com/odil-ai/cropcatcher/blob/main/notebooks/reconciliation.ipynb) reconciles a CSV of folio images with their corresponding IIIF manifests.

Each input row represents **one folio image and its manuscript manifest**. The notebook searches the relevant canvases and writes the **top-N candidate pages** for each folio.

## Input and output

The input uses:

- an image filename resolved against `IMAGES_DIR`
- an expected folio label such as `1r` or `175v`
- one or more IIIF manifest URLs separated by `|`

Rows with missing images or no manifest URL are skipped.

The output uses a **long format**, with one row per `(folio, rank)`:

| Column | Meaning |
|---|---|
| `rank` | Candidate rank, with `1` being the best match. |
| `candidate_service_id`, `candidate_canvas_label` | Matched IIIF canvas and label. |
| `score`, `matches`, `inliers`, `inlier_ratio`, `bbox` | Matching results. |
| `n_canvases_indexed` | Number of canvases searched for this folio. |
| `target_in_manifest` | Whether the expected folio label exists in the indexed canvases. |
| `status`, `error` | Processing status and possible error. |

Because search is the expensive step, run it once with a sufficiently large `TOP_K`. Results can then be filtered with `top_n(df, 2)` or reshaped to one row per folio with `to_wide(df, n)`.

## Grouping by manifest

Several folios usually belong to the same manuscript. Running `search_iiif` independently for each folio would repeatedly download and process the same manifest.

The notebook instead builds one [`Index`](usage.md#index) per manifest and reuses it for all associated folios.

On a 455-canvas manifest with 15 folios, this reduced processing time from about **33 minutes to 6 minutes**.

## Search window

The main cost optimization is to restrict the search to canvases near the expected folio label:

```python
SEARCH_WINDOW = 20          # None = search the full manifest
ON_MISSING_LABEL = "skip"   # or "full"
```

Since fewer canvases are processed, this reduces image downloads, feature extraction and comparisons at the same time.

On the demo manifest, the same result took **3.3 s instead of 51.6 s**.

!!! warning "Label-dependent search"
    The search window is derived from the expected folio label. This makes the search faster, but it is no longer an independent test across the full manifest.

    For validation, run a sample with `SEARCH_WINDOW = None`.

## Resuming and failures

Long batch runs can be resumed:

```python
RESUME = True
```

Completed manifests are written to the output CSV immediately and skipped on the next run.

If `METHOD`, `SIZE` or `SEARCH_WINDOW` changes, delete the previous output file to avoid mixing incompatible results.

Manifest-level failures are recorded with `status="error"` and do not stop the batch. Temporary network failures are retried by the [IIIF layer](iiif.md#retries).

## Non-matching failures

Two situations must be separated from actual matching errors:

- **Target not present in the manifest**: no correct match is possible. Check `target_in_manifest`.
- **Expected folio label cannot be found**: no search window can be constructed, producing `status="skipped_no_label"`.

Only valid rows where the expected target can actually be searched should be used to evaluate matching quality.

## Cost

The total cost can be approximated as:

```text
cost ≈ manifests × canvases_indexed × t_index
     + folios × canvases_searched × t_compare
```

Indexing is paid once per manifest, while comparison is paid for every folio.

Reducing the number of processed canvases is therefore the most effective optimization, because it reduces both indexing and matching costs.