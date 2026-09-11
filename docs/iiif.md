# Searching IIIF

CropCatcher resolves IIIF Presentation manifests (API v2 and v3) and fetches
candidate pages through the IIIF Image API. Tested against Gallica (BnF), the
Vatican Library, IRHT and the Bodleian.

```python
results = matcher.search_iiif(
    query="illumination.jpg",
    manifests=["https://gallica.bnf.fr/iiif/ark:/12148/btv1b8427253m/manifest.json"],
    size="!1024,1024",
    max_workers=16,
    max_canvases=50,
)
```

| Parameter | Default | What it does |
|---|---|---|
| `manifests` | — | One URL or a list. All their canvases become candidates. |
| `size` | `"!1024,1024"` | IIIF Image API size. The **server** resizes, so this controls bytes transferred, not just decode cost. |
| `max_workers` | `8` | Concurrent download+match workers. See [cost](#cost-and-tuning). |
| `max_canvases` | `None` | Keep only the first N canvases of each manifest. |

!!! danger "`max_canvases` is blunt"
    Manifests usually hold far **more** canvases than you have queries — 500+
    pages is common. Truncating almost certainly excludes the right page. Use
    it for throughput tests, not for real work; prefer a
    [targeted window](reconciliation.md#search-window).

## What actually gets downloaded

Every call downloads. There is **no cache**: `load_iiif_image()` issues an
HTTP GET each time, even for a URL fetched a moment ago.

What it does avoid is full resolution — it requests the derivative at `size`,
so ~100–200 KB per page instead of several MB. To query the same corpus
repeatedly without re-fetching, use [`Index`](usage.md#index), which downloads
and describes once.

## Retries

Institutional servers throttle. Gallica returns HTTP 429 under load; others
drop TLS handshakes. Transient failures (408, 429, 5xx, timeouts) are retried
up to `MAX_RETRIES` times with exponential backoff, honouring `Retry-After`
when the server sends it. Permanent errors like 404 fail immediately.

```python
from cropcatcher.sources import iiif

iiif.MAX_RETRIES = 5
iiif.BACKOFF_SECONDS = 2.0
```

## Cost and tuning

Each candidate is downloaded, described and matched as one unit of work,
spread across a thread pool. OpenCV releases the GIL during
`detectAndCompute`, `knnMatch` and `findHomography` ([RANSAC] fitting a
[homography]), so those threads give
real multi-core parallelism — no multiprocessing needed.

Measured on a 16-core machine, 30 canvases at `!1024,1024`:

| | Time |
|---|---|
| Downloading 30 canvases (8 workers) | 1.3 s |
| Matching them (1 thread) | 26 s |
| `search_iiif`, `max_workers=1` | 40.9 s |
| `search_iiif`, `max_workers=16` | **6.9 s** |

**Matching is the main bottleneck, not network access**. max_workers mainly controls CPU parallelism, so increasing it toward your available core count can improve performance.

However, more workers also mean more concurrent requests to the remote server, which may trigger rate limiting.

## Lower-level API

```python
from cropcatcher.sources.iiif import (
    resolve_manifest, image_api_url, load_iiif_image, load_iiif_images,
)

resources = resolve_manifest(manifest_url, max_canvases=20)
resources[0].canvas_label   # 'fol. 12r'
resources[0].service_id     # IIIF Image API service

image_api_url(service_id, size="!512,512")
image = load_iiif_image(service_id, size="!512,512")     # grayscale ndarray
images = load_iiif_images([...], max_workers=8)          # order preserved
```

`Index.add_iiif_services()` indexes a chosen list of services without
resolving a manifest — the primitive behind windowed and two-stage search.

## Inspecting one match

```python
debug = matcher.explain_iiif(query="illumination.jpg", service_id=best.source)
```

Re-downloads that single canvas and returns keypoints, matches and the
homography for [visualization](usage.md#visualizing-a-match).

[SIFT]: https://en.wikipedia.org/wiki/Scale-invariant_feature_transform
[ORB]: https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF
[RANSAC]: https://en.wikipedia.org/wiki/Random_sample_consensus
[homography]: https://en.wikipedia.org/wiki/Homography_(computer_vision)
[kdtree]: https://en.wikipedia.org/wiki/K-d_tree
