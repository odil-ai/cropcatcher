# Release checklist

How to cut a CropCatcher release, and what to verify before doing so.

Publishing is automated: creating a **GitHub release** triggers
[`.github/workflows/release.yml`](.github/workflows/release.yml), which
re-runs the checks, builds the distributions and uploads them to PyPI. A tag
alone does **not** publish.

---

## 0. One-time setup (first release only)

- [ ] **PyPI trusted publishing.** The project does not exist on PyPI yet, so
      create a *pending publisher* at
      <https://pypi.org/manage/account/publishing/> with the values in the
      table below. The environment name must match the workflow exactly, or
      the upload is rejected. No API token is stored anywhere.

| Field | Value |
|---|---|
| PyPI project name | `cropcatcher` |
| Owner / repository | your GitHub org / repo |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

- [ ] **GitHub Pages.** Settings → Pages → Source: **GitHub Actions**. Do this
      before the documentation workflow first runs, otherwise the deploy step
      fails.
- [ ] **`repo_url`** filled in [`mkdocs.yml`](mkdocs.yml) (currently a
      placeholder).
- [ ] **Name still free** on PyPI: `curl -s -o /dev/null -w '%{http_code}\n'
      https://pypi.org/pypi/cropcatcher/json` returns `404`.

---

## 1. Version bump

The version string lives in **four** places and they must agree. The release
workflow aborts if the git tag does not match `pyproject.toml`.

- [ ] [`pyproject.toml`](pyproject.toml) — `version = "X.Y.Z"`
- [ ] [`CITATION.cff`](CITATION.cff) — `version: X.Y.Z`
- [ ] [`README.md`](README.md) — `version = {X.Y.Z}` in the BibTeX block
- [ ] [`docs/index.md`](docs/index.md) — same BibTeX block
- [ ] [`CITATION.cff`](CITATION.cff) — `date-released:` set to the release date

```bash
grep -rn "version" pyproject.toml CITATION.cff | grep -E "^\S+:[0-9]+:version"
```

---

## 2. Pre-flight checks

Everything below is what CI will run. Running it locally first avoids a failed
release.

- [ ] **Lockfile current** — CI installs with `--locked` and fails on drift.
      ```bash
      uv lock --check
      ```
- [ ] **Lint and format**
      ```bash
      uv run ruff check .
      uv run ruff format --check .
      ```
- [ ] **Tests**
      ```bash
      uv run pytest -q
      ```
- [ ] **Documentation builds strictly** — broken links and bad references are
      errors.
      ```bash
      uv sync --group docs && uv run mkdocs build --strict
      ```
- [ ] **Distributions build and their metadata renders**
      ```bash
      uv build && uvx twine check dist/*
      ```

---

## 3. What must *not* ship

The source distribution is scoped by `[tool.hatch.build.targets.sdist]`.
Verify it before every release, since a new top-level directory is included
only if listed there, but a new file inside `tests/` or `scripts/` ships
automatically.

- [ ] **No local working data, notebooks, or rendered docs** in the sdist:
      ```bash
      tar tzf dist/*.tar.gz | grep -E "/(data|notebooks|docs|site)/" || echo "clean"
      ```
- [ ] **Sdist stays small** (roughly 1.3 MB; a jump means something large crept
      in):
      ```bash
      ls -lh dist/
      ```
- [ ] **Wheel contains only the package**:
      ```bash
      unzip -l dist/*.whl | grep -v "^  *[0-9].*cropcatcher/" | head
      ```
- [ ] **Nothing ignored is referenced by shipped files.** `data/`,
      `notebooks/reconciliation.ipynb` and `notebooks/test.ipynb` are
      gitignored, so links to them from the README or the docs are dead links
      once published.

---

## 4. Cut the release

- [ ] Working tree clean, everything pushed to the default branch.
- [ ] CI green on that commit.
- [ ] Tag and push:
      ```bash
      git tag -a vX.Y.Z -m "CropCatcher X.Y.Z"
      git push origin vX.Y.Z
      ```
- [ ] Create the **GitHub release** from that tag, with notes. Publishing it is
      what starts the upload to PyPI.

---

## 5. After publishing

- [ ] Release workflow green, both jobs.
- [ ] Package visible at <https://pypi.org/project/cropcatcher/>.
- [ ] Clean-environment install actually works:
      ```bash
      uv run --isolated --no-project --with cropcatcher \
        python -c "from cropcatcher import Matcher, Index; print('ok')"
      ```
- [ ] Documentation deployed and the new version visible on GitHub Pages.
- [ ] **Drop the pre-publication notices**, which are now false:
      - the `pypi-not published` badge in `README.md` and `docs/index.md`,
        replaceable with `https://img.shields.io/pypi/v/cropcatcher.svg`
      - the "Not published yet" warning in `README.md` and `docs/index.md`
- [ ] Consider a `CHANGELOG.md` entry if one exists by then.

---

## Notes

- **PyPI uploads are immutable.** A version number can never be reused, even
  after deleting the release. Bump and re-release instead.
- **Test first if unsure.** Point the publish step at TestPyPI by adding
  `repository-url: https://test.pypi.org/legacy/` to the
  `pypa/gh-action-pypi-publish` step, with a matching pending publisher there.
- **Notebooks and Markdown are excluded from ruff** (see
  `[tool.ruff] extend-exclude`), so CI does not gate on them. Check notebooks
  by running them.
