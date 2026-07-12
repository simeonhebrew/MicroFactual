# Releasing MicroFactual

Releases are published to PyPI via **Trusted Publishing (OIDC)** — no API tokens
are stored anywhere. The `.github/workflows/release.yml` workflow builds the
distributions and publishes them:

- **push a `v*` tag** → publishes to **TestPyPI**
- **publish a GitHub Release** → publishes to **PyPI**

## One-time setup (maintainer, ~5 minutes)

You must configure the trusted publishers and GitHub environments once. This is
the only step that needs your PyPI account.

### 1. GitHub environments

Repo → **Settings → Environments** → create two environments:

- `pypi`
- `testpypi`

(Optionally add "Required reviewers" or a tag protection rule to `pypi` so a
production publish needs an approval.)

### 2. PyPI trusted publisher

On <https://pypi.org> → your account → **Publishing** → *Add a pending publisher*:

| Field | Value |
|-------|-------|
| PyPI Project Name | `microfactual` |
| Owner | `simeonhebrew` |
| Repository name | `MicroFactual` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

### 3. TestPyPI trusted publisher

Repeat on <https://test.pypi.org> with **Environment name** `testpypi`.

> Note: your earlier TestPyPI upload was under the name `microfactual-ml`. The
> canonical name is now `microfactual` (it matches `import microfactual`), so add
> the pending publisher for `microfactual`.

## Cutting a release

1. Bump the version — **single source of truth** is `microfactual.__version__` in
   `src/microfactual/__init__.py` (the build reads it via `hatch.version`).
2. Move the `## [Unreleased]` section in `CHANGELOG.md` to the new version with a
   date.
3. Commit and merge to `main`.
4. **Test the publish first:**
   ```bash
   git tag v0.2.0 && git push origin v0.2.0
   ```
   This runs the workflow → TestPyPI. Verify:
   ```bash
   pip install -i https://test.pypi.org/simple/ microfactual
   ```
5. **Publish for real:** create a **GitHub Release** for the tag (Releases → Draft
   a new release → choose the tag → Publish). This runs the workflow → PyPI.

## Notes

- The build is **backend = hatchling**; `uv build` produces the sdist + wheel.
- The sdist ships only the package + project metadata (no datasets/notebooks).
- The version is validated with `twine check` before any upload.
