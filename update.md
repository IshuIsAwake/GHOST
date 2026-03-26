# GHOST — Release & Update Guide

## How to release a new version

### 1. Make your code changes

Work on features/fixes as normal. Test locally:

```bash
# Install from source in your dev environment
pip install -e .

# Run training, verify changes work
ghost train_rssp --data ... --gt ... --out-dir runs/test
```

### 2. Bump the version

Update the version in **both** files (they must match):

```
ghost/__init__.py    →  __version__ = "X.Y.Z"
pyproject.toml       →  version = "X.Y.Z"
```

> **PyPI rejects re-uploads of the same version.** Always bump before releasing.

Version convention:
- **Patch** (0.1.3 → 0.1.4): bug fixes, small improvements
- **Minor** (0.1.4 → 0.2.0): new features, new CLI commands
- **Major** (0.2.0 → 1.0.0): breaking changes, API redesign

### 3. Commit and push

```bash
git add -A
git commit -m "v0.X.Y: short description of changes"
git push origin main
```

### 4. Create a GitHub Release

This triggers the CI/CD pipeline that builds compiled wheels for all platforms.

**Option A: GitHub UI**
1. Go to https://github.com/IshuIsAwake/GHOST/releases/new
2. Click "Choose a tag" → type `v0.X.Y` → **"Create new tag on publish"**
3. Title: `v0.X.Y`
4. Description: what changed
5. Click **Publish release**

**Option B: CLI**
```bash
git tag v0.X.Y
git push origin v0.X.Y
gh release create v0.X.Y --title "v0.X.Y" --notes "Description of changes"
```

### 5. CI builds and publishes automatically

The GitHub Actions workflow (`.github/workflows/publish.yml`) will:
1. Build Cython-compiled wheels for Python 3.9–3.12 on Linux, macOS (ARM64), and Windows
2. Build a source distribution (sdist) as fallback for unsupported platforms
3. Upload everything to PyPI via trusted publishing (no API token needed)

Monitor progress at: https://github.com/IshuIsAwake/GHOST/actions

**Expected time:** ~8 minutes for all platforms.

### 6. Verify

```bash
# In a separate test environment (not your dev env)
conda activate ghost_test
pip install --upgrade ghost-hsi
ghost version
```

---

## If CI fails

### How to retry after fixing

You **cannot** re-publish a GitHub Release with the same tag and expect CI to re-trigger cleanly. Do this:

```bash
# 1. Delete the release on GitHub UI (Releases → v0.X.Y → Delete)

# 2. Delete the remote and local tag
git push origin --delete v0.X.Y
git tag -d v0.X.Y

# 3. Push your fix
git add -A
git commit -m "fix: description of what you fixed"
git push origin main

# 4. Recreate the tag and release
git tag v0.X.Y
git push origin v0.X.Y
# Then create the release on GitHub UI or:
gh release create v0.X.Y --title "v0.X.Y" --notes "Release notes"
```

### Common CI failures and fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Pure Python wheel was generated" | Cython missing from build environment | Ensure `cython>=3.0` is in `pyproject.toml` → `[build-system].requires` |
| "Configuration not supported" on macOS | GitHub deprecated the runner (e.g. macos-13) | Remove it from the matrix in `publish.yml` |
| Test step times out (600s+) | `CIBW_TEST_COMMAND` imports torch (~2GB install) | Keep test lightweight — import only `ghost` (no torch deps) |
| PyPI rejects upload | Version already exists on PyPI | Bump version, can't overwrite existing releases |

### Key thing to know about cibuildwheel

cibuildwheel creates **isolated PEP 517 build environments**. These only install packages listed in `pyproject.toml` → `[build-system].requires`. Installing something via `CIBW_BEFORE_BUILD` puts it in the host env, **not** the isolated build env. So build dependencies like Cython **must** go in `pyproject.toml`.

---

## Manual upload (fallback)

If CI is broken and you need to ship urgently:

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build sdist (pure Python fallback — no Cython)
python -m build --sdist

# Upload
python -m twine upload dist/*
# Username: __token__
# Password: <your PyPI API token>
```

> This uploads a source distribution only. Users get pure Python (no compiled .so). Use as a last resort.

---

## One-time setup: PyPI Trusted Publishing

This lets GitHub Actions upload to PyPI without storing API tokens. Already configured for this project.

1. **PyPI side:** https://pypi.org/manage/project/ghost-hsi/settings/publishing/
   - Owner: `IshuIsAwake`
   - Repository: `GHOST`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`

2. **GitHub side:** https://github.com/IshuIsAwake/GHOST/settings/environments
   - Environment called `pypi` exists

---

## Updating environments

### Local dev environment
```bash
cd ~/Projects/AI/GHOST
pip install -e .
# Changes to .py files take effect immediately — no reinstall needed
```

### Separate test environment
```bash
conda activate ghost_test
pip install --upgrade --force-reinstall ghost-hsi
ghost version
```

### Google Colab
```python
!pip install --upgrade ghost-hsi
!ghost version
```

---

## Checklist for every release

- [ ] Code changes tested locally
- [ ] Version bumped in `ghost/__init__.py` AND `pyproject.toml`
- [ ] Committed and pushed to `main`
- [ ] GitHub Release created (triggers CI)
- [ ] CI workflow passed (check Actions tab — all 3 jobs green)
- [ ] `pip install --upgrade ghost-hsi` shows new version
- [ ] `ghost version` prints correct version
