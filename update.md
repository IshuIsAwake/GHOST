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

This triggers the CI/CD pipeline that builds wheels for all platforms.

**Option A: GitHub UI**
1. Go to https://github.com/IshuIsAwake/GHOST/releases/new
2. Click "Choose a tag" → type `v0.X.Y` → "Create new tag on publish"
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
1. Build compiled wheels for Python 3.9, 3.10, 3.11, 3.12 on Linux, macOS, Windows
2. Build a source distribution (sdist) as fallback
3. Run import tests on each wheel
4. Upload everything to PyPI via trusted publishing

Monitor progress at: https://github.com/IshuIsAwake/GHOST/actions

### 6. Verify

```bash
pip install --upgrade ghost-hsi
ghost --version
```

---

## Manual upload (fallback if CI is not set up)

If the GitHub Actions workflow isn't configured yet:

```bash
# Clean
rm -rf dist/ build/ *.egg-info

# Build sdist (works on all Python versions, pure Python fallback)
python -m build --sdist

# Upload
python -m twine upload dist/*
# Username: __token__
# Password: <your PyPI API token>
```

---

## One-time setup: PyPI Trusted Publishing

This lets GitHub Actions upload to PyPI without storing API tokens.

1. **PyPI side:** Go to https://pypi.org/manage/project/ghost-hsi/settings/publishing/
   - Owner: `IshuIsAwake`
   - Repository: `GHOST`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`

2. **GitHub side:** Go to https://github.com/IshuIsAwake/GHOST/settings/environments
   - Create an environment called `pypi`

---

## Updating your local dev environment

```bash
# If installed in editable mode (recommended for development)
cd ~/Projects/AI/GHOST
pip install -e .

# Changes to .py files take effect immediately — no reinstall needed
```

## Updating a separate test environment

```bash
conda activate ghost_test
pip install --upgrade --force-reinstall ghost-hsi
ghost --version
```

## Updating on Google Colab

```python
!pip install --upgrade ghost-hsi
import ghost
print(ghost.__version__)
```

---

## Project structure (key files)

```
GHOST/
├── ghost/
│   ├── __init__.py              # Version string (__version__)
│   ├── cli.py                   # CLI entry point (ghost command)
│   ├── train.py                 # Single-model training
│   ├── train_rssp.py            # RSSP forest ensemble training
│   ├── predict.py               # Inference from saved model
│   ├── visualize.py             # Generate prediction maps
│   ├── datasets/                # Data loading
│   ├── models/                  # Neural network architectures
│   ├── preprocessing/           # Continuum removal, etc.
│   ├── rssp/                    # RSSP tree, trainer, inference, SSM
│   └── utils/                   # Display, logging utilities
├── pyproject.toml               # Package metadata + version
├── setup.py                     # Cython build (optional, CI uses it)
├── MANIFEST.in                  # sdist file inclusion rules
└── .github/workflows/
    └── publish.yml              # CI/CD: build wheels + publish to PyPI
```

---

## Checklist for every release

- [ ] Code changes tested locally
- [ ] Version bumped in `ghost/__init__.py` AND `pyproject.toml`
- [ ] Committed and pushed to `main`
- [ ] GitHub Release created (triggers CI)
- [ ] CI workflow passed (check Actions tab)
- [ ] `pip install --upgrade ghost-hsi` works
- [ ] `ghost --version` shows new version
- [ ] Tested on Colab: `!pip install ghost-hsi && !ghost --version`
