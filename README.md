# GHOST

### Generalizable Hyperspectral Observation & Segmentation Toolkit

> An attempt to generalize Hyperspectral Imaging.

GHOST is a general-purpose hyperspectral segmentation tool — point it at a hyperspectral image and get a segmentation map without writing dataset-specific code. Band count, class count, and spatial dimensions are read at runtime with no hardcoding. Loosely inspired by [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), though much simpler and narrower in scope.

**Supports Python 3.9 - 3.12**

```bash
pip install git+https://github.com/IshuIsAwake/GHOST.git
ghost demo
```

Version 0.2.0 is on GitHub only for now. The command above builds it with Cython, so it needs a C compiler.
`pip install ghost-hsi` still installs 0.1.7 from PyPI, which has the v0.1 pipeline only.

---

## Design Goals

| Goal | Status |
|------|--------|
| **Data Agnosticism** — band count, class count, spatial dims read at runtime | Achieved |
| **Band Count Agnosticism** — any band count runs through the identical pipeline | By design; results measured from 61 bands (LUSC) up |
| **Sensor Agnosticism** — remote sensing, medical pathology, planetary science | Achieved |
| **Spectral-Only Context** — scene-to-scene transfer without spatial dependency | v0.2.0 shipped; single-scene benchmark done, cross-scene test pending |

---

## Numbers so far

| Dataset | Config | OA | mIoU | Kappa | Hardware | Time |
|---------|--------|-----|------|-------|----------|------|
| LUSC | 32 / 8 | 99.42% | 0.9263 | 0.9876 | RTX 3050 (laptop) | 1h 8m |
| Salinas Valley | 32 / 8 | 98.69% | 0.9577 | 0.9855 | Kaggle T4 | 10h 51m |
| Indian Pines | 64 / 16 | 98.16% | 0.9071 | 0.9790 | RTX 3050 (laptop) | 2h 20m |
| Pavia University | 32 / 8 | 97.47% | 0.9531 | 0.9667 | Kaggle T4 | 7h 29m |
| Indian Pines | 32 / 8 | 97.20% | 0.8030 | 0.9681 | RTX 3050 (laptop) | 1h 17m |
| Mars CRISM | 32 / 8 | 71.70% | 0.5228 | 0.6829 | Kaggle T4 | 6h 44m |

Config = base_filters / num_filters. All runs use ce+dice loss and ensemble routing. Roughly +/-1% variance between runs due to random splits and seed sensitivity.

**Caveats:** Evaluation is pixel-level train/test split on a single scene, standard for these benchmarks but limited for real-world generalization. LUSC is a single 512x512 crop. Mars CRISM ground truth is extremely sparse and noisy.

---

## v0.2.0 — per-pixel pipeline

v0.1's U-Net sees neighbouring pixels, so it learns the layout of the training scene. Architecture 0.2.0
classifies every pixel from its own spectrum:

```
Any format (.mat, .hdr, .tif, .h5)
    |
    v
Continuum removal ---- once per scene, on the raw spectra
    |                  full (≥64 bands): upper convex hull of the smoothed spectrum
    |                  simple (3–63): line from the first to the last band
    v
1-D dilated ResNet --- per pixel; blocks are dropped when their reach exceeds the band count
    |
    v
MLP head ------------- class per pixel → prediction map (H, W)
```

It is the default for `ghost train`; `--arch 0.1.7` runs the v0.1 pipeline, and `ghost train_spt` stays on
0.1.7 until SPT is ported. Predict works without labels.

On Indian Pines (5 seeds, same pixels for every method), 0.2.0 reaches **73.9 ± 1.4% OA** with 50 labelled
pixels per class and **87.2 ± 0.5%** with v0.1's 20% split. That is about 7 points above an SVM or random
forest. It is level with v0.1's flat U-Net on OA and 21 points above it on mean per-class accuracy, without
using neighbouring pixels. Details and the protocol are in [benchmarks/](benchmarks/); a cross-scene test
is still to come.

---

## How it works (v0.1.x)

```
Hyperspectral Image (H, W, Bands)
    |
    v
Continuum Removal ---- physics-based normalisation, no PCA
    |
    v
Spectral 3D Conv ----- learns cross-band features, kernel (7,3,3)
    |
    v
SE Attention ---------- per-channel importance weighting
    |
    v
2D U-Net -------------- multi-scale spatial context
    |
    v
SPT ------------------- Spectral Partition Tree
    |                   each node: independent model ensemble
    v
Prediction Map (H, W)
```

The SPT (Spectral Partition Tree) recursively splits classes into groups based on spectral similarity (using SAM distance), and trains separate model ensembles for each group. This helps significantly with class imbalance.

See [architecture.md](architecture.md) for full details.

---

## Quick start

```bash
# Install 0.2.0 from GitHub (PyPI's ghost-hsi is still 0.1.7)
pip install git+https://github.com/IshuIsAwake/GHOST.git

# See bundled dataset paths and example command
ghost demo

# Train (architecture 0.2.0, per-pixel)
ghost train --data data.mat --gt labels.mat --loss dice --out-dir runs/my_experiment

# Segment any scene with the same sensor, with or without labels
ghost predict --model runs/my_experiment/ghost_model.pt --data scene.tif --out-dir runs/my_experiment
ghost visualize --model runs/my_experiment/ghost_model.pt --data data.mat --gt labels.mat \
  --out-dir runs/my_experiment

# v0.1.7: train with the Spectral Partition Tree
ghost train_spt \
  --data data.mat --gt labels.mat \
  --loss dice \
  --base_filters 32 --num_filters 8 \
  --ensembles 5 --leaf_ensembles 3 \
  --epochs 400 --patience 50 --min_epochs 40 \
  --out-dir runs/my_experiment

# Predict with an SPT model
ghost predict \
  --data data.mat --gt labels.mat \
  --model runs/my_experiment/spt_models.pkl \
  --routing forest --out-dir runs/my_experiment

# Visualize
ghost visualize \
  --data data.mat --gt labels.mat \
  --model runs/my_experiment/spt_models.pkl \
  --out-dir runs/my_experiment
```

The v0.1 flat baseline (no SPT) is `ghost train --arch 0.1.7`. `ghost version` lists the architectures.

---

## Data format

- **Data file:** a cube read as `(H, W, Bands)`
- **Ground truth file:** integer class labels `(H, W)`, 0 = unlabelled

Architecture 0.2.0 reads `.mat` (including MATLAB v7.3), ENVI `.hdr`, TIFF/GeoTIFF and `.h5` directly. The
non-`.mat` readers need the `convert` extra:
`pip install "ghost-hsi[convert] @ git+https://github.com/IshuIsAwake/GHOST.git"`. Pixels with a missing or
all-zero spectrum are skipped and appear as 0 in prediction maps. The v0.1.7 pipeline reads `.mat` only.

### Converting to .mat

`convert_to_mat` still converts ENVI, TIFF, GeoTIFF and HDF5 files to `.mat`:

```bash
pip install "ghost-hsi[convert] @ git+https://github.com/IshuIsAwake/GHOST.git"

ghost convert_to_mat \
  --img image.hdr \
  --gt  labels.tif \
  --out-dir converted/
```

All metadata is preserved in a `metadata.json` sidecar file. Optional spatial cropping via `--crop Y X H W`. See [API Reference](API_Reference.md) for full details.

Standard datasets (Indian Pines, Pavia University, Salinas Valley) are available from [the GIC group at UPV/EHU](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).

---

## Known limitations (v0.1.x)

- **Spatial dependence:** U-Net processes neighbouring pixels; models don't reliably transfer across scenes
- **No transfer learning:** Each dataset requires full retraining
- **Single-scene constraint:** Training and inference on same/identical-condition scenes only
- **SSSR router non-functional:** Use `--routing forest` (default and recommended)

## Known limitations (v0.2.0)

- **No SPT or CRF yet:** both are planned for 0.2.1
- **Single scene:** trains on one labelled scene at a time
- **Memory:** the whole scene is loaded and preprocessed at once

---

## Docs

| Document | What's in it |
|----------|-------------|
| [Architecture](architecture.md) | How the pipeline works |
| [API Reference](API_Reference.md) | CLI commands and flags |
| [Benchmarks](benchmarks/) | Indian Pines protocol for 0.2.0 |

Website: [anakinskywalker0.github.io/GhostWEB](https://anakinskywalker0.github.io/GhostWEB/)

---

## License

MIT. See [LICENSE](LICENSE).

---

## Citation

```
@software{ghost2026,
  title  = {GHOST: Generalizable Hyperspectral Observation \& Segmentation Toolkit},
  author = {IshuIsAwake},
  year   = {2026},
  url    = {https://pypi.org/project/ghost-hsi/}
}
```
