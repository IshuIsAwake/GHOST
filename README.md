# GHOST

### Generalizable Hyperspectral Observation & Segmentation Toolkit

> **Work in progress** — this is a personal project I'm building to learn about hyperspectral image segmentation. APIs and CLI flags will probably change.

GHOST is my attempt at making a general-purpose hyperspectral segmentation tool — something where you point it at a `.mat` file and get a segmentation map without writing dataset-specific code. The idea is loosely inspired by [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), though GHOST is much simpler and narrower in scope.

```bash
pip install ghost-hsi
ghost demo
```

---

## What it does

You give it a hyperspectral `.mat` file and a ground truth file. It figures out the band count, class count, and spatial dimensions at runtime and trains a segmentation model.

I've tested it on a few datasets so far:

- **Indian Pines** — 200 bands, 16 classes (remote sensing)
- **Salinas Valley** — 204 bands, 16 classes (remote sensing)
- **Pavia University** — 103 bands, 9 classes (remote sensing)
- **LUSC** — 61 bands, 3 classes (lung cancer histopathology, single crop only)

Same code for all of these, no changes between runs. That's the part I'm most interested in — whether this generalizes.

---

## Numbers so far

| Dataset | Config | OA | mIoU | Kappa | Hardware | Time |
|---------|--------|-----|------|-------|----------|------|
| LUSC | 32 / 8 | 99.42% | 0.9263 | 0.9876 | RTX 3050 (laptop) | 1h 8m |
| Salinas Valley | 32 / 8 | 98.69% | 0.9577 | 0.9855 | Kaggle T4 | 10h 51m |
| Indian Pines | 64 / 16 | 98.16% | 0.9071 | 0.9790 | RTX 3050 (laptop) | 2h 20m |
| Pavia University | 32 / 8 | 97.47% | 0.9531 | 0.9667 | Kaggle T4 | 7h 29m |
| Indian Pines | 32 / 8 | 97.20% | 0.8030 | 0.9681 | RTX 3050 (laptop) | 1h 17m |

**Take these with a grain of salt.** The evaluation setup is basic (pixel-level train/test split on a single scene), which is standard for these benchmarks but doesn't tell you much about real-world generalization. The LUSC result especially is from a single 512x512 crop and is not comparable to published benchmarks — see [results.md](results.md) for the full caveats.

Config = base_filters / num_filters. All runs use ce+dice loss and ensemble routing.

---

## How it works (roughly)

```
.mat file (H, W, Bands)
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

The SPT (Spectral Partition Tree) is the most interesting part to me — it recursively splits classes into groups based on spectral similarity (using SAM distance), and trains separate model ensembles for each group. This seems to help a lot with class imbalance, though I haven't done rigorous ablations yet beyond Indian Pines.

See [architecture.md](architecture.md) if you want the details.

---

## Quick start

```bash
# Install
pip install ghost-hsi

# See bundled dataset paths and example command
ghost demo

# Train with Spectral Partition Tree
ghost train_spt \
  --data data.mat --gt labels.mat \
  --loss dice \
  --base_filters 32 --num_filters 8 \
  --ensembles 5 --leaf_ensembles 3 \
  --epochs 400 --patience 50 --min_epochs 40 \
  --out-dir runs/my_experiment

# Predict
ghost predict \
  --data data.mat --gt labels.mat \
  --model runs/my_experiment/spt_models.pkl \
  --out-dir runs/my_experiment

# Visualize
ghost visualize \
  --data data.mat --gt labels.mat \
  --model runs/my_experiment/spt_models.pkl \
  --out-dir runs/my_experiment
```

A flat baseline (no SPT) is available via `ghost train`.

---

## Data format

GHOST accepts `.mat` files (MATLAB/HDF5 format):
- **Data file:** 3D array `(H, W, Bands)` — the hyperspectral cube
- **Ground truth file:** 2D array `(H, W)` — integer class labels, 0 = background

Keys inside the `.mat` file are auto-detected by array dimensionality.

Standard datasets (Indian Pines, Pavia University, Salinas Valley) are available from [the GIC group at UPV/EHU](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).

Only `.mat` is supported right now. ENVI, GeoTIFF, etc. need manual conversion — see [limitations.md](limitations.md).

---

## Docs

| Document | What's in it |
|----------|-------------|
| [Architecture](architecture.md) | How the pipeline works |
| [API Reference](API_Reference.md) | CLI commands and flags |
| [Results](results.md) | Full numbers, per-class breakdowns, caveats |
| [Limitations](limitations.md) | What doesn't work |
| [TODO](TODO.md) | Things I want to add eventually |

---

## License

Proprietary. All rights reserved. See [LICENSE](LICENSE).

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
