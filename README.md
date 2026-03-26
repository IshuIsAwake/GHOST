# GHOST

### Generalizable Hyperspectral Observation & Segmentation Toolkit

Inspired by nnU-Net's philosophy of out-of-the-box segmentation — applied to hyperspectral imagery.

```bash
pip install ghost-hsi
ghost demo
```

---

## What is GHOST?

GHOST is a hyperspectral image segmentation framework. It takes any `.mat` hyperspectral dataset and produces a segmentation map — no code, no pipeline configuration, no PCA.

It is data-agnostic: band count, class count, and spatial dimensions are read from the file at runtime. The same binary has been tested on:

- **Indian Pines** — 200 bands, 16 classes (remote sensing)
- **Salinas Valley** — 204 bands, 16 classes (remote sensing)
- **Pavia University** — 103 bands, 9 classes (remote sensing)
- **LUSC** — 61 bands, 3 classes (lung cancer histopathology)
- **Mars CRISM** — planetary remote sensing

Zero code changes between any of these. All results reported here were obtained on a laptop with an RTX 3050 (6 GB VRAM).

See [results.md](results.md) for full numbers with caveats.

---

## Results Overview

### Indian Pines

| Config | OA | mIoU | Kappa | Time |
|--------|-----|------|-------|------|
| 32 base / 8 num filters | **97.55%** | 0.8027 | 0.9721 | 77m |

Full results including Salinas, Pavia, LUSC, and ablation studies are in [results.md](results.md).

---

## Architecture

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

See [architecture.md](architecture.md) for technical details.

---

## Quick Start

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

## Data Format

GHOST accepts `.mat` files (MATLAB/HDF5 format):
- **Data file:** 3D array `(H, W, Bands)` — the hyperspectral cube
- **Ground truth file:** 2D array `(H, W)` — integer class labels, 0 = background

Keys inside the `.mat` file are auto-detected by array dimensionality.

Standard datasets (Indian Pines, Pavia University, Salinas Valley) are available from [the GIC group at UPV/EHU](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | Pipeline components, SPT, training details |
| [API Reference](API_Reference.md) | All CLI commands and flags |
| [Results](results.md) | Full results, per-class IoU, caveats |
| [Limitations](limitations.md) | What doesn't work, honest assessment |
| [TODO](TODO.md) | Planned features |

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
