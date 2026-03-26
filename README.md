# GHOST

### Generalizable Hyperspectral Observation & Segmentation Toolkit

> **97% OA | 0.80 mIoU on Indian Pines — trained on a laptop GPU in 77 minutes.**

```bash
pip install ghost-hsi
```

---

## What is GHOST?

GHOST is a hyperspectral image segmentation framework. Point it at any `.mat` hyperspectral dataset and get a segmentation map — no code, no pipeline configuration, no PCA.

It runs on consumer hardware (RTX 3050, 6 GB VRAM) and handles any band count, class count, or spatial resolution automatically.

```bash
ghost train_rssp \
  --data data.mat --gt labels.mat \
  --routing forest --loss dice \
  --out-dir runs/my_experiment

ghost predict \
  --data data.mat --gt labels.mat \
  --model runs/my_experiment/rssp_models.pkl \
  --routing forest --out-dir runs/my_experiment
```

---

## Why GHOST?

### No PCA Required

Every major hyperspectral deep learning method requires PCA as a preprocessing step — choosing how many components to retain, accepting discarded spectral information. For non-standard domains (planetary science, medical imaging, novel sensors), this is a barrier and a source of information loss.

GHOST uses **Continuum Removal**: a physics-informed normalisation that strips brightness variation and isolates absorption feature shape. All spectral bands preserved. No dimensionality choices.

### Fully Data-Agnostic

Band count, class count, spatial dimensions — all read from the file at runtime. Nothing is hardcoded. The same binary that segments Indian Pines (200 bands, 16 classes) also handles lung cancer pathology slides (61 bands, 3 classes) and Mars CRISM data — with **zero code changes**.

### Runs on Consumer Hardware

Full training on a 6 GB laptop GPU. No A100s, no cloud compute, no multi-GPU setups. Designed for researchers who don't have institutional compute access.

---

## Results

All results: `--train_ratio 0.2 --val_ratio 0.1 --seed 42`, forest routing, NVIDIA RTX 3050 (6 GB).

### Indian Pines

| Config | OA | mIoU | Dice | Kappa | Time |
|--------|-----|------|------|-------|------|
| 64 base / 32 num filters | **97.52%** | **0.8593** | 0.9038 | 0.9717 | 6h 2m |
| 32 base / 8 num filters | **97.55%** | 0.8027 | 0.8391 | 0.9721 | 77m |

<details>
<summary>Per-class IoU (64/32 config)</summary>

| Class | IoU | Class | IoU |
|-------|------|-------|------|
| 1 | 0.9143 | 9 | 0.1905 |
| 2 | 0.9351 | 10 | 0.9547 |
| 3 | 0.9425 | 11 | 0.9689 |
| 4 | 0.8811 | 12 | 0.9607 |
| 5 | 0.9415 | 13 | 0.9592 |
| 6 | 0.9961 | 14 | 0.9694 |
| 7 | 0.4286 | 15 | 0.9890 |
| 8 | 0.9256 | 16 | 0.7910 |

Classes 7 and 9 have <30 training samples. Low IoU on these is expected and consistent with published literature.

</details>

### Salinas Valley

| Config | OA | mIoU | Dice | Kappa | Time |
|--------|-----|------|------|-------|------|
| 16 base / 4 num filters | 92.4% | 0.7668 | 0.8276 | 0.9154 | 4h 11m |

### Pavia University

> *Results pending re-run with latest pipeline. Space reserved.*

| Config | OA | mIoU | Dice | Kappa | Time |
|--------|-----|------|------|-------|------|
| 16 base / 4 num filters | — | — | — | — | — |

### LUSC (Lung Squamous Cell Carcinoma)

> **Single image only (1 of 62). Not comparable to published benchmarks.**
> Trained on a 512x512 crop with same-region pixel split. Published methods use patient-level cross-validation across 10 patients. These numbers demonstrate data-agnostic capability, not SOTA claims.

| Metric | Value |
|--------|-------|
| OA | 99.43% |
| mIoU | 88.96% |
| Dice | 93.49% |
| Kappa | 0.9878 |

See [LUSC_ghost_report.md](LUSC_ghost_report.md) for full details on why these results are not directly comparable.

### Mars CRISM / Asteroid Ryugu

Tested on planetary remote sensing data. Results in early exploration phase — not benchmarked against published methods.

---

## Quick Start

### Install

```bash
pip install ghost-hsi
```

### Train

```bash
ghost train_rssp \
  --data data/indian_pines/Indian_pines_corrected.mat \
  --gt   data/indian_pines/Indian_pines_gt.mat \
  --loss dice --routing forest \
  --base_filters 32 --num_filters 8 \
  --forests 5 --leaf_forests 3 \
  --epochs 400 --patience 50 --min_epochs 40 \
  --out-dir runs/indian_pines
```

### Predict

```bash
ghost predict \
  --data  data/indian_pines/Indian_pines_corrected.mat \
  --gt    data/indian_pines/Indian_pines_gt.mat \
  --model runs/indian_pines/rssp_models.pkl \
  --routing forest --out-dir runs/indian_pines
```

### Visualize

```bash
ghost visualize \
  --data    data/indian_pines/Indian_pines_corrected.mat \
  --gt      data/indian_pines/Indian_pines_gt.mat \
  --model   runs/indian_pines/rssp_models.pkl \
  --dataset indian_pines --routing forest \
  --out-dir runs/indian_pines
```

---


## Data Format

GHOST accepts `.mat` files (MATLAB/HDF5 format):
- **Data file:** 3D array with shape `(H, W, Bands)` — the hyperspectral cube
- **Ground truth file:** 2D array with shape `(H, W)` — integer class labels, 0 = background

Keys inside the `.mat` file are auto-detected by array dimensionality. No configuration needed.

Standard datasets (Indian Pines, Pavia University, Salinas Valley) are available from [the GIC group at UPV/EHU](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).

---

## Architecture Overview

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
RSSP Tree ------------- recursive spectral spatial splitting
    |                   each node: independent forest ensemble
    v
Prediction Map (H, W)
```

See [architecture.md](architecture.md) for technical details.

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | Pipeline components, RSSP tree, training details |
| [API Reference](API_Reference.md) | All CLI commands and flags |
| [TODO](TODO.md) | Roadmap, known limitations, planned features |

---

## License

Proprietary. All rights reserved. See [LICENSE](LICENSE).

For source code access, research collaborations, or licensing inquiries, contact the author directly.

---

## Citation

If you use GHOST in your research, please cite:

```
@software{ghost2026,
  title  = {GHOST: Generalizable Hyperspectral Observation \& Segmentation Toolkit},
  author = {IshuIsAwake},
  year   = {2026},
  url    = {https://pypi.org/project/ghost-hsi/}
}
```
