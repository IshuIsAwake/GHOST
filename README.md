# GHOST
### Generalizable Hyperspectral Observation and Segmentation Toolkit

> *Under active development. Feedback and contributions welcome.*

---

## What is GHOST?

GHOST is a hyperspectral image segmentation framework with one goal: point it at any `.mat` hyperspectral dataset and get a segmentation map without writing code or configuring a pipeline.

It is not a state-of-the-art accuracy benchmark. It is a working, accessible baseline for hyperspectral segmentation — particularly useful for novel or non-standard datasets where no trained models exist.

---

## Two Things GHOST Does That Most Tools Don't

### 1. No PCA Required

Every major hyperspectral deep learning method requires PCA dimensionality reduction as a mandatory preprocessing step. This means choosing how many components to retain, verifying variance explained, and accepting that you've discarded spectral information the model deemed low-variance. For non-standard domains — planetary science, medical imaging, novel sensors — this is a significant barrier and a source of information loss.

GHOST uses Continuum Removal instead: a physics-informed normalisation that strips brightness variation and isolates absorption feature shape, preserving all spectral bands. No dimensionality choices. No information discarded.

### 2. Fully Data-Agnostic

Band count, class count, spatial dimensions, and data distribution are all read from the file at runtime. Nothing is hardcoded. A model that works on Indian Pines (200 bands, 16 classes, 145×145 pixels) runs without modification on Pavia University (103 bands, 9 classes, 610×340 pixels) or Mars CRISM spectra.

---

## Results

All results use `--train_ratio 0.2 --val_ratio 0.1 --seed 42` stratified splits, forest routing.

| Dataset | Bands | Classes | Spatial | OA | mIoU |
|---|---|---|---|---|---|
| Indian Pines | 200 | 16 | 145×145 | 94.49% | 0.7156 |
| Salinas Valley | 204 | 16 | 512×217 | 92.10% | 0.7387 |
| Pavia University | 103 | 9 | 610×340 | 85.85% | 0.6032 |
| Asteroid Ryugu† | 7 | 4 | 1024×1024 | 54.65% | 0.3371 |

> Pavia, Salinas, and Ryugu trained at half filter capacity (`--base_filters 16 --num_filters 4`) on a 6GB VRAM consumer GPU. Published SOTA methods report on different training splits and metrics — direct numerical comparison is not meaningful without matching protocols.

† Ryugu uses KMeans pseudo-labels, not supervised ground truth.

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/GHOST.git
cd GHOST
pip install -e .
```

```bash
# Train
ghost train_rssp \
    --data data/indian_pines/Indian_pines_corrected.mat \
    --gt   data/indian_pines/Indian_pines_gt.mat \
    --out-dir runs/indian_pines

# Predict
ghost predict \
    --data  data/indian_pines/Indian_pines_corrected.mat \
    --gt    data/indian_pines/Indian_pines_gt.mat \
    --model runs/indian_pines/rssp_models.pkl \
    --routing all \
    --out-dir runs/indian_pines

# Visualize
ghost visualize \
    --data    data/indian_pines/Indian_pines_corrected.mat \
    --gt      data/indian_pines/Indian_pines_gt.mat \
    --model   runs/indian_pines/rssp_models.pkl \
    --dataset indian_pines \
    --out-dir runs/indian_pines
```

---

## Hardware Requirements

| Setting | VRAM | Flags |
|---|---|---|
| Minimum | 4GB | `--base_filters 16 --num_filters 4 --d_model 32` |
| Recommended | 6GB | `--base_filters 16 --num_filters 4` |
| Full capacity | 8GB+ | defaults |

Indian Pines at full capacity: ~60-70 minutes on RTX 3050.

---

## Repository Structure

```
GHOST/
├── ghost/
│   ├── models/         # HyperspectralNet, 3D conv, SE block, U-Net
│   ├── preprocessing/  # Continuum removal
│   ├── datasets/       # Universal .mat loader
│   ├── rssp/           # Tree builder, trainer, inference, SSSR router
│   ├── train.py
│   ├── train_rssp.py
│   ├── predict.py
│   ├── visualize.py
│   └── cli.py
├── docs/
├── configs/
├── setup.py
└── requirements.txt
```

---

## Documentation

| Document | Description |
|---|---|
| [Architecture](docs/architecture.md) | Component overview |
| [API Reference](docs/api_reference.md) | All CLI flags and Python API |
| [How to Use](docs/how_to_use.md) | Step-by-step guide |

---

## Status

GHOST is under active development. Known limitations exist, particularly on spatially fine-grained scenes and class-imbalanced datasets. Contributions, issues, and feedback are welcome.