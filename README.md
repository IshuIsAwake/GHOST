# GHOST — General Hyperspectral Observation and Segmentation Tool

> *"We are not building a `dataset002_BraTS` or a `dataset004_Hippocampus`. We are building the nnUNet itself."*

---

## What is GHOST?

GHOST is a **general-purpose hyperspectral image segmentation framework** that works out of the box on any `.mat` hyperspectral dataset — no dataset-specific preprocessing, no manual feature engineering, no fine-tuning required to get near state-of-the-art results.

Feed it data. Get a segmentation map.

The goal is the same out-of-box promise that nnUNet delivers for medical images or YOLO delivers for object detection — except for hyperspectral imagery across every domain: Earth observation, planetary science (Mars CRISM, asteroid Ryugu), medical imaging (retinal scans, brain tissue), materials science, and beyond.

---

## Inspiration

| Tool | What GHOST borrows |
|---|---|
| **nnUNet** | Auto-configuring architecture, dataset-agnostic design philosophy |
| **YOLO / Ultralytics** | Simple install → immediate usable results without expertise |
| **Random Forests** | Ensemble voting to correct individual model errors |
| **Mamba / SSMs** | Selective state-space modelling for spectral sequence fingerprinting |
| **Physics** | Continuum Removal as a built-in, label-free preprocessing step |

---

## Architecture Overview

```
                        RAW HYPERSPECTRAL IMAGE
                              (B, C, H, W)
                                   │
                    ┌──────────────▼──────────────┐
                    │      Continuum Removal        │  ← Physics-informed
                    │  removes albedo/brightness    │    preprocessing
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     Spectral 3D Conv Stack    │  ← Local spectral +
                    │   (B, 1, C, H, W) → features │    spatial features
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │         SE Block              │  ← Channel attention
                    │   learned channel weighting   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      2D U-Net Encoder         │  ← Spatial context
                    │    + skip connections         │    at 4 scales
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      2D U-Net Decoder         │  ← Full resolution
                    │    segmentation map           │    prediction
                    └─────────────────────────────┘

                    ════════════════════════════════
                              RSSP LAYER
                    ════════════════════════════════

                         SAM-based class tree
                              (built once)
                         /                  \
                   LEFT subtree         RIGHT subtree
                 (spectrally similar)  (spectrally similar)
                  /           \           /          \
               ...            ...       ...          ...

                    Each node = independent forest
                    of the above HyperspectralNet

                    ════════════════════════════════
                              SSSR LAYER
                    ════════════════════════════════

               SpectralSSMEncoder (frozen after pretraining)
               produces per-pixel spectral fingerprints

               At each internal node, a tiny routing head
               (fingerprint → p_left) replaces hard argmax
               routing with soft probabilistic routing.

               Final prediction = weighted sum across all leaves.
```

---

## Three Star Features

### 1. Continuum Removal — Physics-Informed Preprocessing

Raw reflectance values vary enormously across sensors, illumination conditions, and atmospheric states. Continuum Removal normalises each pixel's spectrum by its convex hull envelope:

```
CR(λ) = spectrum(λ) / continuum(λ)
```

This removes brightness and isolates the **shape** of absorption features — the mineralogically and biologically meaningful signal. A geologist does this by hand. GHOST does it automatically for every pixel, every dataset, every sensor.

### 2. RSSP — Recursive Spectral Splitting with Parallel Forests

Standard flat classifiers treat all 16 (or 9, or 30) classes as equally confusable. They're not. RSSP builds a binary tree by recursively splitting classes using Spectral Angle Mapper (SAM) distances on their mean spectra — spectrally distant classes get separated early, spectrally similar ones are grouped and handled by a specialist model.

At each tree node, an independent **forest of HyperspectralNets** is trained on only that node's subset of classes. Inference cascades top-down.

Result: each model solves a simpler, more spectrally coherent problem.

### 3. SSSR — Selective Spectral State Routing

The classic failure mode of hierarchical classifiers: if the root routes a pixel to the wrong child, it's lost permanently. RSSP's forest partially addresses this — but routing is still hard argmax.

SSSR replaces hard routing with **soft probabilistic routing**. A frozen spectral fingerprint encoder (trained once on the dataset) produces a compact representation of each pixel's spectral identity. At each internal node, a lightweight routing head converts this fingerprint into `p_left ∈ (0,1)`.

Pixels flow down **both** branches, weighted by routing confidence. The final prediction is a weighted sum across all leaf nodes. A pixel the root is uncertain about (p_left = 0.55) still benefits from both subtrees rather than being committed to one.

---

## Results

All results use 20% train / 10% val / 70% test stratified split.

### Indian Pines (200 bands, 16 classes)

| Configuration | OA | mIoU |
|---|---|---|
| GHOST (no RSSP) | 0.8600 | 0.5100 |
| GHOST + RSSP (Forest routing) | 0.9422 | 0.6981 |
| GHOST + RSSP + SSSR (Hybrid routing) | 0.9038 | 0.6053 |

### Pavia University (103 bands, 9 classes)
*Note: run at half filter capacity due to 6GB VRAM constraint*

| Configuration | OA | mIoU |
|---|---|---|
| GHOST (no RSSP) | 0.6400 | 0.2200 |
| GHOST + RSSP | 0.7790 | 0.3960 |
| GHOST + RSSP + SSSR | *pending* | *pending* |

---

## Quick Start

```bash
pip install .

# 1. Train on any hyperspectral .mat file (Full RSSP + SSSR Hybrid Routing)
ghost train_rssp \
    --data data/indian_pines/Indian_pines_corrected.mat \
    --gt data/indian_pines/Indian_pines_gt.mat \
    --ssm_save ssm_ip_v2.pt \
    --ssm_epochs 300 \
    --epochs 300 \
    --forests 5 \
    --routing hybrid \
    --save rssp_models_v2.pkl \
    --out-dir runs/indian_pines

# 2. Re-use a pretrained SSM encoder & compare routing modes
# The predict command runs all routing modes by default if --routing is not provided or set to 'all'
ghost predict \
    --data data/indian_pines/Indian_pines_corrected.mat \
    --gt data/indian_pines/Indian_pines_gt.mat \
    --model runs/indian_pines/rssp_models_v2.pkl \
    --ssm_load runs/indian_pines/ssm_ip_v2.pt \
    --out-dir runs/indian_pines_test
```

See [HOW_TO_USE.md](HOW_TO_USE.md) for full usage instructions.

---

## Repository Structure

```
GHOST/
├── ghost/
│   ├── models/                # Full pipeline models (HyperspectralNet, spectral 3D blocks)
│   ├── preprocessing/         # Physics-informed preprocessing (Continuum Removal)
│   ├── datasets/              # Universal .mat loader (HyperspectralDataset)
│   ├── rssp/                  # Recursive Spectral Splitting and Routing modules
│   ├── train.py               # Flat model training entry point
│   ├── train_rssp.py          # RSSP + SSSR training entry point
│   ├── predict.py             # Inference entry point
│   └── cli.py                 # Terminal router for the `ghost` command
├── setup.py                   # Packaging metadata
└── docs/                      # Documentation
```
