# GHOST — General Hyperspectral Out-of-the-box Segmentation Tool

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
| GHOST + RSSP | 0.9072 | 0.5900 |
| GHOST + RSSP + SSSR | *pending* | *pending* |

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
pip install ghost-hsi   # coming soon

# Train on any hyperspectral .mat file
python train_rssp.py \
    --data  path/to/data.mat \
    --gt    path/to/labels.mat

# Re-use a pretrained SSM encoder (skip SSM pretraining)
python train_rssp.py \
    --data     path/to/data.mat \
    --gt       path/to/labels.mat \
    --ssm_load ssm_pretrained.pt
```

See [HOW_TO_USE.md](HOW_TO_USE.md) for full usage instructions.

---

## Repository Structure

```
GHOST/
├── models/
│   ├── hyperspectral_net.py   # Full pipeline model
│   ├── spectral_3d_block.py   # 3D conv spectral stack
│   ├── spectral_ssm.py        # SSSR fingerprint encoder
│   ├── se_block.py            # Squeeze-and-excitation block
│   ├── encoder_2d.py          # U-Net encoder
│   └── decoder_2d.py          # U-Net decoder
├── preprocessing/
│   └── continuum_removal.py   # Physics-informed preprocessing
├── datasets/
│   └── hyperspectral_dataset.py  # Universal .mat loader
├── rssp/
│   ├── sam_clustering.py      # SAM-based tree builder
│   ├── rssp_trainer.py        # Node + router training
│   ├── rssp_inference.py      # Soft cascade inference
│   ├── ssm_pretrain.py        # SSM encoder pretraining
│   └── sssr_router.py         # Per-node routing heads
├── train.py                   # Flat model training
├── train_rssp.py              # RSSP + SSSR training
└── docs/
    ├── README.md
    ├── ARCHITECTURE.md
    ├── PARAMETERS.md
    └── HOW_TO_USE.md
```