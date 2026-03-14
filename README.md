# GHOST
### General Hyperspectral Observation and Segmentation Toolkit

> *Get hyperspectral intelligence out of the box. No fine tuning. No augmentation. Near SOTA results.*

---

## What is GHOST?

GHOST is a **general-purpose hyperspectral image segmentation framework** that works out of the box on any `.mat` hyperspectral dataset — no dataset-specific preprocessing, no manual feature engineering, no fine-tuning required to get near state-of-the-art results.

**Feed it data. Get a segmentation map.**

The goal is the same out-of-box promise that nnUNet delivers for medical images or YOLO delivers for object detection — except for hyperspectral imagery across every domain: Earth observation, planetary science (Mars CRISM, asteroid Ryugu), medical imaging, materials science, and beyond.

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
    --out-dir runs/indian_pines \
    --title   "GHOST — Indian Pines"
```

---

## Results

All results use `--train_ratio 0.2 --val_ratio 0.1 --seed 42` stratified splits, forest routing.

| Dataset | Bands | Classes | Spatial | OA | mIoU | Dice | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| Indian Pines | 200 | 16 | 145×145 | **94.49%** | 0.7156 | 0.7399 | 0.7649 | 0.7308 |
| Salinas Valley | 204 | 16 | 512×217 | **92.10%** | 0.7387 | 0.7840 | 0.9287 | 0.7843 |
| Pavia University | 103 | 9 | 610×340 | **85.85%** | 0.6032 | 0.6685 | 0.8549 | 0.6797 |
| Mars CRISM | 107 | TBD | 195×640 | — | — | — | — | — |
| Asteroid Ryugu† | 7 | 4 | 1024×1024 | 54.65% | 0.3371 | — | — | — |

> Pavia, Salinas, and Ryugu trained at half filter capacity (`--base_filters 16 --num_filters 4`) on a 6GB VRAM consumer GPU.
> † Ryugu uses KMeans pseudo-labels — not supervised ground truth.

---

## Three Star Features

### 1. Continuum Removal — Physics-Informed Preprocessing

Raw reflectance varies enormously across sensors, illumination conditions, and atmospheric states. Continuum Removal normalises each pixel's spectrum by its convex hull envelope:

```
CR(λ) = spectrum(λ) / continuum(λ)
```

This strips brightness and isolates the **shape** of absorption features — the mineralogically and biologically meaningful signal. A geologist does this by hand. GHOST does it automatically, differentiably, for every pixel.

### 2. RSSP — Recursive Spectral Splitting with Parallel Forests

Standard flat classifiers treat all 16 classes as equally confusable. They are not. RSSP builds a binary tree by recursively splitting classes using **Spectral Angle Mapper (SAM)** distances on their mean spectra — spectrally distant classes are separated early, similar ones handled by a specialist model.

At each node, an independent **forest of HyperspectralNets** trains on only that node's class subset. Each model solves a simpler, more spectrally coherent problem. Rare minority classes that are crushed at the root level get proper specialist attention further down the tree.

### 3. SSSR — Selective Spectral State Routing

The classic failure of hierarchical classifiers: a wrong turn at the root is permanent. SSSR replaces hard argmax routing with **soft probabilistic routing**. A frozen spectral fingerprint encoder produces a compact spectral identity for each pixel. A lightweight routing head at each internal node produces `p_left ∈ (0, 1)`.

Pixels flow down **both** branches, weighted by confidence. The final prediction is a weighted sum across all leaf nodes. An uncertain pixel benefits from multiple subtrees rather than being permanently misrouted.

---

## Architecture Overview

```
RAW HYPERSPECTRAL IMAGE  (B, C, H, W)
         │
         ▼
┌─────────────────────────┐
│    Continuum Removal     │  Physics-informed · strips albedo/brightness
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Spectral 3D Conv Stack  │  kernel=(7,3,3) · models band adjacency
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│       SE Block           │  Learned per-channel attention
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   2D U-Net Encoder       │  4 scales · skip connections
│   2D U-Net Decoder       │  Full resolution prediction
└─────────────────────────┘

══════════════ RSSP ══════════════

SAM-based class tree (built from data)

        Root: all K classes
        /                \
   LEFT subtree      RIGHT subtree
  (spectrally         (spectrally
   coherent)           coherent)
    /      \            /      \
  ...      ...        ...      ...

  Each node = independent HyperspectralNet forest

══════════════ SSSR ══════════════

SpectralSSMEncoder (pretrained · frozen)
→ per-pixel spectral fingerprints

Per-node routing head:
  fingerprint → p_left ∈ (0,1)
  → soft weighted cascade inference
  → final pred = Σ_leaf (weight × prob)
```

---

## Inspiration

| Tool | What GHOST borrows |
|---|---|
| **nnUNet** | Auto-configuring architecture, dataset-agnostic design philosophy |
| **YOLO / Ultralytics** | One command → immediate usable results |
| **Random Forests** | Ensemble voting to reduce variance |
| **Mamba / SSMs** | Selective spectral sequence fingerprinting |
| **Geological spectroscopy** | Continuum Removal as physics-informed preprocessing |

---

## Repository Structure

```
GHOST/
├── ghost/
│   ├── models/
│   │   ├── hyperspectral_net.py    # Full pipeline model
│   │   ├── spectral_3d_block.py    # 3D conv spectral stack
│   │   ├── spectral_ssm.py         # SSSR fingerprint encoder
│   │   ├── se_block.py             # Squeeze-and-excitation
│   │   ├── encoder_2d.py           # U-Net encoder
│   │   └── decoder_2d.py           # U-Net decoder
│   ├── preprocessing/
│   │   └── continuum_removal.py    # Physics-informed preprocessing
│   ├── datasets/
│   │   └── hyperspectral_dataset.py  # Universal .mat loader
│   ├── rssp/
│   │   ├── sam_clustering.py       # SAM-based tree builder
│   │   ├── rssp_trainer.py         # Node + router training
│   │   ├── rssp_inference.py       # Soft cascade inference
│   │   ├── ssm_pretrain.py         # SSM encoder pretraining
│   │   └── sssr_router.py          # Per-node routing heads
│   ├── train.py                    # Flat model training
│   ├── train_rssp.py               # RSSP + SSSR training
│   ├── predict.py                  # Standalone inference
│   ├── visualize.py                # Segmentation visualizer
│   └── cli.py                      # `ghost` command router
├── docs/
│   ├── architecture.md             # Deep technical reference
│   ├── api_reference.md            # Full CLI + Python API docs
│   ├── how_to_use.md               # Step-by-step usage guide
│   └── roadmap.md                  # What's next
├── configs/
│   └── indian_pines.yaml           # Example config
├── setup.py
└── requirements.txt
```

---

## Hardware Requirements

| Setting | VRAM | Command flags |
|---|---|---|
| Minimum | 4GB | `--base_filters 16 --num_filters 4 --d_model 32` |
| Recommended | 6GB | `--base_filters 16 --num_filters 4` |
| Full capacity | 8GB+ | defaults |

Indian Pines at full capacity: ~60-70 minutes on RTX 3050.

---

## Documentation

| Document | Description |
|---|---|
| [Architecture](docs/architecture.md) | Every component explained — intuition, analogy, and mathematics |
| [API Reference](docs/api_reference.md) | All CLI flags and Python API with examples |
| [How to Use](docs/how_to_use.md) | Step-by-step from raw data to segmentation results |
| [Roadmap](docs/roadmap.md) | Near-term and long-term plans |

---

*Built at [Hackathon Name] · GHOST sees what others miss.*