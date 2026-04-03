# GHOST

### Generalizable Hyperspectral Observation & Segmentation Toolkit

> An attempt to generalize Hyperspectral Imaging.

GHOST is a general-purpose hyperspectral segmentation tool — point it at a hyperspectral image and get a segmentation map without writing dataset-specific code. Band count, class count, and spatial dimensions are read at runtime with no hardcoding. Loosely inspired by [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), though much simpler and narrower in scope.

**Supports Python 3.9 - 3.12**

```bash
pip install ghost-hsi
ghost demo
```

---

## Design Goals

| Goal | Status |
|------|--------|
| **Data Agnosticism** — band count, class count, spatial dims read at runtime | Achieved |
| **Band Count Agnosticism** — works on 3 to 400+ bands with identical pipeline | Achieved |
| **Sensor Agnosticism** — remote sensing, medical pathology, planetary science | Achieved |
| **Spectral-Only Context** — scene-to-scene transfer without spatial dependency | In progress (v0.2.x) |

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
  --routing forest --out-dir runs/my_experiment

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

### Converting other formats (v0.1.7+)

GHOST can convert ENVI, TIFF, GeoTIFF, and HDF5 files to `.mat`:

```bash
pip install ghost-hsi[convert]

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

---

## Docs

| Document | What's in it |
|----------|-------------|
| [Architecture](architecture.md) | How the pipeline works |
| [API Reference](API_Reference.md) | CLI commands and flags |
| [Commands](commands.md) | Usage examples for each dataset |

Website: [anakinskywalker0.github.io/GhostWEB](https://anakinskywalker0.github.io/GhostWEB/)

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
