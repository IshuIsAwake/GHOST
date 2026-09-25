# GHOST API Reference

All commands via the `ghost` CLI. Install: `pip install ghost-hsi`

```bash
ghost <command> [arguments]
```

| Command | Description |
|---------|-------------|
| `ghost train` | Train a model. Architecture 0.2.0 (per-pixel) by default; `--arch 0.1.7` for the v0.1 flat U-Net |
| `ghost train_spt` | v0.1.7 pipeline: Spectral Partition Tree + ensembles (architecture 0.1.7 only) |
| `ghost predict` | Segment a scene; score it when labels are given |
| `ghost visualize` | Generate a segmentation figure |
| `ghost convert_to_mat` | Convert ENVI / TIFF / GeoTIFF / HDF5 to `.mat` format |
| `ghost demo` | Show bundled dataset paths and example commands |
| `ghost version` | Print the version and the architectures it ships |
| `ghost flower` | Easter egg |

---

## Choosing the architecture (`--arch`)

One install ships two architectures: `0.1.7` (3-D conv + U-Net, SPT) and `0.2.0` (per-pixel continuum
removal + 1-D dilated ResNet, the default). Give the exact version, with or without the `v`.

| Command | How the architecture is chosen |
|---------|--------------------------------|
| `train` | `--arch`, default `0.2.0` |
| `train_spt` | Always `0.1.7`; `--arch 0.2.0` is an error until SPT is ported |
| `predict`, `visualize` | Read from the `--model` checkpoint; `--arch`, if given, must match it |

---

## ghost train (architecture 0.2.0, default)

Per-pixel training on one labelled scene. Continuum removal runs once on the raw spectra; each pixel is then
classified from its spectrum alone.

```bash
ghost train --data <cube> --gt <labels> [options]
```

Accepts `.mat` (including MATLAB v7.3), ENVI `.hdr`, `.tif`/GeoTIFF and `.h5` for both files.

| Flag | Default | Description |
|------|---------|-------------|
| `--cr` | `auto` | Continuum removal: `auto` (full ≥64 bands, simple 3–63, off <3), `full`, `simple`, `off`, `none` (scene z-score, for ablations) |
| `--split` | `ratio` | `ratio`: v0.1's per-class split (same pixels for the same seed); `fixed`: N per class; `disjoint`: spatial blocks |
| `--train_ratio` / `--val_ratio` | `0.2` / `0.1` | ratio and disjoint splits. For `fixed`, `--val_ratio` is the share of the non-training pixels |
| `--samples_per_class` / `--minority_samples` | `50` / `15` | fixed split: training pixels per class, and for classes smaller than that |
| `--block_size` | `H/10` | disjoint split: block side in pixels |
| `--channels`, `--embed_dim`, `--depth`, `--kernel_size` | `64`, `128`, `5`, `7` | Encoder; blocks are dropped while their receptive field would exceed the band count |
| `--head_hidden`, `--dropout` | `128`, `0.3` | MLP head |
| `--pool` | `avg` | `avg` over bands, or `flatten` to keep band positions |
| `--epochs`, `--patience`, `--min_epochs` | `300`, `50`, `40` | Early stop on validation mIoU |
| `--batch_size`, `--lr`, `--weight_decay` | `256`, `1e-3`, `1e-4` | AdamW; LR halves when validation loss plateaus |
| `--loss` | `ce` | `ce`, `squared_ce`, `focal`, `dice` (0.5·CE + 0.5·Dice, as in v0.1) |
| `--seed` | `42` | Seeds the split and the initialisation |
| `--device` | `auto` | `auto`, `cpu`, `cuda` |
| `--out-dir`, `--save`, `--log` | `.`, `ghost_model.pt`, `training_log.csv` | Outputs |

| Output | Description |
|--------|-------------|
| `ghost_model.pt` | Checkpoint: weights, preprocessing, split indices and a fingerprint of the training scene |
| `training_log.csv` | One row per epoch |
| `test_results.csv` | Test metrics, same columns as v0.1 |
| `class_report.csv` | Per-class pixels, IoU, precision, recall |
| `run_config.json` | Versions, git commit, settings, split sizes, timings and the majority-class baseline of the test split |

---

## ghost predict (architecture 0.2.0)

```bash
ghost predict --model ghost_model.pt --data <cube> [--gt <labels>] [--out-dir <dir>]
```

Writes `prediction.npy` (label per pixel, 0 where the spectrum is unusable) and `prediction.png`. With `--gt`:
on the training scene only its held-out test pixels are scored, which reproduces training's test result; on
any other scene every labelled pixel is scored. Results go to `predict_results.csv` and
`predict_class_report.csv`. `--dataset indian_pines|pavia|salinas` names the classes in the legend.

---

## ghost visualize (architecture 0.2.0)

```bash
ghost visualize --model ghost_model.pt --data <cube> [--gt <labels>] [--dataset indian_pines] [--out-dir <dir>]
```

Writes `segmentation.png`: false colour | ground truth (if given) | prediction for every pixel.
`--r_band/--g_band/--b_band` and `--title` work as in v0.1.

---

## ghost train_spt (architecture 0.1.7)

The v0.1.7 training command. Builds the Spectral Partition Tree and trains per-node model ensembles.

```bash
ghost train_spt --data <path> --gt <path> [options]
```

### Required

| Flag | Description |
|------|-------------|
| `--data` | Path to hyperspectral `.mat` file. Shape: `(H, W, Bands)` |
| `--gt` | Path to ground truth `.mat` file. Shape: `(H, W)`, integer class IDs. 0 = background |

### Data

| Flag | Default | Description |
|------|---------|-------------|
| `--train_ratio` | `0.2` | Fraction of labeled pixels per class for training |
| `--val_ratio` | `0.1` | Fraction of labeled pixels per class for validation |

### Model

| Flag | Default | Description |
|------|---------|-------------|
| `--base_filters` | `32` | U-Net base filters. Channel progression: `f, 2f, 4f, 8f, 16f` |
| `--num_filters` | `8` | Spectral 3D conv filters per layer |
| `--num_blocks` | `3` | Number of 3D conv blocks |

### Training

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `400` | Base epoch budget for root node. Child nodes get scaled budgets |
| `--lr` | `1e-4` | Learning rate (AdamW) |
| `--loss` | `ce` | Loss function: `ce`, `dice`, `focal`, `squared_ce` |
| `--focal_gamma` | `2.0` | Gamma for focal loss (only used with `--loss focal`) |
| `--patience` | `50` | Early stop after N epochs without improvement |
| `--min_epochs` | `40` | Never early-stop before this epoch |
| `--warmup_epochs` | `0` | Linear LR warmup epochs |
| `--val_interval` | `20` | Validate every N epochs |
| `--seed` | `42` | Random seed |

### SPT (Spectral Partition Tree)

| Flag | Default | Description |
|------|---------|-------------|
| `--depth` | `auto` | Tree depth. `auto`: stops at depth 3 or SAM < 0.05. `full`: always recurse. Integer: fixed max depth |
| `--ensembles` | `5` | Ensemble size per internal node |
| `--leaf_ensembles` | `3` | Ensemble size per leaf node (<=2 classes) |

### Routing (Experimental)

| Flag | Default | Description |
|------|---------|-------------|
| `--routing` | `forest` | Routing mode: `forest` (recommended), `hybrid`, `soft` |
| `--d_model` | `64` | SSM fingerprint dimensionality |
| `--d_state` | `16` | SSM filters per branch |
| `--ssm_epochs` | `300` | SSM pretraining epochs. Set to `1` when using `--routing forest` |
| `--ssm_lr` | `1e-3` | SSM pretraining learning rate |
| `--ssm_save` | `ssm_pretrained.pt` | SSM weights save path (inside `--out-dir`) |
| `--ssm_load` | `None` | Load pre-existing SSM weights (skip pretraining) |

### Output

| Flag | Default | Description |
|------|---------|-------------|
| `--out-dir` | `.` | Output directory (created if needed) |
| `--save` | `spt_models.pkl` | Model bundle filename |

### Output Files

| File | Description |
|------|-------------|
| `spt_models.pkl` | Complete model bundle: tree + all ensembles + SSM state |
| `ssm_pretrained.pt` | Standalone SSM encoder weights |
| `training_history.csv` | Epoch-by-epoch metrics for all nodes |

### Examples

**Recommended (fast, good results):**

```bash
ghost train_spt \
  --data data/indian_pines/Indian_pines_corrected.mat \
  --gt   data/indian_pines/Indian_pines_gt.mat \
  --loss dice \
  --base_filters 32 --num_filters 8 \
  --ensembles 5 --leaf_ensembles 3 \
  --epochs 400 --patience 50 --min_epochs 40 \
  --val_interval 20 \
  --out-dir runs/indian_pines
```

**Low VRAM (4-6 GB):**

```bash
ghost train_spt \
  --data data.mat --gt labels.mat \
  --loss dice \
  --base_filters 16 --num_filters 4 --d_model 32 \
  --ensembles 3 --epochs 300 \
  --out-dir runs/low_vram
```

**Maximum accuracy (8+ GB, slow):**

```bash
ghost train_spt \
  --data data.mat --gt labels.mat \
  --loss dice \
  --base_filters 64 --num_filters 32 \
  --ensembles 5 --leaf_ensembles 3 \
  --epochs 400 --patience 50 \
  --out-dir runs/full_power
```

---

## ghost train --arch 0.1.7

v0.1 flat model training (no SPT). Useful as a baseline.

```bash
ghost train --arch 0.1.7 --data <path> --gt <path> [options]
```

### Flags

Same as `ghost train_spt` for: `--data`, `--gt`, `--train_ratio`, `--val_ratio`, `--base_filters`, `--num_filters`, `--num_blocks`, `--epochs`, `--lr`, `--seed`, `--out-dir`, `--save`.

Additional:

| Flag | Default | Description |
|------|---------|-------------|
| `--fp16` | `False` | Mixed precision training. Reduces VRAM ~40% |
| `--log` | `training_log.csv` | Per-epoch log filename |

### Output Files

| File | Description |
|------|-------------|
| `best_model.pth` | Best model weights (by val mIoU) |
| `training_log.csv` | `epoch, train_loss, val_loss, val_oa, val_miou, ...` |
| `test_results.csv` | Final test metrics |

---

## ghost predict (SPT checkpoints, architecture 0.1.7)

Run inference on the test split using a trained SPT model.

```bash
ghost predict --data <path> --gt <path> --model <path> [options]
```

### Required

| Flag | Description |
|------|-------------|
| `--data` | Hyperspectral data `.mat` file |
| `--gt` | Ground truth `.mat` file |
| `--model` | Path to `spt_models.pkl` from `ghost train_spt` |

### Optional

| Flag | Default | Description |
|------|---------|-------------|
| `--routing` | `all` | `forest`, `hybrid`, `soft`, or `all` (runs all three) |
| `--ssm_load` | `None` | Standalone SSM weights. Falls back to embedded state in pkl |
| `--train_ratio` | `0.2` | Must match training value |
| `--val_ratio` | `0.1` | Must match training value |
| `--seed` | `42` | Must match training value |
| `--out-dir` | `.` | Output directory |

### Output Files

| File | Description |
|------|-------------|
| `test_results_forest.csv` | OA, mIoU, Dice, Precision, Recall for ensemble routing |
| `test_results_hybrid.csv` | Same for hybrid routing |
| `test_results_soft.csv` | Same for soft routing |

### Example

```bash
ghost predict \
  --data  data/indian_pines/Indian_pines_corrected.mat \
  --gt    data/indian_pines/Indian_pines_gt.mat \
  --model runs/indian_pines/spt_models.pkl \
  --routing forest --out-dir runs/indian_pines
```

---

## ghost visualize (SPT checkpoints, architecture 0.1.7)

Generate a 3-panel PNG: false colour composite | ground truth | GHOST prediction.

```bash
ghost visualize --data <path> --gt <path> --model <path> [options]
```

### Required

| Flag | Description |
|------|-------------|
| `--data` | Hyperspectral data `.mat` file |
| `--gt` | Ground truth `.mat` file |
| `--model` | Path to `spt_models.pkl` |

### Optional

| Flag | Default | Description |
|------|---------|-------------|
| `--routing` | `forest` | Routing mode for the prediction panel |
| `--dataset` | `None` | Dataset name for class labels: `indian_pines`, `pavia`, `salinas` |
| `--r_band` | `Bands*0.75` | Band index for red channel in false colour |
| `--g_band` | `Bands*0.50` | Band index for green channel |
| `--b_band` | `Bands*0.25` | Band index for blue channel |
| `--title` | `GHOST Segmentation` | Figure title |
| `--ssm_load` | `None` | Standalone SSM weights |
| `--train_ratio` | `0.2` | Must match training value |
| `--val_ratio` | `0.1` | Must match training value |
| `--seed` | `42` | Must match training value |
| `--out-dir` | `.` | Output directory |

### Output Files

| File | Description |
|------|-------------|
| `segmentation_<routing>.png` | 3-panel figure, 180 DPI, dark background |

### Example

```bash
ghost visualize \
  --data    data/indian_pines/Indian_pines_corrected.mat \
  --gt      data/indian_pines/Indian_pines_gt.mat \
  --model   runs/indian_pines/spt_models.pkl \
  --dataset indian_pines \
  --title   "GHOST - Indian Pines" \
  --out-dir runs/indian_pines
```

---

## ghost convert_to_mat

Convert ENVI, TIFF, GeoTIFF, or HDF5 hyperspectral images to `.mat` format. Requires optional dependencies: `pip install ghost-hsi[convert]`

```bash
ghost convert_to_mat --img <path> --out-dir <path> [options]
```

### Required

| Flag | Description |
|------|-------------|
| `--img` | Path to hyperspectral image file (`.hdr`, `.tif`, `.tiff`, `.h5`, `.hdf5`, `.nc`) |
| `--out-dir` | Output directory for `.mat` and metadata files |

### Optional

| Flag | Default | Description |
|------|---------|-------------|
| `--gt` | `None` | Path to ground-truth labels (`.mat`, `.png`, `.tif`, `.hdr`) |
| `--crop` | `None` | Spatial crop: `Y X Height Width` (e.g. `--crop 448 2560 512 512`) |
| `--data-key` | `data` | Key name for image data in the output `.mat` |
| `--gt-key` | `gt` | Key name for ground truth in the output `.mat` |

### Format auto-detection

| Extension | Format | Library |
|-----------|--------|---------|
| `.hdr`, `.img` | ENVI | `spectral` |
| `.tif`, `.tiff` | TIFF / GeoTIFF | `rasterio` |
| `.h5`, `.hdf5`, `.he5`, `.hdf`, `.nc` | HDF5 / NetCDF4 | `h5py` |

For HDF5 files with multiple datasets, specify the dataset path: `file.h5:/radiance/radiance`

### Output files

| File | Description |
|------|-------------|
| `data.mat` | Hyperspectral image data |
| `gt.mat` | Ground truth labels (only if `--gt` provided) |
| `metadata.json` | All preserved metadata (CRS, wavelengths, transforms, band names, etc.) |

### Examples

```bash
# ENVI to .mat
ghost convert_to_mat \
  --img scene.hdr \
  --gt  labels.tif \
  --out-dir converted/

# GeoTIFF with spatial crop
ghost convert_to_mat \
  --img satellite.tif \
  --out-dir converted/ \
  --crop 100 200 512 512

# HDF5 with specific dataset
ghost convert_to_mat \
  --img aviris.nc:/radiance/radiance \
  --out-dir converted/
```

---

## Configuration Presets

| Scenario | Key Flags |
|----------|-----------|
| **First run / demo** | `--loss dice --epochs 400 --patience 50` |
| **Low VRAM (4 GB)** | `--base_filters 16 --num_filters 4 --d_model 32` |
| **Max accuracy** | `--base_filters 64 --num_filters 32 --ensembles 5` |
| **Fast iteration** | `--base_filters 32 --num_filters 8 --epochs 200 --ensembles 3` |
