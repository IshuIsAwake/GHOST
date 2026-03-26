# GHOST API Reference

All commands via the `ghost` CLI. Install: `pip install ghost-hsi`

```bash
ghost <command> [arguments]
```

| Command | Description |
|---------|-------------|
| `ghost train` | Train a single flat model (no SPT) |
| `ghost train_spt` | **Full GHOST pipeline** — Spectral Partition Tree + ensembles |
| `ghost predict` | Run inference on test split, compute metrics |
| `ghost visualize` | Generate 3-panel segmentation figure |
| `ghost demo` | Show bundled dataset paths and example command |
| `ghost version` | Print version |
| `ghost flower` | Easter egg |

---

## ghost train_spt

The primary training command. Builds the Spectral Partition Tree and trains per-node model ensembles.

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

## ghost train

Flat model training (no SPT). Useful as a baseline.

```bash
ghost train --data <path> --gt <path> [options]
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

## ghost predict

Run inference on the test split using a trained model.

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

## ghost visualize

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

## Configuration Presets

| Scenario | Key Flags |
|----------|-----------|
| **First run / demo** | `--loss dice --epochs 400 --patience 50` |
| **Low VRAM (4 GB)** | `--base_filters 16 --num_filters 4 --d_model 32` |
| **Max accuracy** | `--base_filters 64 --num_filters 32 --ensembles 5` |
| **Fast iteration** | `--base_filters 32 --num_filters 8 --epochs 200 --ensembles 3` |
