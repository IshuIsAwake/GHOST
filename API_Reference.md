# GHOST API Reference

Complete reference for all CLI commands, Python API, and configuration parameters.

---

## Table of Contents

1. [CLI Commands](#cli-commands)
   - [ghost train](#ghost-train)
   - [ghost train_rssp](#ghost-train_rssp)
   - [ghost predict](#ghost-predict)
   - [ghost visualize](#ghost-visualize)
2. [Python API](#python-api)
   - [HyperspectralDataset](#hyperspectral-dataset)
   - [HyperspectralNet](#hyperspectralnet)
   - [SpectralSSMEncoder](#spectralssme-ncoder)
   - [build_rssp_tree](#build_rssp_tree)
   - [train_tree](#train_tree)
   - [run_inference](#run_inference)
   - [compute_rssp_metrics](#compute_rssp_metrics)
   - [pretrain_ssm](#pretrain_ssm)
   - [SSSRRouter](#sssrrouter)
3. [Results Reference](#results-reference)

---

## CLI Commands

All commands are accessed via the `ghost` entry point installed by `pip install -e .`

```bash
ghost <command> [arguments]
```

---

### ghost train

Trains a single flat `HyperspectralNet` end-to-end. No RSSP tree, no SSSR routing. Use this to establish a baseline before enabling RSSP.

```bash
ghost train --data <path> --gt <path> [options]
```

#### Required Arguments

| Argument | Type | Description |
|---|---|---|
| `--data` | `str` | Path to hyperspectral data `.mat` file. Array must be shape `(H, W, C)`. |
| `--gt` | `str` | Path to ground truth labels `.mat` file. Array must be shape `(H, W)` with integer class IDs. Class 0 is treated as background and ignored. |

#### Data Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--train_ratio` | `float` | `0.2` | Fraction of labeled pixels per class used for training. Applied per-class (stratified). |
| `--val_ratio` | `float` | `0.1` | Fraction of labeled pixels per class used for validation. Applied per-class (stratified). |

#### Model Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--base_filters` | `int` | `32` | Base filter count for the 2D U-Net encoder and decoder. Channel progression is `f → f×2 → f×4 → f×8 → f×16`. Reduce to `16` for VRAM-constrained setups. |
| `--num_filters` | `int` | `8` | Number of output filters per layer in the Spectral3DStack. Output channels = `num_filters × C`. Reduce to `4` for VRAM-constrained setups. |
| `--num_blocks` | `int` | `3` | Number of 3D convolution blocks in the Spectral3DStack. |

#### Training Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--epochs` | `int` | `300` | Total number of training epochs. |
| `--lr` | `float` | `1e-4` | Initial learning rate for AdamW. Scheduler halves this when val loss plateaus. |
| `--seed` | `int` | `42` | Random seed for reproducibility. Controls weight initialisation, data shuffling, and split generation. |
| `--fp16` | `flag` | `False` | Enable mixed precision training. Reduces VRAM by ~40% with minor performance tradeoff. Not available in `train_rssp`. |

#### Output Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--out-dir` | `str` | `.` | Directory to save all output files. Created if it does not exist. |
| `--save` | `str` | `best_model.pth` | Filename for the best model checkpoint, saved inside `--out-dir`. |
| `--log` | `str` | `training_log.csv` | Filename for the per-epoch training log CSV, saved inside `--out-dir`. |

#### Output Files

| File | Description |
|---|---|
| `best_model.pth` | Model weights at the epoch with best validation mIoU. |
| `training_log.csv` | Per-epoch: `epoch, train_loss, val_loss, val_oa, val_miou, val_dice, val_precision, val_recall` |
| `test_results.csv` | Final test metrics: `best_epoch, test_oa, test_miou, test_dice, test_precision, test_recall` |

#### Example

```bash
ghost train \
    --data data/indian_pines/Indian_pines_corrected.mat \
    --gt   data/indian_pines/Indian_pines_gt.mat \
    --base_filters 32 \
    --num_filters 8 \
    --epochs 300 \
    --lr 1e-4 \
    --out-dir runs/indian_pines_flat \
    --save best_model.pth
```

---

### ghost train_rssp

Full GHOST pipeline: SSM encoder pretraining → RSSP tree construction → per-node forest training → SSSR router training. This is the primary training command.

```bash
ghost train_rssp --data <path> --gt <path> [options]
```

#### Required Arguments

| Argument | Type | Description |
|---|---|---|
| `--data` | `str` | Path to hyperspectral data `.mat` file. Array must be shape `(H, W, C)`. |
| `--gt` | `str` | Path to ground truth labels `.mat` file. Array must be shape `(H, W)` with integer class IDs. |

#### Data Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--train_ratio` | `float` | `0.2` | Fraction of labeled pixels per class for training. |
| `--val_ratio` | `float` | `0.1` | Fraction of labeled pixels per class for validation. |

#### RSSP Tree Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--depth` | `str` | `auto` | Tree depth control. `auto`: stops at depth 3 or when intra-node SAM < 0.05. `full`: always recurse until ≤ 2 classes remain. Integer (e.g. `2`): fixed maximum depth. |

#### Forest Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--forests` | `int` | `5` | Number of independent forest members per node. Higher values reduce variance but increase training time linearly. |
| `--base_filters` | `int` | `32` | Base filter count for each node's HyperspectralNet. Use `16` for VRAM ≤ 6GB. |
| `--num_filters` | `int` | `8` | Spectral3DStack filters per node. Use `4` for VRAM ≤ 6GB. |
| `--num_blocks` | `int` | `3` | 3D convolution blocks per node. |
| `--epochs` | `int` | `300` | Base epoch budget for the root node. Child nodes receive a proportionally reduced budget: `max(epochs//2, epochs × node_classes/total_classes)`. |
| `--lr` | `float` | `1e-4` | Learning rate for node forest training. |

#### SSM / SSSR Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--d_model` | `int` | `64` | Dimensionality of the SSM fingerprint vector. Higher values produce richer representations but increase router parameter count. Use `32` for VRAM ≤ 6GB. |
| `--d_state` | `int` | `16` | Number of filters per branch in the SpectralSSMEncoder. Controls the capacity of each scale branch (narrow/mid/wide). |
| `--ssm_epochs` | `int` | `300` | Epochs for SSM encoder pretraining. Set to `1` to skip pretraining when using `--routing forest`. |
| `--ssm_lr` | `float` | `1e-3` | Learning rate for SSM pretraining. Higher than forest LR because pretraining uses mini-batches. |
| `--ssm_save` | `str` | `ssm_pretrained.pt` | Filename to save the pretrained SSM encoder weights, inside `--out-dir`. |
| `--ssm_load` | `str` | `None` | Path to pre-existing SSM weights. If provided, SSM pretraining is skipped entirely. Use this for repeated runs on the same dataset. |
| `--routing` | `str` | `hybrid` | Routing mode for cascade inference at evaluation time. `hybrid`: forest base + SSM correction (recommended). `forest`: forest-only soft routing, no SSM. `soft`: SSM-only routing. |

#### General Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--seed` | `int` | `42` | Random seed for all stochastic operations. |
| `--save` | `str` | `rssp_models.pkl` | Filename for the trained model bundle (all node forests + routers + tree + SSM state), saved inside `--out-dir`. |
| `--out-dir` | `str` | `.` | Output directory. Created if it does not exist. |

#### Output Files

| File | Description |
|---|---|
| `rssp_models.pkl` | Complete model bundle: tree structure, all node forests, all router states, embedded SSM state dict. |
| `ssm_pretrained.pt` | Standalone SSM encoder weights. Reusable across runs via `--ssm_load`. |

#### Example — Full run

```bash
ghost train_rssp \
    --data data/indian_pines/Indian_pines_corrected.mat \
    --gt   data/indian_pines/Indian_pines_gt.mat \
    --ssm_epochs 300 \
    --epochs 300 \
    --forests 5 \
    --routing hybrid \
    --out-dir runs/indian_pines \
    --save rssp_models.pkl \
    --ssm_save ssm_indian_pines.pt
```

#### Example — Reusing pretrained SSM

```bash
ghost train_rssp \
    --data data/indian_pines/Indian_pines_corrected.mat \
    --gt   data/indian_pines/Indian_pines_gt.mat \
    --ssm_load runs/indian_pines/ssm_indian_pines.pt \
    --epochs 300 \
    --out-dir runs/indian_pines_v2
```

#### Example — VRAM-constrained setup (≤ 6GB)

```bash
ghost train_rssp \
    --data data/pavia/PaviaU.mat \
    --gt   data/pavia/PaviaU_gt.mat \
    --base_filters 16 \
    --num_filters 4 \
    --d_model 32 \
    --ssm_epochs 200 \
    --epochs 300 \
    --routing forest \
    --out-dir runs/pavia
```

---

### ghost predict

Loads a trained `rssp_models.pkl` and runs cascade inference on the test split. Supports all three routing modes and can run all three in a single call with `--routing all`.

```bash
ghost predict --data <path> --gt <path> --model <path> [options]
```

#### Required Arguments

| Argument | Type | Description |
|---|---|---|
| `--data` | `str` | Path to hyperspectral data `.mat` file. |
| `--gt` | `str` | Path to ground truth labels `.mat` file. |
| `--model` | `str` | Path to `rssp_models.pkl` produced by `ghost train_rssp`. |

#### Optional Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--ssm_load` | `str` | `None` | Path to standalone SSM weights. If not provided, SSM state is loaded from the embedded state inside `rssp_models.pkl`. |
| `--train_ratio` | `float` | `0.2` | Must match the value used during training to reproduce the same test split. |
| `--val_ratio` | `float` | `0.1` | Must match the value used during training. |
| `--seed` | `int` | `42` | Must match the value used during training. |
| `--routing` | `str` | `all` | Routing mode. `all`: runs forest, hybrid, and soft sequentially and saves separate CSVs for each. `hybrid`, `forest`, or `soft`: single mode. |
| `--out-dir` | `str` | `.` | Output directory for result CSVs. |

#### Output Files

| File | Description |
|---|---|
| `test_results_forest.csv` | OA, mIoU, Dice, Precision, Recall for forest routing. |
| `test_results_hybrid.csv` | Same for hybrid routing. |
| `test_results_soft.csv` | Same for soft routing. |

#### Example

```bash
ghost predict \
    --data  data/indian_pines/Indian_pines_corrected.mat \
    --gt    data/indian_pines/Indian_pines_gt.mat \
    --model runs/indian_pines/rssp_models.pkl \
    --routing all \
    --out-dir runs/indian_pines
```

---

### ghost visualize

Runs inference and produces a three-panel PNG: false colour composite, ground truth labels, and GHOST prediction side by side.

```bash
ghost visualize --data <path> --gt <path> --model <path> [options]
```

#### Required Arguments

| Argument | Type | Description |
|---|---|---|
| `--data` | `str` | Path to hyperspectral data `.mat` file. |
| `--gt` | `str` | Path to ground truth labels `.mat` file. |
| `--model` | `str` | Path to `rssp_models.pkl`. |

#### Optional Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--ssm_load` | `str` | `None` | Path to standalone SSM weights. Falls back to embedded SSM state in pkl if not provided. |
| `--train_ratio` | `float` | `0.2` | Must match training value. |
| `--val_ratio` | `float` | `0.1` | Must match training value. |
| `--seed` | `int` | `42` | Must match training value. |
| `--routing` | `str` | `forest` | Routing mode for the visualized prediction. `forest` recommended — it is the strongest routing mode in most configurations. |
| `--dataset` | `str` | `None` | Dataset name for human-readable class labels in the legend. Supported values: `indian_pines`, `pavia`, `salinas`. If not provided or unrecognised, falls back to `Class 1, Class 2, ...` |
| `--r_band` | `int` | `C × 0.75` | Band index to map to the red channel in the false colour composite. |
| `--g_band` | `int` | `C × 0.50` | Band index to map to the green channel. |
| `--b_band` | `int` | `C × 0.25` | Band index to map to the blue channel. |
| `--title` | `str` | `GHOST Segmentation` | Title displayed above the three-panel figure. |
| `--out-dir` | `str` | `.` | Output directory for the saved PNG. |

#### Output Files

| File | Description |
|---|---|
| `segmentation_<routing>.png` | Three-panel figure at 180 DPI on a dark background. |

#### Example — Indian Pines

```bash
ghost visualize \
    --data    data/indian_pines/Indian_pines_corrected.mat \
    --gt      data/indian_pines/Indian_pines_gt.mat \
    --model   runs/indian_pines/rssp_models.pkl \
    --dataset indian_pines \
    --routing forest \
    --out-dir runs/indian_pines \
    --title   "GHOST — Indian Pines"
```

#### Example — Custom band selection for false colour

```bash
ghost visualize \
    --data   data/pavia/PaviaU.mat \
    --gt     data/pavia/PaviaU_gt.mat \
    --model  runs/pavia/rssp_models.pkl \
    --dataset pavia \
    --r_band 60 \
    --g_band 30 \
    --b_band 10 \
    --title  "GHOST — Pavia University" \
    --out-dir runs/pavia
```

---

## Python API

All modules are importable from the `ghost` package after `pip install -e .`

---

### HyperspectralDataset

```python
from ghost.datasets.hyperspectral_dataset import HyperspectralDataset
```

Universal `.mat` loader. Auto-detects data and labels keys by array dimensionality. Performs stratified splits per class.

```python
HyperspectralDataset(
    data_path,
    gt_path,
    split='train',
    train_ratio=0.2,
    val_ratio=0.1,
    data_key=None,
    labels_key=None,
    seed=42,
    use_fp16=False
)
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data_path` | `str` | required | Path to `.mat` file containing the hyperspectral data cube. |
| `gt_path` | `str` | required | Path to `.mat` file containing ground truth labels. |
| `split` | `str` | `'train'` | Which split to return. One of `'train'`, `'val'`, `'test'`. |
| `train_ratio` | `float` | `0.2` | Fraction of each class's pixels for training. |
| `val_ratio` | `float` | `0.1` | Fraction of each class's pixels for validation. |
| `data_key` | `str` | `None` | Key for the data array inside the `.mat` file. If `None`, auto-detected as the 3D array. |
| `labels_key` | `str` | `None` | Key for the labels array. If `None`, auto-detected as the 2D array. |
| `seed` | `int` | `42` | Random seed for stratified split generation. |
| `use_fp16` | `bool` | `False` | If `True`, applies min-max normalisation instead of Z-score to prevent fp16 overflow in continuum removal. |

#### Attributes

| Attribute | Type | Description |
|---|---|---|
| `data` | `torch.Tensor (C, H, W)` | Full normalised hyperspectral cube. |
| `labels` | `torch.Tensor (H, W)` | Full integer label map. |
| `num_bands` | `int` | Number of spectral bands `C`. |
| `num_classes` | `int` | Number of classes including background (= `labels.max() + 1`). |
| `train_coords` | `np.ndarray (N, 2)` | Row/column coordinates of training pixels. |
| `val_coords` | `np.ndarray (M, 2)` | Row/column coordinates of validation pixels. |
| `test_coords` | `np.ndarray (K, 2)` | Row/column coordinates of test pixels. |
| `split_mask` | `torch.Tensor (H, W)` | Label map with only the current split's pixels non-zero. |

#### Example

```python
from ghost.datasets.hyperspectral_dataset import HyperspectralDataset

train_ds = HyperspectralDataset(
    'data/indian_pines/Indian_pines_corrected.mat',
    'data/indian_pines/Indian_pines_gt.mat',
    split='train'
)

print(train_ds.num_bands)    # 200
print(train_ds.num_classes)  # 17
print(train_ds.data.shape)   # torch.Size([200, 145, 145])
```

---

### HyperspectralNet

```python
from ghost.models.hyperspectral_net import HyperspectralNet
```

Full segmentation pipeline: ContinuumRemoval → Spectral3DStack → SEBlock → Encoder2D → Decoder2D.

```python
HyperspectralNet(
    num_bands,
    num_classes,
    num_filters=8,
    num_blocks=3,
    base_filters=32,
    use_fp16=False
)
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_bands` | `int` | required | Number of input spectral bands `C`. Use `dataset.num_bands`. |
| `num_classes` | `int` | required | Number of output classes including background. Use `dataset.num_classes`. |
| `num_filters` | `int` | `8` | Filters per layer in Spectral3DStack. |
| `num_blocks` | `int` | `3` | Number of 3D conv blocks. |
| `base_filters` | `int` | `32` | Base filter count for 2D U-Net. |
| `use_fp16` | `bool` | `False` | Passed to ContinuumRemoval for numerical stability control. |

#### Forward

```python
logits = model(x)  # x: (B, C, H, W) → logits: (B, num_classes, H, W)
```

#### Example

```python
from ghost.models.hyperspectral_net import HyperspectralNet

model = HyperspectralNet(
    num_bands=200,
    num_classes=17,
    base_filters=32,
    num_filters=8
)
print(sum(p.numel() for p in model.parameters()))  # ~12M parameters
```

---

### SpectralSSMEncoder

```python
from ghost.models.spectral_ssm import SpectralSSMEncoder
```

Physics-informed spectral fingerprint encoder. ContinuumRemoval → parallel multi-scale 1D CNN → channel attention → d_model fingerprint.

```python
SpectralSSMEncoder(
    d_model=64,
    d_state=16,
    use_fp16=False
)
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `d_model` | `int` | `64` | Output fingerprint dimensionality. |
| `d_state` | `int` | `16` | Filters per branch (narrow/mid/wide). Total concatenated filters = `3 × d_state`. |
| `use_fp16` | `bool` | `False` | Passed to ContinuumRemoval. |

#### Forward

```python
fingerprints = encoder(x)  # x: (B, C, H, W) → fingerprints: (B, d_model, H, W)
```

#### Notes

- Assumes `C ≥ 50` for the wide branch kernel (size 31). For fewer bands use `--routing forest` and set `--ssm_epochs 1`.
- After pretraining, freeze all parameters: `for p in encoder.parameters(): p.requires_grad_(False)`

---

### build_rssp_tree

```python
from ghost.rssp.sam_clustering import build_rssp_tree
```

Constructs the RSSP binary tree from data statistics. Computes class mean spectra, applies continuum removal, builds SAM distance matrix, and recursively splits.

```python
tree, sam_matrix, means = build_rssp_tree(
    data,
    labels,
    num_classes,
    depth_mode='auto',
    max_depth=None,
    min_pixels=10,
    sam_threshold=0.05
)
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `np.ndarray (C, H, W)` | required | Hyperspectral data cube. |
| `labels` | `np.ndarray (H, W)` | required | Integer label map. |
| `num_classes` | `int` | required | Total number of classes including background. |
| `depth_mode` | `str or int` | `'auto'` | `'auto'`: stop at depth 3 or SAM < threshold. `'full'`: always recurse. Integer: fixed max depth. |
| `max_depth` | `int` | `None` | Used internally when `depth_mode` is integer. |
| `min_pixels` | `int` | `10` | Minimum pixels per class required to allow splitting. |
| `sam_threshold` | `float` | `0.05` | Mean intra-node SAM below which splitting is skipped (classes too similar). |

#### Returns

| Return | Type | Description |
|---|---|---|
| `tree` | `dict` | Nested dict with keys `classes`, `depth`, `left`, `right`. |
| `sam_matrix` | `np.ndarray (K, K)` | Full pairwise SAM distance matrix. |
| `means` | `np.ndarray (K, C)` | Per-class mean spectra (raw, before continuum removal). |

---

### train_tree

```python
from ghost.rssp.rssp_trainer import train_tree
```

Recursively trains all RSSP tree nodes. Returns a dict mapping node IDs to their trained model bundles.

```python
trained_models = train_tree(
    tree,
    data,
    labels,
    total_classes,
    train_coords,
    val_coords,
    fp_map,
    ssm_d_model=64,
    base_epochs=300,
    num_forests=5,
    base_filters=32,
    num_filters=8,
    num_blocks=3,
    lr=1e-4,
    device='cuda',
    node_id='root'
)
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tree` | `dict` | required | Tree dict from `build_rssp_tree`. |
| `data` | `torch.Tensor (C, H, W)` | required | Full hyperspectral data. |
| `labels` | `torch.Tensor (H, W)` | required | Full label map. |
| `total_classes` | `int` | required | Total number of non-background classes. Used for epoch budget scaling. |
| `train_coords` | `np.ndarray (N, 2)` | required | Training pixel coordinates. |
| `val_coords` | `np.ndarray (M, 2)` | required | Validation pixel coordinates. |
| `fp_map` | `torch.Tensor (H, W, d_model)` | required | Pre-computed fingerprint map from `SpectralSSMEncoder`. |
| `ssm_d_model` | `int` | `64` | Must match the `d_model` used to produce `fp_map`. |
| `base_epochs` | `int` | `300` | Epoch budget for root node. |
| `num_forests` | `int` | `5` | Forest ensemble size per node. |
| `base_filters` | `int` | `32` | U-Net base filters per node model. |
| `num_filters` | `int` | `8` | 3D stack filters per node model. |
| `num_blocks` | `int` | `3` | 3D conv blocks per node model. |
| `lr` | `float` | `1e-4` | Learning rate. |
| `device` | `str` | `'cuda'` | PyTorch device string. |
| `node_id` | `str` | `'root'` | Starting node ID. Do not change for normal usage. |

#### Returns

`dict` mapping `node_id → node_info_dict`. Node IDs follow the pattern: `'root'`, `'root_L'`, `'root_R'`, `'root_L_L'`, etc.

---

### run_inference

```python
from ghost.rssp.rssp_inference import run_inference
```

Runs soft cascade inference and returns the argmax prediction map.

```python
pred = run_inference(
    tree,
    trained_models,
    data,
    ssm_encoder,
    device,
    num_global_classes,
    routing='hybrid'
)
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tree` | `dict` | required | Tree dict from `build_rssp_tree`. |
| `trained_models` | `dict` | required | Node model dict from `train_tree`. |
| `data` | `torch.Tensor (C, H, W)` | required | Full hyperspectral data. |
| `ssm_encoder` | `SpectralSSMEncoder or None` | required | Frozen encoder. Pass `None` when `routing='forest'`. |
| `device` | `torch.device` | required | Inference device. |
| `num_global_classes` | `int` | required | Total number of classes including background. |
| `routing` | `str` | `'hybrid'` | `'hybrid'`, `'forest'`, or `'soft'`. |

#### Returns

`np.ndarray (H, W)` — predicted global class IDs.

---

### compute_rssp_metrics

```python
from ghost.rssp.rssp_inference import compute_rssp_metrics
```

Computes OA, mIoU, Dice, Precision, and Recall over labeled pixels only.

```python
oa, miou, dice, precision, recall = compute_rssp_metrics(
    pred,
    labels_np,
    num_classes
)
```

#### Parameters

| Parameter | Type | Description |
|---|---|---|
| `pred` | `np.ndarray (H, W)` | Predicted class map from `run_inference`. |
| `labels_np` | `np.ndarray (H, W)` | Ground truth label map. Pixels with value 0 are excluded. |
| `num_classes` | `int` | Total classes including background. |

#### Returns

`(float, float, float, float, float)` — `(OA, mIoU, Dice, Precision, Recall)`

---

### pretrain_ssm

```python
from ghost.rssp.ssm_pretrain import pretrain_ssm
```

Pretrains the `SpectralSSMEncoder` as a pixel-level classifier. Returns the encoder with best validation accuracy weights loaded.

```python
encoder = pretrain_ssm(
    data,
    labels,
    train_coords,
    val_coords,
    d_model=64,
    d_state=16,
    num_classes=None,
    epochs=300,
    lr=1e-3,
    batch_size=512,
    device='cuda',
    save_path='ssm_pretrained.pt'
)
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `torch.Tensor (C, H, W)` | required | Full hyperspectral data. |
| `labels` | `torch.Tensor (H, W)` | required | Full label map. |
| `train_coords` | `np.ndarray (N, 2)` | required | Training pixel coordinates. |
| `val_coords` | `np.ndarray (M, 2)` | required | Validation pixel coordinates. |
| `d_model` | `int` | `64` | Fingerprint dimensionality. |
| `d_state` | `int` | `16` | Filters per branch. |
| `num_classes` | `int` | `None` | If `None`, inferred from `labels.max() + 1`. |
| `epochs` | `int` | `300` | Pretraining epochs. |
| `lr` | `float` | `1e-3` | Learning rate (mini-batch, so higher than forest LR). |
| `batch_size` | `int` | `512` | Mini-batch size for pixel-level training. |
| `device` | `str` | `'cuda'` | Training device. |
| `save_path` | `str` | `'ssm_pretrained.pt'` | Path to save best encoder weights. |

#### Returns

`SpectralSSMEncoder` with best validation accuracy weights loaded.

---

### SSSRRouter

```python
from ghost.rssp.sssr_router import SSSRRouter
```

Per-node routing head. Maps SSM fingerprints to left-child routing probability.

```python
SSSRRouter(d_model=64)
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `d_model` | `int` | `64` | Must match the `d_model` of the `SpectralSSMEncoder` that produced the fingerprints. |

#### Forward

```python
# Spatial input
p_left = router(fingerprints)
# fingerprints: (B, d_model, H, W) → p_left: (B, H, W)

# Pixel batch input
p_left = router(fingerprints)
# fingerprints: (N, d_model) → p_left: (N,)
```

Output is always in `(0, 1)` via Sigmoid.

---

## Results Reference

All results use `--train_ratio 0.2 --val_ratio 0.1 --seed 42` stratified splits. Forest routing (`--routing forest`) unless noted.

| Dataset | Bands | Classes | Spatial | OA | mIoU | Dice | Precision | Recall | Filters |
|---|---|---|---|---|---|---|---|---|---|
| Indian Pines | 200 | 16 | 145×145 | **94.49%** | 0.7156 | 0.7399 | 0.7649 | 0.7308 | Full |
| Pavia University | 103 | 9 | 610×340 | **85.85%** | 0.6032 | 0.6685 | 0.8549 | 0.6797 | Half* |
| Salinas Valley | 204 | 16 | 512×217 | **92.10%** | 0.7387 | 0.7840 | 0.9287 | 0.7843 | Half* |
| Mars CRISM | 107 | TBD | 195×640 | — | — | — | — | — | In progress |
| Asteroid Ryugu | 7 | 4† | 1024×1024 | 54.65% | 0.3371 | 0.3520 | 0.6765 | 0.6513 | Half* |

\* Half filters: `--base_filters 16 --num_filters 4 --d_model 32`. Required for scenes exceeding ~300×300 on 6GB VRAM.

† Pseudo-labels generated by KMeans clustering. Not supervised ground truth.

### Hardware

All results produced on NVIDIA RTX 3050 (6GB VRAM). Indian Pines training time: ~60-70 minutes.