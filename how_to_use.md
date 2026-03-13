# How to Use GHOST

Step-by-step guide from raw data to segmentation results.

---

## Requirements

```bash
pip install torch torchvision scipy numpy
```

GPU with CUDA is strongly recommended. CPU-only runs are possible but very slow.

Minimum VRAM: 4GB (with reduced filter settings). Recommended: 8GB+.

---

## Step 1 — Prepare Your Data

GHOST expects `.mat` files with the following structure:

- **Data file**: a 3D array of shape `(H, W, C)` — height × width × spectral bands
- **Labels file**: a 2D array of shape `(H, W)` — integer class IDs, 0 = background/unlabeled

GHOST auto-detects the correct keys inside the `.mat` file. If auto-detection fails (ambiguous keys), pass them manually:

```python
from datasets.hyperspectral_dataset import HyperspectralDataset

ds = HyperspectralDataset(
    data_path  = 'data.mat',
    gt_path    = 'labels.mat',
    data_key   = 'my_data_key',    # optional override
    labels_key = 'my_labels_key',  # optional override
    split      = 'train'
)
```

---

## Step 2 — Choose Your Training Mode

### Option A: Flat model (quick baseline)

No RSSP tree, no SSSR routing. Single HyperspectralNet trained end-to-end.

```bash
python train.py \
    --data data/indian_pines/Indian_pines_corrected.mat \
    --gt   data/indian_pines/Indian_pines_gt.mat
```

Use this to establish a baseline OA and mIoU before enabling RSSP.

### Option B: Full RSSP + SSSR (recommended)

```bash
python train_rssp.py \
    --data data/indian_pines/Indian_pines_corrected.mat \
    --gt   data/indian_pines/Indian_pines_gt.mat \
    --ssm_save ssm_indian_pines.pt \
    --ssm_epochs 300 \
    --routing hybrid
```

This runs three stages automatically:
1. SSM encoder pretraining (fast, ~100 epochs on labeled pixels)
2. RSSP tree construction (instant, SAM-based clustering)
3. Node-by-node forest training + router training (the long part)

---

## Step 3 — Repeated Runs on the Same Dataset

The SSM pretraining step only needs to run once per dataset. On subsequent runs, load the saved weights:

```bash
python train_rssp.py \
    --data     data/indian_pines/Indian_pines_corrected.mat \
    --gt       data/indian_pines/Indian_pines_gt.mat \
    --ssm_load ssm_indian_pines.pt \
    --routing hybrid
```

To run inference or test alternate routing modes without retraining:

```bash
# Compare all routing modes
for mode in hybrid forest soft; do
    python predict.py \
        --data data/indian_pines/Indian_pines_corrected.mat \
        --gt data/indian_pines/Indian_pines_gt.mat \
        --model rssp_models.pkl \
        --ssm_load ssm_indian_pines.pt \
        --routing $mode
done
```

---

## Step 4 — Monitor Training

Progress is printed live during training. For each node, every 20 epochs:

```
  Epoch  20 | Loss: 0.4231 | Val Loss: 0.3912 | OA: 0.8123 | mIoU: 0.5341 | Dice: 0.6102 | Prec: 0.6421 | Rec: 0.5893
```

At the end of training:

```
========================================
Test OA:        0.9072
Test mIoU:      0.5900
Test Dice:      0.6841
Test Precision: 0.7102
Test Recall:    0.6523
========================================
```

**What to watch:**
- Val mIoU is the primary quality signal. If it's still climbing at the end of training, increase `--epochs`
- If val mIoU oscillates wildly, the learning rate may be too high. Try `--lr 5e-5`
- Router BCE loss during SSSR training should converge below 0.5. If it stays near 0.693, the SSM fingerprints are not discriminative enough — increase `--ssm_epochs`

---

## Step 5 — Hardware-Limited Setups

If you have limited VRAM (6GB or less), use reduced filter settings:

```bash
python train_rssp.py \
    --data        your_data.mat \
    --gt          your_labels.mat \
    --base_filters 16 \
    --num_filters  4 \
    --d_model      32
```

For fp16 mixed precision (saves ~40% VRAM):

```bash
python train.py --data ... --gt ... --fp16
```

Note: `--fp16` is only available in `train.py` (flat model). RSSP training enforces fp32 at the continuum removal step regardless.

---

## Current Constraints

| Constraint | Detail |
|---|---|
| Input format | `.mat` files only. ENVI `.hdr`/`.raw`, GeoTIFF support planned |
| Spatial size | No hard limit, but very large scenes (>2000×2000) may exceed RAM during fingerprint precomputation |
| Number of classes | Works well for 5–30 classes. Below 5, RSSP tree has little benefit. Above 30, minority classes may have too few pixels for stable node training |
| Minimum labeled pixels per class | At least 10 pixels per class required for tree splitting. Classes with fewer pixels are kept in their parent node |
| VRAM | 6GB minimum at reduced settings. 8GB for default settings |

---

## Datasets Tested

| Dataset | Bands | Classes | Spatial | Notes |
|---|---|---|---|---|
| Indian Pines | 200 | 16 | 145×145 | Standard benchmark |
| Pavia University | 103 | 9 | 610×340 | Requires reduced filters at 6GB VRAM |
| HUU Vaccine | TBD | TBD | TBD | In progress |
| CRISM (Mars) | TBD | TBD | TBD | Planetary, very different spectral range |

---

## Targets to Aim For

These are empirical observations across tested datasets. Your results will vary with dataset difficulty and hardware.

| Metric | Acceptable | Good | Excellent |
|---|---|---|---|
| OA | > 0.80 | > 0.88 | > 0.93 |
| mIoU | > 0.40 | > 0.55 | > 0.70 |
| Dice | > 0.50 | > 0.65 | > 0.75 |

If results are well below "acceptable" after 300 epochs:
1. Check that your `.mat` keys are loaded correctly (print `data.shape` and `labels.max()`)
2. Verify class distribution — if one class has 5 pixels total, mIoU will be poor
3. Try increasing `--train_ratio` to `0.3`
4. Try `--depth full` to see if a deeper tree helps

---

## File Outputs

| File | Contents |
|---|---|
| `best_model.pth` | Best flat model weights (train.py only) |
| `training_log.csv` | Per-epoch metrics during flat training |
| `test_results.csv` | Final test metrics (flat training) |
| `rssp_models.pkl` | All RSSP node models, routers, tree structure, SSM state |
| `ssm_pretrained.pt` | Standalone SSM encoder weights (reusable across runs) |