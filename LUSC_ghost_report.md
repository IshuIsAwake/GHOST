# GHOST on LUSC — Evaluation Report

## Dataset: HMI-LUSC (Hyperspectral Microscopic Imaging — Lung Squamous Cell Carcinoma)

**Source:** [Intelligent-Imaging-Center/HMILungDataset](https://github.com/Intelligent-Imaging-Center/HMILungDataset)

**Papers:**
- Data descriptor: "HMI-LUSC: A Histological Hyperspectral Imaging Dataset for Lung Squamous Cell Carcinoma" (Nature Scientific Data, 2026)
- Methods: "Improving lung cancer pathological hyperspectral diagnosis through cell-level annotation refinement" (Nature Scientific Reports, 2025)

### What the data is

Microscopic histological images of H&E-stained lung tissue sections from LUSC patients, captured with a custom hyperspectral microscope. These show **cellular-level detail** (individual cells, stroma, extracellular matrix) — not whole lung or radiology scans.

- **Full dataset:** 62 images from 10 patients, 3088x2064 pixels, 61 bands (450–750 nm)
- **Format:** ENVI (Raw + Raw.hdr) with PNG labels

### Classes

| Class | Label Color | Biological Meaning |
|-------|------------|---------------------|
| 0 | Black | Background (no tissue) |
| 1 | Red | Non-tumor cells (normal epithelium, lymphocytes) |
| 2 | Green | Tumor cells (malignant squamous cells) |
| 3 | Blue | Non-cell tissue (stroma, connective matrix) |

---

## What GHOST did

### Preprocessing (not part of GHOST)

1. Converted ENVI + PNG labels → .mat using `spectral` python library
2. Mapped RGB label colors to integer class IDs
3. Cropped a 512×512 patch (center at y=448, x=2560) containing all 3 classes
4. Saved as `lusc_512_data.mat` and `lusc_512_gt.mat`

### Training

- **Zero code changes** to GHOST
- Same CLI, same architecture, same hyperparameters as remote sensing datasets
- Command: `ghost train_rssp --loss dice --routing forest --forests 5 --leaf_forests 3 --epochs 400`
- Training time: ~46 minutes on a 6GB GPU

### Pixel distribution in 512×512 crop

| Class | Pixels | % of total |
|-------|--------|-----------|
| 0 (background) | 168,343 | 64.2% |
| 1 (non-tumor cells) | 953 | 0.4% |
| 2 (tumor cells) | 59,938 | 22.9% |
| 3 (non-cell tissue) | 32,910 | 12.6% |

---

## Results

| Metric | Value |
|--------|-------|
| OA | 99.43% |
| mIoU | 88.96% |
| Dice | 93.49% |
| Kappa | 0.9878 |
| AA | 92.68% |

### Per-class IoU

| Class | IoU | Notes |
|-------|-----|-------|
| 1 (non-tumor cells) | 68.36% | Only 953 pixels — scattered small dots |
| 2 (tumor cells) | 99.14% | Large contiguous mass — spatially easy |
| 3 (non-cell tissue) | 99.37% | Large contiguous region |

---

## Published State-of-the-Art (for comparison)

| Model | OA | Tumor Sensitivity | Notes |
|-------|-----|-------------------|-------|
| HybridSN-Att (their best) | 92.52% | 54.34% | Patient-level cross-validation |
| RF/SVM | — | 80–87% | Cell vs non-cell only |
| Omni-Fuse (similar HSI pathology) | — | — | mIoU ~69% |
| MCL-Net (similar HSI pathology) | 90.48% | — | mIoU ~73% |

---

## Why the results are NOT directly comparable

1. **Spatial leakage:** GHOST trained and tested on the same 512×512 tissue region (stratified pixel split). Published benchmarks use patient-level cross-validation (train on some patients, test on others). Spatial correlation inflates our metrics.

2. **Single image:** We used 1 of 62 images. Published results use the full dataset across 10 patients.

3. **Class imbalance:** Background is 64% of pixels and gets masked. The tumor class (where clinical value lies) forms a large contiguous blob in our crop — spatially easy to segment. The truly hard task (detecting scattered non-tumor cells, Class 1) got our weakest result (68% IoU).

4. **No patient generalization tested:** The real clinical question is whether the model generalizes across patients. We haven't tested this.

---

## What this proves

- **GHOST is data-agnostic:** The same architecture that segments satellite remote sensing imagery (Indian Pines, Pavia, Salinas) also segments medical pathology slides — with zero code changes.
- **The pipeline works end-to-end** on ENVI-format microscopic HSI data after a simple .mat conversion.
- **The numbers are promising** but not benchmark-comparable due to evaluation setup differences.

## What this does NOT prove

- That GHOST beats SOTA on medical HSI segmentation
- That the model generalizes across patients
- Clinical readiness or reliability for diagnostic use

---

## For LinkedIn

Use this as a **visual demo of data-agnostic capability**, not a benchmark claim. The 3 remote sensing datasets (Indian Pines, Pavia, Salinas) with proper evaluation are the real proof of performance. The LUSC result is a compelling "and it works on medical data too" teaser for future work.

## Future work (v2)

- Multi-image training support (train across all 62 images)
- Native ENVI/GeoTIFF format support (no manual conversion)
- Patient-level cross-validation for fair benchmarking
- Proper comparison against HybridSN-Att and other published methods
