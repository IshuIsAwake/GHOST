# GHOST — Results

All results obtained on NVIDIA RTX 3050 (6 GB VRAM, laptop).
Default splits: `--train_ratio 0.2 --val_ratio 0.1 --seed 42`.

---

## Indian Pines

200 bands, 16 classes, 145x145 pixels.

### SPT + Dice Loss (recommended)

| Config | OA | mIoU | Dice | Kappa | AA | Time |
|--------|-----|------|------|-------|----|------|
| 64 base / 32 num filters | **97.52%** | **0.8593** | 0.9038 | 0.9717 | — | 6h 2m* |
| 32 base / 8 num filters | **97.55%** | 0.8027 | 0.8391 | 0.9721 | — | 77m |

\* The 64/32 run was affected by a Conv3D performance regression (fixed in v0.1.4). Actual time on current code will be lower.

<details>
<summary>Per-class IoU (64/32 config)</summary>

| Class | IoU | Class | IoU |
|-------|------|-------|------|
| 1 | 0.9143 | 9 | 0.1905 |
| 2 | 0.9351 | 10 | 0.9547 |
| 3 | 0.9425 | 11 | 0.9689 |
| 4 | 0.8811 | 12 | 0.9607 |
| 5 | 0.9415 | 13 | 0.9592 |
| 6 | 0.9961 | 14 | 0.9694 |
| 7 | 0.4286 | 15 | 0.9890 |
| 8 | 0.9256 | 16 | 0.7910 |

Classes 7 and 9 have <30 training samples. Low IoU on these is expected and consistent with published literature.

</details>

### Ablation: Flat Model vs SPT

Same dataset, same hyperparameters (32 base / 8 num filters), CE loss.

| Model | OA | mIoU | Dice |
|-------|-----|------|------|
| Flat (no SPT) | 89.12% | 0.5179 | 0.3712 |
| SPT + CE (ensemble routing) | 94.49% | 0.6856 | 0.7339 |
| SPT + CE+Dice (ensemble routing) | 97.04% | 0.7851 | 0.8257 |

---

## Salinas Valley

204 bands, 16 classes, 512x217 pixels.

| Config | OA | mIoU | Dice | Kappa | Time |
|--------|-----|------|------|-------|------|
| 16 base / 4 num filters | 92.4% | 0.7668 | 0.8276 | 0.9154 | 4h 11m* |

\* Time from an older run with the Conv3D performance issue. Pending re-run.

---

## Pavia University

103 bands, 9 classes, 610x340 pixels.

Pending re-run with latest pipeline.

---

## LUSC (Lung Squamous Cell Carcinoma)

61 bands, 3 classes. Microscopic histopathology (H&E-stained lung tissue).
Source: [HMI-LUSC dataset](https://github.com/Intelligent-Imaging-Center/HMILungDataset).

### Caveats

These results are **not benchmark-comparable**:

1. **Single image only** — trained on 1 of 62 images (a 512x512 crop), not the full dataset
2. **Spatial leakage** — train/test split is pixel-level within the same tissue region, not patient-level cross-validation as in published benchmarks
3. **Class distribution** — the tumor class forms a large contiguous blob in this crop, making it spatially easy to segment
4. **No patient generalisation tested**

These numbers demonstrate that GHOST handles non-remote-sensing HSI data without code changes. They are not SOTA claims.

| Metric | Value |
|--------|-------|
| OA | 99.43% |
| mIoU | 88.96% |
| Dice | 93.49% |
| Kappa | 0.9878 |

<details>
<summary>Per-class IoU</summary>

| Class | IoU | Notes |
|-------|-----|-------|
| 1 (non-tumor cells) | 68.36% | Only 953 pixels — scattered small dots |
| 2 (tumor cells) | 99.14% | Large contiguous mass |
| 3 (non-cell tissue) | 99.37% | Large contiguous region |

</details>

<details>
<summary>Published methods (for context, not direct comparison)</summary>

| Model | OA | Notes |
|-------|-----|-------|
| HybridSN-Att | 92.52% | Patient-level cross-validation, 10 patients |
| MCL-Net | 90.48% | mIoU ~73% |
| Omni-Fuse | — | mIoU ~69% |

These use the full dataset with proper evaluation protocols. Direct comparison with GHOST's single-crop results is not valid.

</details>

---

## Mars CRISM

Planetary remote sensing data. Tested by a collaborator during a hackathon. Results pending independent re-run.

---

## Notes

- All runs use `--routing forest` (ensemble-based inference through the SPT)
- Hybrid and soft routing modes exist but are experimental — ensemble routing is recommended
- Training times marked with \* were affected by a now-fixed Conv3D performance issue and will be lower on v0.1.4+
