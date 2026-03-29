# GHOST — Results

These are the numbers I've gotten so far. Posting them mostly for my own reference and for anyone curious.

**Important caveats that apply to everything here:**
- All evaluation is pixel-level train/test split on a single scene. This is standard for these benchmarks, but it means spatial autocorrelation inflates the numbers — pixels near each other tend to look alike, and the train/test pixels are spatially interleaved. Real-world generalization would require scene-level or at minimum spatially disjoint splits, which I haven't done.
- I've only run each configuration once or twice. The two Indian Pines runs showed <1% variance, which is encouraging, but I wouldn't read too much into the exact numbers.
- All runs use `--routing forest` (ensemble-based inference through the SPT) and ce+dice loss unless noted.

Default splits: `--train_ratio 0.2 --val_ratio 0.1 --seed 42`.

**Hardware:**
- **RTX 3050** — laptop GPU (6 GB VRAM). Used for Indian Pines and LUSC.
- **Kaggle T4** — single Tesla T4 (16 GB VRAM). Used for Salinas and Pavia.

---

## Salinas Valley

204 bands, 16 classes, 512×217 pixels. Hardware: Kaggle T4.

| Config | OA | mIoU | Dice | Kappa | Precision | Recall | AA | Time |
|--------|-----|------|------|-------|-----------|--------|----|------|
| 32 base / 8 num filters | 98.69% | 0.9577 | 0.9771 | 0.9855 | 0.9843 | 0.9732 | 0.9732 | 10h 51m |

<details>
<summary>Per-class IoU</summary>

| Class | IoU | Class | IoU |
|-------|------|-------|------|
| 1 | 0.9601 | 9 | 0.9961 |
| 2 | 0.9709 | 10 | 0.9769 |
| 3 | 0.9942 | 11 | 0.7387 |
| 4 | 0.9929 | 12 | 0.8744 |
| 5 | 0.9952 | 13 | 0.9704 |
| 6 | 0.9910 | 14 | 0.8265 |
| 7 | 0.9664 | 15 | 0.9782 |
| 8 | 0.9859 | 16 | 0.8819 |

Class 11 (Lettuce-romaine 4wk, IoU 0.74) and Class 14 (IoU 0.83) are the weakest. Class 12 (Lettuce-romaine 5wk, IoU 0.87) is also below 0.90 — these are visually similar lettuce growth stages that are hard to separate spectrally, so this isn't surprising.

</details>

---

## Pavia University

103 bands, 9 classes, 610×340 pixels. Hardware: Kaggle T4.

| Config | OA | mIoU | Dice | Kappa | Precision | Recall | AA | Time |
|--------|-----|------|------|-------|-----------|--------|----|------|
| 32 base / 8 num filters | 97.47% | 0.9531 | 0.9755 | 0.9667 | 0.9681 | 0.9843 | 0.9843 | 7h 29m |

<details>
<summary>Per-class IoU</summary>

| Class | IoU |
|-------|------|
| 1 (Asphalt) | 0.9951 |
| 2 (Meadows) | 0.9569 |
| 3 (Gravel) | 0.9340 |
| 4 (Trees) | 0.8577 |
| 5 (Metal sheets) | 0.9958 |
| 6 (Bare soil) | 0.9175 |
| 7 (Bitumen) | 0.9989 |
| 8 (Bricks) | 0.9615 |
| 9 (Shadows) | 0.9669 |

Class 4 (Trees, IoU 0.86) is the weakest — tree canopy pixels are spectrally mixed with meadow and soil.

</details>

---

## Indian Pines

200 bands, 16 classes, 145×145 pixels. Hardware: RTX 3050 (laptop).

### SPT + CE+Dice Loss

| Config | OA | mIoU | Dice | Kappa | Precision | Recall | AA | Time | VRAM |
|--------|-----|------|------|-------|-----------|--------|----|------|------|
| 64 base / 16 num filters | 98.16% | 0.9071 | 0.9454 | 0.9790 | 0.9687 | 0.9294 | 0.9294 | 2h 20m | 3.6 GB |
| 32 base / 8 num filters | 97.20% | 0.8030 | 0.8390 | 0.9681 | 0.8794 | 0.8272 | 0.8272 | 1h 17m | 3.4 GB |

Two runs of the 32/8 config: Run 1: 97.20% OA / 0.8030 mIoU, Run 2: 97.41% OA / 0.7962 mIoU. Less than 1% variance, but that's only two runs so take it for what it is.

<details>
<summary>Per-class IoU (32/8 config)</summary>

| Class | IoU | Test Pixels | Precision | Recall |
|-------|------|-------------|-----------|--------|
| 1 | 0.8788 | 33 | 1.0000 | 0.8788 |
| 2 | 0.9630 | 1001 | 0.9630 | 1.0000 |
| 3 | 0.9536 | 581 | 0.9616 | 0.9914 |
| 4 | 0.8956 | 167 | 0.9157 | 0.9760 |
| 5 | 0.9282 | 339 | 0.9359 | 0.9912 |
| 6 | 0.9902 | 511 | 1.0000 | 0.9902 |
| 7 | 0.0000 | 21 | 0.0000 | 0.0000 |
| 8 | 0.9554 | 336 | 1.0000 | 0.9554 |
| 9 | 0.0667 | 14 | 0.5000 | 0.0714 |
| 10 | 0.9211 | 681 | 0.9953 | 0.9251 |
| 11 | 0.9651 | 1719 | 0.9682 | 0.9971 |
| 12 | 0.9181 | 416 | 0.9208 | 0.9976 |
| 13 | 0.9592 | 144 | 0.9792 | 0.9792 |
| 14 | 0.9630 | 886 | 0.9630 | 1.0000 |
| 15 | 0.9745 | 271 | 0.9853 | 0.9889 |
| 16 | 0.5147 | 66 | 1.0000 | 0.5147 |

Classes 7 (21 px) and 9 (14 px) have <30 test pixels. Getting zero or near-zero IoU on these is basically expected — there just isn't enough data. This is a known issue in Indian Pines that shows up in published work too. Class 16 (66 px) also suffers from this.

</details>

<details>
<summary>Per-class IoU (64/16 config)</summary>

| Class | IoU |
|-------|------|
| 1 | 0.9143 |
| 2 | 0.9491 |
| 3 | 0.9714 |
| 4 | 0.8983 |
| 5 | 0.9600 |
| 6 | 0.9961 |
| 7 | 0.6667 |
| 8 | 0.9554 |
| 9 | 0.5000 |
| 10 | 0.9634 |
| 11 | 0.9728 |
| 12 | 0.9607 |
| 13 | 0.9863 |
| 14 | 0.9715 |
| 15 | 0.9963 |
| 16 | 0.8507 |

The bigger model helps the rare classes quite a bit: Class 7 goes from 0.00 → 0.67, Class 9 from 0.07 → 0.50, Class 16 from 0.51 → 0.85. Still not great, but at least it's learning something. Whether this is just memorization on such small samples, I honestly don't know.

</details>

### Ablation: flat model vs SPT

Same dataset, same hyperparameters (32 base / 8 num filters), CE loss. Hardware: RTX 3050 (laptop).

| Model | OA | mIoU | Dice |
|-------|-----|------|------|
| Flat (no SPT) | 89.12% | 0.5179 | 0.3712 |
| SPT + CE (ensemble routing) | 94.49% | 0.6856 | 0.7339 |
| SPT + CE+Dice (ensemble routing) | 97.20% | 0.8030 | 0.8257 |

The SPT clearly helps here, but this is one dataset with one set of hyperparameters. I'd want to see this hold up on more datasets before drawing strong conclusions.

---

## LUSC (Lung Squamous Cell Carcinoma)

61 bands, 3 classes, 512×512 crop. Microscopic histopathology (H&E-stained lung tissue). Hardware: RTX 3050 (laptop).
Source: [HMI-LUSC dataset](https://github.com/Intelligent-Imaging-Center/HMILungDataset).

### Why these numbers don't mean much

I want to be upfront: these results look great on paper but the evaluation setup is deeply flawed for drawing any real conclusions:

1. **Single image** — trained on 1 of 62 images (a 512×512 crop), not the full dataset
2. **Spatial leakage** — train/test pixels are from the same tissue region, not patient-level splits
3. **Easy geometry** — the tumor class forms a big contiguous blob in this crop
4. **No generalization tested** — haven't tried other patients or crops

I'm including these mainly to show that the pipeline doesn't crash on non-remote-sensing HSI data. The numbers themselves shouldn't be compared to anything.

### SPT + Dice Loss

| Config | OA | mIoU | Dice | Kappa | Precision | Recall | AA | Time | VRAM |
|--------|-----|------|------|-------|-----------|--------|----|------|------|
| 32 base / 8 num filters | 99.42% | 0.9263 | 0.9593 | 0.9876 | 0.9293 | 0.9969 | 0.9969 | 1h 8m | 5.6 GB |

<details>
<summary>Per-class IoU</summary>

| Class | IoU | Notes |
|-------|-----|-------|
| 1 (non-tumor cells) | 0.7971 | Smallest class — scattered small dots |
| 2 (tumor cells) | 0.9908 | Large contiguous mass |
| 3 (non-cell tissue) | 0.9908 | Large contiguous region |

</details>

<details>
<summary>Published methods (for context only — not a fair comparison)</summary>

| Model | OA | Notes |
|-------|-----|-------|
| HybridSN-Att | 92.52% | Patient-level cross-validation, 10 patients |
| MCL-Net | 90.48% | mIoU ~73% |
| Omni-Fuse | — | mIoU ~69% |

These use the full dataset with proper evaluation. Comparing GHOST's single-crop numbers to these would be misleading.

</details>

---

## Notes

- Hybrid and soft routing modes exist but are experimental and worse than ensemble routing in every test I've done
- Salinas and Pavia were run on Kaggle (T4) because they need more VRAM than my laptop has
- All reported metrics are from `ghost predict` (standalone inference on saved weights), not training-time evaluation. There's a known issue with model weights getting corrupted when downloading from Kaggle — see [limitations.md](limitations.md)
