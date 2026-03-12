# GHOST Architecture — Deep Reference

This document is for researchers and developers who want to understand the design decisions, mathematics, and tradeoffs behind every component of GHOST.

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Continuum Removal](#1-continuum-removal)
3. [Spectral 3D Convolution Stack](#2-spectral-3d-convolution-stack)
4. [Squeeze-and-Excitation Block](#3-squeeze-and-excitation-block)
5. [2D U-Net Encoder / Decoder](#4-2d-u-net-encoder--decoder)
6. [RSSP — Recursive Spectral Splitting](#5-rssp--recursive-spectral-splitting)
7. [SSSR — Selective Spectral State Routing](#6-sssr--selective-spectral-state-routing)
8. [Training Strategy](#7-training-strategy)
9. [Known Limitations](#8-known-limitations)

---

## Design Philosophy

GHOST is built around one constraint: **no dataset-specific assumptions**.

Most hyperspectral deep learning papers hardcode band counts, class counts, patch sizes, and normalisation strategies. This produces models that achieve 95%+ OA on Indian Pines and fail completely when pointed at a Mars CRISM observation.

Every component in GHOST is conditioned on the data at runtime:
- Band count `C` is read from the `.mat` file
- Class count is inferred from `labels.max()`
- The RSSP tree topology is computed from the dataset's own spectral statistics
- The SSM encoder adapts its kernel scales to `C`

The result is a model that degrades gracefully rather than breaking when dataset properties change.

---

## 1. Continuum Removal

**File:** `preprocessing/continuum_removal.py`

### What it does

For each pixel spectrum `s(λ)`, the continuum is defined as the linear interpolation between the spectrum's minimum and maximum values across bands:

```
continuum(λ) = s_min + (s_max - s_min) × (λ / C)
```

The continuum-removed spectrum is:

```
CR(λ) = s(λ) / (continuum(λ) + ε)
```

### Why it matters

Raw reflectance is a product of two signals: the material's intrinsic absorption features (what we want) and the overall brightness/albedo (sensor gain, illumination, atmospheric scattering — what we don't want). Continuum removal isolates absorption features by normalising away brightness.

This is standard practice in geological spectroscopy. GHOST applies it as a differentiable nn.Module at the front of every forward pass, making it part of the learned pipeline rather than a preprocessing script.

### Why fp32 is enforced here

Continuum removal involves division of small floating point numbers. In fp16, the denominator `continuum + ε` can underflow to zero, producing NaN gradients. The forward pass explicitly casts to fp32 inside a `torch.autocast('cuda', enabled=False)` context regardless of the training precision setting.

---

## 2. Spectral 3D Convolution Stack

**File:** `models/spectral_3d_block.py`

### Architecture

Input `(B, C, H, W)` is reshaped to `(B, 1, C, H, W)` — treating the channel dimension as a volumetric depth. `num_blocks` layers of 3D convolution are then applied:

```
kernel_size = (7, 3, 3)   # (spectral, spatial_h, spatial_w)
```

Each block: `Conv3d → BatchNorm3d → ReLU`

Output is reshaped to `(B, num_filters × C, H, W)`.

### Why 3D convolution

Standard 2D convolution on hyperspectral data treats spectral bands as independent channels. This loses the *ordering* of bands — band 100 is adjacent to band 101 in wavelength, which carries physical meaning (absorption features have smooth spectral profiles).

3D convolution with a spectral kernel of size 7 explicitly models local spectral correlations: each output feature depends on 7 consecutive bands, not just one. The spatial kernels (3×3) additionally couple this with local spatial context simultaneously.

### Output channels

`num_filters × C` — for default settings (8 filters, 200 bands) this produces 1600 feature channels. This is intentionally large: the SE block immediately following is responsible for compressing this to the meaningful subset.

---

## 3. Squeeze-and-Excitation Block

**File:** `models/se_block.py`

### Architecture

```
x (B, C, H, W)
→ GlobalAvgPool → (B, C)
→ Linear(C, C/r) → ReLU
→ Linear(C/r, C) → Sigmoid
→ w (B, C, 1, 1)
→ x * w
```

Default reduction ratio `r = 16`.

### Purpose

After the 3D stack, we have 1600 channels. Many are redundant or irrelevant for a given dataset. SE block learns a per-channel scalar weight — effectively asking "which spectral feature maps matter?" — and rescales the feature volume accordingly.

This is the network's first opportunity to do learned spectral selection after the physics-informed continuum removal. Together they form a two-stage spectral normalisation: physics first, learned second.

---

## 4. 2D U-Net Encoder / Decoder

**Files:** `models/encoder_2d.py`, `models/decoder_2d.py`

### Encoder

Four downsampling stages with MaxPool2d(2):

```
enc1: (in_channels) → f          @ full resolution
enc2: f             → f×2        @ /2
enc3: f×2           → f×4        @ /4
enc4: f×4           → f×8        @ /8
bottleneck: f×8     → f×16       @ /16
```

Each stage is a `ConvBlock`: `Conv2d(3×3) → BN → ReLU → Dropout2d(0.3) → Conv2d(3×3) → BN → ReLU`

### Decoder

Symmetric upsampling with `ConvTranspose2d(2×2, stride=2)` followed by skip connection concatenation:

```
up4: f×16 → f×8   concat with enc4 → dec4: f×16 → f×8
up3: f×8  → f×4   concat with enc3 → dec3: f×8  → f×4
up2: f×4  → f×2   concat with enc2 → dec2: f×4  → f×2
up1: f×2  → f     concat with enc1 → dec1: f×2  → f
final: f → num_classes (1×1 conv)
```

Size mismatches between upsampled feature maps and skip connections are resolved by bilinear interpolation — this handles non-power-of-two input dimensions gracefully.

### Why U-Net for hyperspectral

Hyperspectral segmentation requires both local spectral precision (which mineral is this specific pixel?) and spatial context (what are the surrounding pixels?). The encoder captures multi-scale spatial context; skip connections preserve fine spatial detail lost during downsampling. This combination has been empirically validated across many dense prediction tasks.

---

## 5. RSSP — Recursive Spectral Splitting

**Files:** `rssp/sam_clustering.py`, `rssp/rssp_trainer.py`

### Motivation

A flat 16-class classifier must simultaneously distinguish classes that are spectrally very different (e.g. water vs. corn) and classes that are nearly identical (e.g. corn-notill vs. corn-mintill). These are fundamentally different problems of different difficulty. Forcing a single model to solve both simultaneously wastes capacity and produces confusion in the hard cases.

RSSP builds a divide-and-conquer hierarchy: spectrally distant classes are separated first, spectrally similar classes are handled by a specialist model that only sees their subset.

### Tree Construction

**Step 1: Class mean spectra**

For each class `c`, compute the mean spectrum across all labeled pixels:

```
μ_c = mean({s(r,c) : labels[r,c] == c})     shape: (C,)
```

**Step 2: Continuum removal on means**

Apply CR to each mean spectrum to get normalised spectral shapes.

**Step 3: SAM distance matrix**

Spectral Angle Mapper distance between classes `i` and `j`:

```
SAM(i, j) = arccos( μ_i · μ_j / (||μ_i|| × ||μ_j||) )
```

Returns angle in radians. 0 = identical spectra. π/2 = orthogonal spectra.

**Step 4: Recursive binary splitting**

At each node, find the two most spectrally distant classes (maximum SAM distance) — these become the left and right seeds. All remaining classes are assigned to the nearest seed. A pixel balance check prevents severe imbalance (>3:1 ratio) with a single class swap.

**Stopping conditions (auto mode):**
- Node has ≤ 2 classes
- Any class has fewer than `min_pixels=10` labeled pixels
- Depth ≥ 3
- Mean intra-node SAM distance < `sam_threshold=0.05` (classes already very similar)

### Node Training

Each internal and leaf node trains an independent `HyperspectralNet` forest. Classes are re-labelled locally: global IDs {3, 7, 12} → local IDs {1, 2, 3}. The node model is a standard `num_local_classes + 1` classifier (background = 0, ignored in loss).

**Epoch budget scaling:**
```
node_epochs = max(base_epochs // 2,
                  int(base_epochs × len(node_classes) / total_classes))
```
Root always gets `base_epochs`. Smaller nodes get proportionally fewer epochs — they have simpler problems.

**Forest ensemble:**
`num_forests` independent models are trained per node with different random seeds. At inference, their softmax probabilities are averaged before taking argmax. This is the primary defence against bad random initialisations.

---

## 6. SSSR — Selective Spectral State Routing

**Files:** `models/spectral_ssm.py`, `rssp/sssr_router.py`, `rssp/ssm_pretrain.py`

### Motivation

RSSP's cascade inference routes each pixel to a child node based on the parent node's argmax prediction. If the root misclassifies a pixel (even with 51% confidence), it goes to the wrong subtree permanently.

The forest ensemble partially mitigates this but cannot fix a systematic routing error. SSSR replaces hard routing with **soft probabilistic routing** — pixels flow down both branches, weighted by learned spectral fingerprint similarity.

### Spectral Fingerprint Encoder

`SpectralSSMEncoder` processes continuum-removed spectra through three parallel multi-scale 1D convolutional branches:

```
narrow branch: Conv1d(kernel=7)  × 2   — local absorption features
mid    branch: Conv1d(kernel=15) × 2   — medium-range spectral shape
wide   branch: Conv1d(kernel=31) × 2   — broad continuum trends
```

The three branches are concatenated and passed through a channel attention gate (squeeze-and-excitation style) — selecting which scale of spectral information is most relevant for this pixel. Global average pooling then produces a fixed-size `d_model`-dimensional fingerprint regardless of input `C`.

**Why parallel convolutions, not sequential SSM:**
A sequential scan over 200 bands suffers vanishing gradients — the signal from band 1 barely influences the output at band 200. Parallel convolutions have direct gradient paths to every band, train stably, and the multi-scale design preserves the "selective attention to different spectral scales" spirit of SSMs.

**Pretraining:**
The encoder is pretrained as a pixel-level 16-class classifier using only `train_coords`. The classification head is discarded after pretraining — only the encoder weights are kept. This gives the encoder a discriminative spectral prior before it's frozen.

### Routing Heads

At each **internal** RSSP node, a lightweight routing head is trained:

```
SSSRRouter: Linear(d_model, 32) → ReLU → Linear(32, 1) → Sigmoid
```

Ground truth: `target = 1.0` if pixel ∈ left_classes, `0.0` if pixel ∈ right_classes. Loss: binary cross-entropy.

The head is tiny — approximately `d_model × 32 + 32 × 1 ≈ 2080` parameters per node. Negligible cost.

### Soft Cascade Inference

At each internal node, the router produces `p_left(r,c) ∈ (0,1)` per pixel. The pixel's contribution to each subtree is weighted:

```
weight_left  = path_weight × p_left
weight_right = path_weight × (1 - p_left)
```

Weights multiply **down the tree** — a pixel confident at every node accumulates weight ≈ 1.0 at its correct leaf. An ambiguous pixel spreads its weight across multiple leaves and benefits from the ensemble effect of all of them.

Final prediction:

```
pred(r,c) = argmax over classes of Σ_leaf (weight_leaf(r,c) × prob_leaf(r,c))
```

---

## 7. Training Strategy

### Loss function

`CrossEntropyLoss(ignore_index=0)` — background pixels (class 0) are excluded from loss computation. This is critical for sparse label settings where most pixels are unlabeled.

### Optimiser

`AdamW` with `weight_decay=1e-4`. `ReduceLROnPlateau` with `patience=10, factor=0.5` — halves learning rate when val loss plateaus for 10 evaluation intervals.

### Evaluation cadence

Validation every 20 epochs per node. Best checkpoint selected by `val_mIoU` (not OA) — mIoU is more sensitive to minority class performance, which is the harder and more important problem.

### Split strategy

Stratified split by class: for each class independently, shuffle its labeled pixels, take `train_ratio` for training, `val_ratio` for validation, remainder for test. This guarantees every class appears in every split, even for rare classes with 20–50 total pixels.

---

## 8. Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Sequential SSM scan causes vanishing gradients | Replaced with parallel multi-scale convolutions | Slight loss of long-range spectral memory |
| Conv kernel sizes (7,15,31) assume C ≥ 50 | Sensors with very few bands (e.g. RGB) may have overlapping kernels | Kernels scale with C automatically |
| RSSP tree built from mean spectra | Rare classes with few pixels have noisy mean estimates | `min_pixels` threshold prevents splitting on unreliable means |
| Soft routing requires SSM pretraining | Adds a pretraining step before main training | Reusable across runs on the same dataset via `--ssm_load` |
| 6GB VRAM constraint | Requires halving filters for large spatial datasets (Pavia) | `--base_filters 16 --num_filters 4` for constrained environments |
| Hard routing for single-class leaf nodes | Leaf nodes with one class have trivial routing but no model | Prediction defaults to that class with weight 1.0 |