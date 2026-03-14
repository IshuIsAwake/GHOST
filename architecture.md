# GHOST Architecture — Complete Technical Reference

> *"We are not building a `dataset002_BraTS` or a `dataset004_Hippocampus`. We are building the nnUNet itself."*

This document is the authoritative reference for every architectural decision in GHOST. Each section covers four layers of explanation: the intuition, the analogy, the formal motivation, and the mathematics. Whether you are a domain scientist, a machine learning engineer, or a first-year student, this document is written for you.

---

## Table of Contents

1. [What Problem Are We Solving?](#1-what-problem-are-we-solving)
2. [Design Philosophy](#2-design-philosophy)
3. [System Overview](#3-system-overview)
4. [Continuum Removal](#4-continuum-removal)
5. [Spectral 3D Convolution Stack](#5-spectral-3d-convolution-stack)
6. [Squeeze-and-Excitation Block](#6-squeeze-and-excitation-block)
7. [2D U-Net Encoder and Decoder](#7-2d-u-net-encoder-and-decoder)
8. [RSSP — Recursive Spectral Splitting with Parallel Forests](#8-rssp--recursive-spectral-splitting-with-parallel-forests)
9. [SSSR — Selective Spectral State Routing](#9-sssr--selective-spectral-state-routing)
10. [Training Strategy](#10-training-strategy)
11. [Inference Pipeline](#11-inference-pipeline)
12. [Known Limitations and Roadmap](#12-known-limitations-and-roadmap)

---

## 1. What Problem Are We Solving?

### In Plain English

A regular camera captures three numbers per pixel — red, green, and blue. A hyperspectral camera captures hundreds of numbers per pixel, one for each narrow slice of the electromagnetic spectrum. This means instead of knowing a pixel is "greenish," you know its exact reflectance at 450nm, 451nm, 452nm... all the way to 2500nm. Different materials — minerals, crops, water, asphalt, tumors — absorb and reflect light at different wavelengths in ways that are as unique as fingerprints.

The challenge is: given this enormous cube of data, automatically label every pixel with the correct material class.

### Why Is It Hard?

- **High dimensionality.** 200 bands × 145 × 145 pixels = 4.2 million numbers per image.
- **Spatial correlation.** Adjacent pixels are usually the same material. Context matters.
- **Spectral correlation.** Adjacent bands are physically related. Band 100 and band 101 are almost identical.
- **Class imbalance.** Some materials cover thousands of pixels; others cover twenty.
- **No universal model.** A model trained on Indian farmland crops fails completely on Martian minerals.

### What GHOST Does Differently

Almost every published hyperspectral deep learning paper trains a model on one specific dataset with hardcoded assumptions — fixed band count, fixed class count, fixed preprocessing. These models do not generalise. GHOST is built to be **completely data-agnostic**: band count, class count, image dimensions, and data distribution are all read at runtime. You point it at any `.mat` file and it works.

---

## 2. Design Philosophy

GHOST is built around one constraint: **no dataset-specific assumptions**.

Every component is conditioned on the data at runtime:

| Property | How GHOST handles it |
|---|---|
| Band count `C` | Read from `.mat` file, flows through entire architecture |
| Class count `K` | Inferred from `labels.max()` |
| Image dimensions `H × W` | U-Net handles any size via bilinear interpolation |
| Data distribution | Z-score normalisation computed per dataset |
| Class difficulty | RSSP tree topology computed from dataset's own spectral statistics |

The inspiration is **nnUNet** — a medical image segmentation framework that automatically configures itself for any new dataset and consistently achieves near state-of-the-art results without manual tuning. GHOST applies the same philosophy to hyperspectral imagery across every domain.

---

## 3. System Overview

The full GHOST pipeline processes a single hyperspectral image through six sequential stages:

```
RAW HYPERSPECTRAL IMAGE  (B, C, H, W)
         │
         ▼
┌─────────────────────┐
│   Continuum Removal  │  Physics-informed preprocessing
│   removes albedo     │  Isolates absorption features
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Spectral 3D Conv   │  Local spectral + spatial feature extraction
│  Stack              │  Models band adjacency explicitly
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   SE Block          │  Learned channel attention
│   channel weighting │  Selects relevant spectral features
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2D U-Net Encoder   │  Multi-scale spatial context
│  + skip connections │  Four downsampling stages
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2D U-Net Decoder   │  Full resolution prediction
│  segmentation map   │  Skip connections restore detail
└─────────────────────┘

════════════ RSSP LAYER ════════════

SAM-based spectral class tree (built once from data)

    Root: all classes
    /              \
LEFT subtree    RIGHT subtree
(spectrally     (spectrally
 similar)        similar)
  /    \           /    \
 ...   ...       ...    ...

Each node = independent forest of HyperspectralNets

════════════ SSSR LAYER ════════════

SpectralSSMEncoder (frozen after pretraining)
Produces per-pixel spectral fingerprints.

At each internal node, a tiny routing head
(fingerprint → p_left) replaces hard argmax
routing with soft probabilistic routing.

Final prediction = weighted sum across all leaves.
```

---

## 4. Continuum Removal

**File:** `ghost/preprocessing/continuum_removal.py`

### The Intuition

Imagine two photographs of the same field — one taken on a bright sunny day, one on a cloudy afternoon. The colours look completely different, but it's the same grass. If you tried to train a classifier on raw pixel values, it would struggle because the absolute brightness drowns out the actual material signature.

Geologists figured this out decades ago. They don't look at raw reflectance — they look at the *shape* of the spectrum relative to its own envelope. This is called continuum removal.

### The Analogy

Think of a spectrum like a mountain range seen from the side. The continuum is the straight line connecting the mountain peaks — the overall "backdrop." Continuum removal divides the actual landscape by this backdrop, leaving only the valleys (absorption features) standing out relative to a flat baseline. Two spectra with different overall brightnesses but the same absorption pattern will look identical after continuum removal.

### Why It Matters for Deep Learning

Without continuum removal, a neural network must simultaneously learn:
1. That material X has absorption at 1400nm
2. That this absorption holds whether the pixel is in shadow or in sunlight

With continuum removal, the network only needs to learn (1). The physics does (2) automatically. This is the single most important preprocessing step in reflectance spectroscopy and GHOST applies it as a differentiable `nn.Module` — meaning it is part of the computational graph, not a separate offline script.

### Mathematics

Given a pixel spectrum `s(λ)` with `C` bands indexed by `λ ∈ {0, 1, ..., C-1}`:

**Step 1 — Compute the continuum (linear envelope):**

```
s_min = min(s(λ))  over all λ
s_max = max(s(λ))  over all λ

continuum(λ) = s_min + (s_max - s_min) × (λ / (C - 1))
```

This is a linear ramp from the spectrum's minimum to its maximum, parameterised by band position.

**Step 2 — Divide to remove:**

```
CR(λ) = s(λ) / (continuum(λ) + ε)
```

where `ε = 1e-8` prevents division by zero.

**Properties:**
- CR values near 1.0 indicate the spectrum is close to its continuum (no absorption)
- CR values below 1.0 indicate absorption features (the meaningful signal)
- CR is invariant to multiplicative scaling of `s(λ)` — albedo and illumination are multiplicative

### Implementation Note: Why fp32 is Enforced

Continuum removal involves division of small floating point numbers. In fp16, the denominator `continuum + ε` can underflow to zero, producing NaN gradients. The forward pass explicitly casts to fp32 inside a `torch.autocast('cuda', enabled=False)` context regardless of the training precision setting. This is not a performance choice — it is a numerical stability requirement.

```python
with torch.autocast('cuda', enabled=False):
    x = self.continuum_removal(x.float())  # always fp32
```

---

## 5. Spectral 3D Convolution Stack

**File:** `ghost/models/spectral_3d_block.py`

### The Intuition

Standard 2D convolution on a hyperspectral image treats each band as an independent channel — like treating the pages of a book as if their order didn't matter. But band order matters enormously: band 100 at 1000nm is physically adjacent to band 101 at 1005nm, and absorption features are smooth curves that span dozens of consecutive bands.

3D convolution treats the spectral dimension as a volumetric depth, allowing the kernel to explicitly model "what do seven consecutive bands look like together?"

### The Analogy

Think of the hyperspectral cube as a loaf of sliced bread. 2D convolution looks at each slice independently. 3D convolution holds seven slices together and reads the pattern across them — it can detect that bands 98-104 together form an absorption trough, a feature that is invisible when looking at any single slice.

### Architecture

Input `(B, C, H, W)` is first reshaped to `(B, 1, C, H, W)` — inserting a singleton channel dimension so that 3D convolution treats the spectral axis as depth.

```
Input:  (B, C, H, W)
        ↓  unsqueeze(1)
        (B, 1, C, H, W)
        ↓  num_blocks × Single3DBlock
        (B, num_filters, C, H, W)
        ↓  reshape
Output: (B, num_filters × C, H, W)
```

Each `Single3DBlock`:
```
Conv3d(kernel=(7, 3, 3)) → BatchNorm3d → ReLU
```

Kernel dimensions:
- `7` along spectral axis — captures local absorption feature width (~35nm window at 5nm/band)
- `3 × 3` along spatial axes — captures immediate spatial neighbourhood

### Why the Kernel Size is 7

GHOST targets VNIR-SWIR sensors (400nm–2500nm) with approximately 200 bands, giving ~10nm per band. Major absorption features (water at 1400nm, hydroxyl at 2200nm) span 50-100nm, corresponding to 5-10 bands. A kernel of size 7 comfortably captures these features while remaining computationally tractable. The padding `(3, 1, 1)` ensures spatial dimensions are preserved.

### Output Channel Count

For default settings (`num_filters=8`, `num_blocks=3`, `C=200`):

```
out_channels = num_filters × C = 8 × 200 = 1600
```

This is intentionally large. The SE block that follows is responsible for compressing this to the meaningful subset. The philosophy: over-complete representation first, learned selection second.

### Mathematics

For block `k` with input `F^(k-1) ∈ R^(B × F_{k-1} × C × H × W)`:

```
F^(k) = ReLU(BN(W^(k) * F^(k-1) + b^(k)))
```

where `*` denotes 3D convolution and `W^(k) ∈ R^(F_k × F_{k-1} × 7 × 3 × 3)`.

Final reshape collapses the filter and spectral dimensions:

```
output(b, f·C + c, h, w) = F^(num_blocks)(b, f, c, h, w)
```

for `f ∈ {0,...,F-1}`, `c ∈ {0,...,C-1}`.

---

## 6. Squeeze-and-Excitation Block

**File:** `ghost/models/se_block.py`

### The Intuition

After the 3D stack, we have 1600 feature channels. Many are redundant — for a mineral mapping task, features encoding vegetation absorption bands are useless; for a crop classification task, it is the reverse. The SE block asks: "which channels matter for this particular image?" and rescales accordingly.

### The Analogy

Imagine you are a sommelier presented with a glass of wine. You don't pay equal attention to everything — you focus on the finish, the tannins, the fruit notes, and ignore the ambient noise. The SE block is the network learning where to focus its attention across the 1600 spectral feature channels.

### Architecture

```
x: (B, C, H, W)
→ GlobalAvgPool2d → (B, C)           # Squeeze: summarise spatial info
→ Linear(C, C/r) → ReLU              # Excitation: compact representation
→ Linear(C/r, C) → Sigmoid           # Excitation: gate weights ∈ (0,1)
→ w: (B, C, 1, 1)
→ x × w                              # Rescale each channel
```

Default reduction ratio `r = 16`.

### Why Global Average Pooling for Squeeze

The squeeze operation needs a single descriptor per channel that captures its overall response across the image. GlobalAvgPool is the simplest and most stable choice — it computes the mean activation of each channel across all spatial positions. This is a channel-level summary: "how active was this feature map overall?"

### Mathematics

**Squeeze:**
```
z_c = (1 / H·W) Σ_{h,w} x(c, h, w)     ∀c ∈ {1,...,C}
```

**Excitation:**
```
s = σ(W_2 · δ(W_1 · z))
```

where:
- `W_1 ∈ R^(C/r × C)` — dimensionality reduction
- `δ` — ReLU
- `W_2 ∈ R^(C × C/r)` — dimensionality restoration  
- `σ` — Sigmoid, producing `s ∈ (0,1)^C`

**Scale:**
```
x̃_c = s_c · x_c
```

### This is GHOST's First Learned Spectral Selection

Continuum removal is physics-based spectral normalisation. The SE block is **learned** spectral selection. Together they form a two-stage spectral normalisation pipeline: physics strips brightness, learning strips irrelevant features.

---

## 7. 2D U-Net Encoder and Decoder

**Files:** `ghost/models/encoder_2d.py`, `ghost/models/decoder_2d.py`

### The Intuition

After spectral processing, the problem becomes a standard dense prediction task: assign a class label to every pixel. The U-Net architecture is the established gold standard for dense prediction, having proven itself across medical image segmentation, satellite imagery, and now hyperspectral classification.

### The Analogy

The encoder is like looking at a map while progressively zooming out — at full resolution you see individual roads; at 1/16 resolution you see entire city districts. Each zoom level reveals different scales of context. The decoder then zooms back in, but now informed by the high-level context it gathered at each level. The skip connections are like sticky notes left at each zoom level that say "remember this detail when you zoom back in."

### Encoder Architecture

Four downsampling stages, each halving spatial resolution:

```
Input: (B, spectral_out, H, W)

enc1:  ConvBlock(spectral_out → f)       H × W
pool:  MaxPool2d(2)
enc2:  ConvBlock(f → f×2)               H/2 × W/2
pool:  MaxPool2d(2)
enc3:  ConvBlock(f×2 → f×4)             H/4 × W/4
pool:  MaxPool2d(2)
enc4:  ConvBlock(f×4 → f×8)             H/8 × W/8
pool:  MaxPool2d(2)
bottleneck: ConvBlock(f×8 → f×16)       H/16 × W/16
```

For `base_filters=32`: channel progression is 32 → 64 → 128 → 256 → 512.

Each `ConvBlock`:
```
Conv2d(3×3) → BN → ReLU → Dropout2d(0.3) → Conv2d(3×3) → BN → ReLU
```

The `Dropout2d(0.3)` randomly zeros entire feature maps during training — stronger regularisation than standard dropout because it forces the network to not rely on any single channel.

### Decoder Architecture

Symmetric upsampling with skip connections:

```
bottleneck (B, f×16, H/16, W/16)
↓  ConvTranspose2d(2×2, stride=2)
up4: (B, f×8, H/8, W/8) + enc4 → concat → ConvBlock → (B, f×8, H/8, W/8)
↓  ConvTranspose2d(2×2, stride=2)
up3: (B, f×4, H/4, W/4) + enc3 → concat → ConvBlock → (B, f×4, H/4, W/4)
↓  ConvTranspose2d(2×2, stride=2)
up2: (B, f×2, H/2, W/2) + enc2 → concat → ConvBlock → (B, f×2, H/2, W/2)
↓  ConvTranspose2d(2×2, stride=2)
up1: (B, f, H, W) + enc1 → concat → ConvBlock → (B, f, H, W)
↓  Conv2d(1×1)
output: (B, num_classes, H, W)
```

### Why Skip Connections?

Without skip connections, the bottleneck must encode everything. Fine spatial detail — exact boundaries between adjacent crop fields, pixel-precise edges — cannot survive the 16× downsampling. Skip connections directly wire encoder features to the corresponding decoder level, bypassing the information bottleneck. This is the core innovation of U-Net.

### Handling Non-Power-of-Two Dimensions

MaxPool2d with stride 2 on an odd dimension (e.g., 145 → 72) produces a size that cannot be exactly recovered by ConvTranspose2d. GHOST handles this with bilinear interpolation before each skip concatenation:

```python
def _match_size(self, x, target):
    if x.shape[2:] != target.shape[2:]:
        x = F.interpolate(x, size=target.shape[2:], 
                         mode='bilinear', align_corners=False)
    return x
```

This makes the architecture robust to any input dimension without requiring padding tricks.

### Mathematics

For a `ConvBlock` at encoder level `k`:

```
E_k = ReLU(BN(Conv(ReLU(BN(Conv(E_{k-1}))))))
```

For a decoder level `k`:

```
U_k = ConvTranspose(D_{k+1})
U_k = interpolate(U_k, size=E_k.shape[2:])   # if needed
D_k = ConvBlock(concat(U_k, E_k))
```

Final logits:
```
logits = Conv_{1×1}(D_1)    ∈ R^(B × K × H × W)
```

where `K` is the number of classes.

---

## 8. RSSP — Recursive Spectral Splitting with Parallel Forests

**Files:** `ghost/rssp/sam_clustering.py`, `ghost/rssp/rssp_trainer.py`

### The Core Problem RSSP Solves

A flat 16-class classifier must simultaneously distinguish:
- Water vs. corn (trivially different spectra — SAM distance ~0.8 radians)
- Corn-notill vs. corn-mintill (nearly identical spectra — SAM distance ~0.05 radians)

These are fundamentally different problems of different difficulty. Forcing a single model to solve both simultaneously wastes capacity on the easy cases and confuses the hard ones. The minority classes (rare crops with 20-50 pixels) get crushed by the dominant classes.

### The Analogy

Imagine a hospital that treats both broken legs and heart conditions with the same general practitioner. The GP is overwhelmed, makes more mistakes on both, and the rare conditions get misdiagnosed most often. Now imagine splitting into orthopedics and cardiology, each with its own specialist. Each specialist solves a simpler, more focused problem — and rare conditions within each specialty now get proper attention.

RSSP builds exactly this hierarchy, automatically, from the spectral statistics of the data itself.

### Step 1: Computing Class Mean Spectra

For each class `c ∈ {1,...,K}`:

```
μ_c = (1/|P_c|) Σ_{(r,col) ∈ P_c} s(r, col)
```

where `P_c` is the set of all labeled pixel coordinates for class `c` and `s(r, col) ∈ R^C` is the spectrum at position `(r, col)`.

After computing means, continuum removal is applied to each `μ_c` to get normalised spectral shapes that are comparable across classes.

### Step 2: Spectral Angle Mapper Distance Matrix

The Spectral Angle Mapper (SAM) distance between two spectra measures the angle between them in C-dimensional space:

```
SAM(μ_i, μ_j) = arccos( (μ_i · μ_j) / (‖μ_i‖ · ‖μ_j‖) )
```

**Why angle and not Euclidean distance?**

Euclidean distance is sensitive to brightness — two identical spectra at different illumination levels would appear far apart. The angle between two vectors is invariant to their magnitudes, measuring only the difference in *shape*. SAM = 0 means identical spectral shapes; SAM = π/2 means orthogonal (completely different) spectral shapes.

The full `K × K` SAM distance matrix `D` is computed once and used for all tree construction decisions.

### Step 3: Recursive Binary Splitting

The tree is built top-down. At each node with class set `C_node`:

**Finding seeds:**
```
(seed_a, seed_b) = argmax_{i,j ∈ C_node} D[i, j]
```

The two most spectrally distant classes become the seeds for the two groups. These are the classes that are most different from each other — they anchor the split.

**Assigning remaining classes:**

For each remaining class `c`:
```
if D[c, seed_a] ≤ D[c, seed_b]:
    assign c to group_a
else:
    assign c to group_b
```

This creates spectrally coherent groups — each group contains classes that are more similar to each other than to classes in the other group.

**Pixel balance correction:**

If one group has more than 3× the pixels of the other, the largest non-seed class in the bigger group is moved to the smaller group. This prevents severely imbalanced nodes where one subtree contains 95% of the data.

### Stopping Conditions (auto mode)

The tree stops splitting a node when any of the following are true:

| Condition | Reason |
|---|---|
| `len(classes) ≤ 2` | Cannot split fewer than 3 classes meaningfully |
| `min_pixels < 10` | Any class has too few pixels for stable training |
| `depth ≥ 3` | Prevents over-deep trees with tiny training sets |
| `mean_SAM < 0.05` | All classes are already spectrally very similar — splitting won't help |

### Step 4: Per-Node Forest Training

Each internal and leaf node trains an **independent forest** of `HyperspectralNet` models:

**Local relabelling:**

Global class IDs are remapped to local IDs for each node. For a node with classes `{3, 7, 12}`:
```
global → local: {3: 1, 7: 2, 12: 3}
```
Background is always 0 and is masked from the loss. This means each node trains a `(local_classes + 1)`-way classifier — much simpler than the full `K`-way problem.

**Epoch budget scaling:**

The root node gets the full `base_epochs` budget. Child nodes get proportionally fewer:

```
node_epochs = max(base_epochs // 2, 
                  int(base_epochs × len(node_classes) / total_classes))
```

A node with 2 classes gets far fewer epochs than a node with 10 classes — the problem is simpler and overfits faster with long training.

**Forest ensemble:**

`num_forests` independent models are trained per node with different random seeds. At inference, softmax probabilities are averaged before argmax:

```
P_ensemble = (1/F) Σ_{f=1}^{F} softmax(logits_f)
```

Ensemble averaging reduces variance — a single model initialised poorly might concentrate all probability on the wrong class; averaging across 5 models corrects this.

### Why RSSP Helps Minority Classes

A rare class with 20 pixels is completely overwhelmed at the root node when competing against a class with 2000 pixels. After RSSP splits, the rare class is grouped with 2-3 other spectrally similar classes. Now it represents 25-30% of the node's training pixels instead of 0.3%. The specialist model at this node can actually learn its features.

### Tree Example — Indian Pines (16 classes)

```
Root: [1,2,3,...,16]  ← all classes, biggest model
├── LEFT: [11,10,6,8,4,13,1]  ← grass/crop-like spectra
│   ├── LEFT: [11,1]  ← most similar pair
│   └── RIGHT: [10,6,8,4,13]
│       ├── LEFT: [10,4]
│       └── RIGHT: [6,8,13]
│           ├── LEFT: [6]  ← leaf: single class
│           └── RIGHT: [8,13]
└── RIGHT: [2,14,3,12,5,15,16,7,9]  ← woody/structural spectra
    ├── LEFT: [2,12,5,7,9]
    └── RIGHT: [14,3,15,16]
```

---

## 9. SSSR — Selective Spectral State Routing

**Files:** `ghost/models/spectral_ssm.py`, `ghost/rssp/sssr_router.py`, `ghost/rssp/ssm_pretrain.py`

### The Problem SSSR Solves

RSSP's cascade inference routes each pixel to a child node based on the parent node's prediction. If the root misclassifies a pixel — even with 51% confidence — it goes to the wrong subtree permanently. Hard routing turns every small error into a cascaded failure.

**Example:** A `corn-notill` pixel at the root is slightly misrouted toward the "woody/structural" subtree. Every subsequent node in that subtree is a specialist in the wrong classes. The pixel is permanently lost.

### The Analogy

Imagine a library classification system where you must choose Science or Humanities for every book, and then further subdivide. A book on the history of mathematics sits right on the boundary. Hard routing forces you to commit: it goes to Science and the Humanities specialists never see it. Soft routing says: put 60% of this book in Science and 40% in Humanities. Both sides contribute to the final answer, weighted by their confidence.

### Component 1: SpectralSSMEncoder — The Fingerprint Machine

The encoder produces a compact `d_model`-dimensional fingerprint for each pixel that captures its spectral identity independent of the tree structure.

**Why not use the forest's own features?**

The forest features are node-specific — the root encodes a 16-class problem, a leaf encodes a 2-class problem. The routing signal needs to be computed from features that are consistent across all nodes. A separately pretrained, frozen encoder provides this consistency.

**Architecture — Parallel Multi-Scale 1D CNN:**

```
Input: (N, C)  ← N pixel spectra, C bands
       ↓  unsqueeze(1)
       (N, 1, C)

Narrow branch: Conv1d(kernel=7)  × 2  → (N, d_state, C)
Mid branch:    Conv1d(kernel=15) × 2  → (N, d_state, C)
Wide branch:   Conv1d(kernel=31) × 2  → (N, d_state, C)

Concatenate: (N, 3·d_state, C)

Channel Attention Gate (SE-style):
  AdaptiveAvgPool1d(1) → (N, 3·d_state)
  Linear → GELU → Linear → Sigmoid
  → gate: (N, 3·d_state, 1)

Gated features: (N, 3·d_state, C)
GlobalAvgPool1d: (N, 3·d_state)
Linear + LayerNorm: (N, d_model)
```

**Why three scales?**

Spectral features exist at multiple scales:
- **Narrow (kernel=7, ~35nm):** Sharp absorption edges, fine feature tips
- **Mid (kernel=15, ~75nm):** Main absorption band widths
- **Wide (kernel=31, ~155nm):** Broad continuum shapes, inter-band relationships

No single scale captures all meaningful spectral information. The channel attention gate then learns which scale is most informative for each pixel.

**Why parallel convolutions and not sequential RNN/SSM?**

A sequential scan over 200 bands suffers vanishing gradients — the gradient signal from band 1 is attenuated through 200 time steps before it influences the output. Parallel convolutions have direct gradient paths to every band. The multi-scale design preserves the "selective attention to different spectral ranges" spirit of state space models without the training instability.

### Pretraining the Encoder

The encoder is pretrained as a pixel-level `K`-class classifier:

```
encoder → d_model fingerprint → Linear(d_model, K) → CrossEntropyLoss
```

Training uses only `train_coords` pixels with mini-batch sampling (`batch_size=512`). After pretraining, the classification head is discarded. Only the encoder weights are kept and **frozen for all subsequent training**. This gives the encoder a discriminative spectral prior before it produces routing signals.

### Component 2: SSSRRouter — The Routing Head

At each **internal** RSSP node, a lightweight routing head is trained:

```
SSSRRouter: Linear(d_model, 32) → ReLU → Linear(32, 1) → Sigmoid
→ p_left ∈ (0, 1)
```

**Ground truth for training:**

```
target = 1.0  if pixel ∈ left_classes
target = 0.0  if pixel ∈ right_classes
```

Loss: class-balanced binary cross-entropy:

```
w_left  = total / (2 × n_left)
w_right = total / (2 × n_right)

L = mean( w_i × BCE(p_left_i, target_i) )
```

The class balancing ensures that a node with 90% left-class pixels and 10% right-class pixels trains equally on both sides.

**Parameter cost:**

```
params per node = d_model × 32 + 32 + 32 × 1 + 1 ≈ d_model × 32 + 33
```

For `d_model=64`: ~2080 parameters per node. With 10 internal nodes: ~20,800 parameters total. Negligible.

### Hybrid Soft Cascade Inference

Three routing modes are supported:

**Mode 1: `--routing forest`**

The forest ensemble's class probabilities are used directly as routing signal:

```
forest_p_left = Σ_{c ∈ left_classes} P_ensemble[:, c, :, :]
```

This sums the probability mass assigned to left-subtree classes. If the forest gives 80% probability to corn-notill (a left-class), then `p_left = 0.80`. No SSM used.

**Mode 2: `--routing soft`**

The SSSR router head alone determines routing:

```
p_left = SSSRRouter(fingerprints)
```

Pure SSM-based routing. Degrades badly if SSM pretraining is weak.

**Mode 3: `--routing hybrid` (default)**

Forest provides a highly accurate base. SSM corrects when confident:

```
ssm_confidence = 2 × |ssm_p_left - 0.5|     ∈ [0, 1]
```

When `ssm_p_left = 0.5` (SSM is maximally uncertain): `confidence = 0`
When `ssm_p_left = 0.0` or `1.0` (SSM is maximally confident): `confidence = 1`

```
p_left = forest_p_left + confidence × (ssm_p_left - forest_p_left)
```

When SSM is uncertain, this reduces to `p_left = forest_p_left` — pure forest routing. When SSM is confident, it pulls toward its own prediction. SSSR is **purely additive**: worst case is identical to forest routing.

### Soft Path Weighting

Pixel contributions flow down the tree as weighted probabilities:

```
weight_left  = path_weight × p_left
weight_right = path_weight × (1 - p_left)
```

These weights multiply down the tree. A pixel that routes confidently at every node accumulates `weight ≈ 1.0` at its correct leaf. An ambiguous pixel spreads its weight across multiple leaves.

**Final prediction:**

```
pred(r, col) = argmax_c [ Σ_{leaf l} weight_l(r, col) × P_l(c, r, col) ]
```

The final class prediction is the argmax over the weighted sum of all leaf node probabilities. Ambiguous pixels benefit from the ensemble effect of multiple leaves rather than being committed to one.

---

## 10. Training Strategy

### Loss Function

```python
CrossEntropyLoss(ignore_index=0)
```

Background pixels (class 0) are excluded from loss computation. This is critical for sparse label settings — in Indian Pines, approximately 80% of pixels are unlabeled. Without `ignore_index=0`, the loss would be dominated by the trivial "predict background" signal.

### Optimiser

```
AdamW(weight_decay=1e-4)
```

AdamW decouples weight decay from the gradient update, providing better generalisation than standard Adam. The `1e-4` weight decay is a mild L2 regulariser.

### Learning Rate Schedule

```
ReduceLROnPlateau(patience=10, factor=0.5)
```

When validation loss plateaus for 10 evaluation intervals, the learning rate is halved. This allows aggressive early learning followed by fine-grained convergence without manual schedule design.

### Evaluation Cadence

Validation every 20 epochs per node. Best checkpoint selected by **validation mIoU**, not OA.

**Why mIoU over OA?**

Overall Accuracy is dominated by large classes. A model that achieves 95% OA by perfectly classifying Soybean-mintill (2455 pixels) and completely missing Oats (20 pixels) has performed poorly on the scientifically interesting minority classes. mIoU weights each class equally:

```
mIoU = (1/K) Σ_c [ TP_c / (TP_c + FP_c + FN_c) ]
```

A model that improves mIoU has genuinely improved on the hard cases, not just padded the dominant ones.

### Stratified Data Splits

For each class `c` independently:

```
shuffle(pixels_c)
train_c = pixels_c[:n × train_ratio]
val_c   = pixels_c[n × train_ratio : n × (train_ratio + val_ratio)]
test_c  = pixels_c[n × (train_ratio + val_ratio):]
```

This guarantees every class appears in every split, including rare classes with 20 pixels total. Without stratification, a rare class might land entirely in the test set, making training impossible.

---

## 11. Inference Pipeline

### Full Image Inference

Inference processes the entire image in a single forward pass through the cascade:

```python
weighted_probs = cascade_soft_inference(
    tree, trained_models,
    data.unsqueeze(0),      # (1, C, H, W)
    ssm_encoder, device, num_classes,
    routing='forest'
)
pred = weighted_probs.squeeze(0).argmax(dim=0)  # (H, W)
```

### Chunked Fingerprint Computation

For large scenes (Salinas at 512×217, Pavia at 610×340), computing the full fingerprint map in one GPU call exceeds VRAM. GHOST processes the image in row chunks:

```python
for row_start in range(0, H, chunk_size):
    row_end = min(row_start + chunk_size, H)
    chunk = data[:, row_start:row_end, :].unsqueeze(0).to(DEVICE)
    fp_chunk = ssm_encoder(chunk).squeeze(0).permute(1, 2, 0).cpu()
    fp_map[row_start:row_end, :, :] = fp_chunk
```

This is mathematically identical to full-image computation — chunking only affects memory, not the result.

### Metrics

Five metrics are computed over labeled pixels only (`labels > 0`):

| Metric | Formula | What it measures |
|---|---|---|
| OA | `(TP_all) / N` | Overall fraction correct |
| mIoU | `mean_c[TP_c / (TP_c + FP_c + FN_c)]` | Per-class overlap, equal weight |
| Dice | `mean_c[2TP_c / (2TP_c + FP_c + FN_c)]` | F1-score per class |
| Precision | `mean_c[TP_c / (TP_c + FP_c)]` | How clean the predictions are |
| Recall | `mean_c[TP_c / (TP_c + FN_c)]` | How complete the detection is |

---

## 12. Known Limitations and Roadmap

### Current Limitations

| Limitation | Impact | Planned Fix |
|---|---|---|
| Conv kernel sizes assume C ≥ 50 | Kernels (7,15,31) overlap for few-band sensors | Scale kernels proportionally to C |
| Full-image batch training | OOM on scenes larger than ~300×300 with full filters | Patch-based training |
| SSM pretraining uses uniform batches | Class imbalance causes majority-class collapse | Stratified batch sampling |
| Sequential tree training | No GPU parallelism across nodes | Async node training |
| `.mat` input only | Limits usable datasets | ENVI, GeoTIFF, HDF5 loaders |
| No resumable training | Power/connection loss loses all progress | Per-node checkpoint saves |

### Near-Term Roadmap

**Masked Spectral Modelling pretraining**
Train the SSM encoder to reconstruct randomly masked bands rather than classify pixels. This produces a universal spectral encoder that requires zero labeled pixels — self-supervised learning from the hyperspectral data itself.

**Patch-based training**
Slice large scenes into 64×128 patches for training. This removes the VRAM constraint on large datasets entirely and enables data augmentation through random patch sampling.

**Transfer learning across scenes**
Sequential fine-tuning across temporal observations of the same scene (e.g., Hayabusa2 Ryugu images across mission phases). Band-agnostic encoder and decoder layers transfer fully; only the spectral input layer reinitialises.

**Multi-sensor generalisation**
Replace band-index positional encoding with wavelength-value encoding. This would make GHOST sensor-agnostic — a model trained on AVIRIS could generalise to HyMap without retraining, because both sensors share the same physical absorption features even at different band resolutions.

### Long-Term Vision

GHOST aims to be the **nnUNet of hyperspectral imaging** — a framework that any domain scientist can install, point at their data, and receive competitive segmentation results without any machine learning expertise. The same way nnUNet democratised medical image segmentation, GHOST should democratise hyperspectral analysis across Earth observation, planetary science, medical imaging, and materials science.

---

*Document maintained by the GHOST development team. Last updated for v0.1 hackathon release.*