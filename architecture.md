# GHOST Architecture

A technical overview of the pipeline components. This is not exhaustive — the code is the authoritative reference.

---

## Pipeline

```
RAW .mat FILE  (H, W, C)
      │
      ▼
Continuum Removal       strips brightness, isolates absorption shape
      │
      ▼
Spectral 3D Conv Stack  local spectral + spatial features, kernel (7,3,3)
      │
      ▼
SE Block                learned per-channel attention weights
      │
      ▼
2D U-Net                multi-scale spatial context, skip connections
      │
      ▼
RSSP Tree               per-node specialist ensembles (see below)
```

All dimensions — band count, class count, spatial size — are read from the file at runtime. Nothing is hardcoded.

---

## Components

### Continuum Removal

Divides each pixel's spectrum by its convex hull envelope:

```
CR(λ) = spectrum(λ) / continuum(λ)
```

This normalises brightness variation across sensors and illumination conditions, leaving only absorption feature shape. It replaces PCA as the dimensionality-agnostic preprocessing step. No components to choose, no information discarded.

### Spectral 3D Convolution Stack

3D convolutions with kernel `(7, 3, 3)` — 7 bands deep, 3×3 spatially. Models spectral band adjacency explicitly rather than treating every band as independent. Runs `num_blocks` sequential layers; output is `num_filters × C` channels.

### Squeeze-and-Excitation Block

Global average pools each channel to a scalar, passes through a small MLP, and multiplies each channel by its learned weight. Selects which spectral features are relevant, suppresses the rest.

### 2D U-Net

Four downsampling stages via MaxPool2d, symmetric decoder with ConvTranspose2d. Skip connections wire encoder feature maps directly to corresponding decoder levels, preserving spatial detail that cannot survive the bottleneck. Handles non-power-of-two dimensions via bilinear interpolation before skip concatenation.

Each ConvBlock:
```
Conv2d(3×3) → BN → ReLU → Dropout2d(0.3) → Conv2d(3×3) → BN → ReLU
```

### RSSP — Recursive Spectral Splitting with Parallel Forests

Builds a binary class hierarchy from the dataset's own spectral statistics. At each node, the two most spectrally distant classes (by SAM distance on continuum-removed mean spectra) become split seeds. Remaining classes are assigned to whichever seed they're closer to. A pixel balance correction prevents one branch from dominating.

Each node trains an independent ensemble of `num_forests` HyperspectralNet models. Global class IDs are remapped to local IDs per node. Epoch budget scales with node complexity.

Tree construction stops when: fewer than 3 classes remain, any class has fewer than 10 pixels, depth ≥ 3, or mean intra-node SAM < 0.05.

At inference, softmax probabilities are averaged across forest members and propagated down the tree with soft routing weights.

### SSSR — Selective Spectral State Routing *(experimental)*

Replaces hard argmax routing at each tree node with probabilistic soft routing. A frozen spectral encoder produces per-pixel fingerprints; a lightweight routing head at each node outputs `p_left ∈ (0,1)`. Pixels flow down both branches weighted by confidence. In hybrid mode, forest routing is the base and SSSR corrects proportionally to its own confidence. Inactive by default.

---

## Training

- **Loss:** CrossEntropyLoss with `ignore_index=0` (background masked)
- **Optimiser:** AdamW, `weight_decay=1e-4`
- **Scheduler:** ReduceLROnPlateau, `patience=10, factor=0.5`
- **Checkpoint:** best validation mIoU per node
- **Splits:** stratified per class — every class appears in every split

---

## Known Limitations

- Local spectral context only (7-band kernel). Long-range spectral dependencies are not modelled.
- No spatial regularisation — predictions are per-pixel; homogeneous regions can show speckle noise.
- Class imbalance is handled structurally by RSSP but not at the loss level. On imbalanced datasets, dominant classes can suppress minority predictions.
- Full-image training paradigm is data-efficient but incompatible with transformer/Mamba components that require patch-based sample diversity to train effectively.