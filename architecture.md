# GHOST Architecture

---

## Pipeline Overview

```
                         Input: .mat file (H, W, Bands)
                                    |
                    ┌───────────────┴───────────────┐
                    |                               |
             Continuum Removal              Raw spectra saved
             (physics-based norm)           for SPT tree building
                    |                               |
                    v                               v
            Spectral 3D Conv Stack          SAM distance matrix
            kernel (7, 3, 3)                between class means
            num_blocks layers                       |
                    |                               v
                    v                        Spectral Partition Tree
              SE Attention                  (recursive spectral
              (channel weighting)            class splitting)
                    |                               |
                    v                               |
              2D U-Net Encoder/Decoder              |
              (4-level, skip connections)            |
                    |                               |
                    └───────────┬───────────────────┘
                                |
                    Per-node model ensemble
                    (each tree node trains N independent models)
                                |
                                v
                    Soft cascade inference
                    (probabilities averaged across ensemble members,
                     propagated down tree branches)
                                |
                                v
                     Prediction Map (H, W)
```

---

## Components

### 1. Continuum Removal

Replaces PCA as the dimensionality-agnostic preprocessing step.

```
CR(lambda) = spectrum(lambda) / continuum(lambda)
```

The continuum is the convex hull envelope of the reflectance spectrum. Dividing by it removes brightness variation across sensors and illumination conditions, leaving only **absorption feature shape**.

- No components to choose
- No information discarded
- Works on any band count without configuration

**Source:** `ghost/preprocessing/continuum_removal.py`

### 2. Spectral 3D Convolution Stack

3D convolutions with kernel `(7, 3, 3)` — 7 bands spectral depth, 3x3 spatial.

- Models spectral band adjacency explicitly (nearby bands are correlated)
- `num_blocks` sequential layers (default: 3)
- Output channels: `num_filters x C`
- Handles arbitrary band counts at runtime

**Source:** `ghost/models/spectral_3d_block.py`

### 3. Squeeze-and-Excitation (SE) Block

Channel attention mechanism:

1. Global average pool each channel to a scalar
2. Pass through a 2-layer MLP (reduce → expand)
3. Sigmoid activation → per-channel weight
4. Multiply each channel by its learned weight

Selects which spectral features matter, suppresses noise channels.

**Source:** `ghost/models/se_block.py`

### 4. 2D U-Net

Standard encoder-decoder with skip connections:

- **Encoder:** 4 downsampling stages via MaxPool2d
- **Decoder:** symmetric upsampling via ConvTranspose2d
- **Skip connections:** encoder features concatenated to decoder at each level
- **ConvBlock:** `Conv2d(3x3) -> BN -> ReLU -> Dropout(0.3) -> Conv2d(3x3) -> BN -> ReLU`
- Channel progression: `f -> 2f -> 4f -> 8f -> 16f` (where `f = base_filters`)

Handles non-power-of-two spatial dimensions via bilinear interpolation before skip concatenation.

**Source:** `ghost/models/encoder_2d.py`, `ghost/models/decoder_2d.py`

### 5. SPT — Spectral Partition Tree

The core of GHOST's approach to multi-class segmentation.

**Tree Construction:**

1. Compute mean spectrum per class (after continuum removal)
2. Build pairwise SAM (Spectral Angle Mapper) distance matrix
3. Find the two most spectrally distant classes → split seeds
4. Assign remaining classes to whichever seed they're closer to
5. Apply pixel balance correction (prevents one branch from dominating)
6. Recurse on each branch

**Stopping conditions:**
- Fewer than 3 classes in a node
- Any class has fewer than 10 pixels
- Depth >= 3
- Mean intra-node SAM < 0.05 (classes too similar to split further)

**Training:**

Each tree node trains an independent ensemble of `num_ensembles` HyperspectralNet models:
- Global class IDs are remapped to local IDs per node
- Epoch budget scales with node complexity: `max(epochs//2, epochs x node_classes/total_classes)`
- Each ensemble member uses a different random seed
- Best model (by validation mIoU) is checkpointed per ensemble member

**Inference:**

Soft cascade — softmax probabilities are averaged across ensemble members, then propagated down the tree with routing weights. Final prediction is argmax of accumulated global-class probabilities.

**Source:** `ghost/rssp/sam_clustering.py`, `ghost/rssp/rssp_trainer.py`, `ghost/rssp/rssp_inference.py`

### 6. SSSR Router (Experimental)

Selective Spectral State Routing. Replaces hard argmax at each tree node with probabilistic soft routing.

- A frozen spectral encoder produces per-pixel fingerprints
- A lightweight routing head at each node outputs `p_left in (0, 1)`
- In hybrid mode: ensemble routing is base, SSSR corrects proportionally to confidence

**Status:** Experimental. Ensemble routing outperforms hybrid/soft in all tested configurations. The SSM encoder will be reworked in a future version.

**Source:** `ghost/rssp/sssr_router.py`, `ghost/models/spectral_ssm.py`

---

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Loss | CrossEntropy | Options: `ce`, `dice`, `focal`, `squared_ce` |
| Optimiser | AdamW | `weight_decay=1e-4` |
| Scheduler | ReduceLROnPlateau | `patience=10, factor=0.5` |
| Early stopping | patience=50 | No improvement for N epochs → stop |
| Checkpoint | best val mIoU | Per node, per ensemble member |
| Splits | stratified | Every class appears in every split |
