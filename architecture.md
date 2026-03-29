# GHOST — Architecture

This is how the pipeline works. Nothing here is particularly novel — it's mostly standard components wired together in a way that seemed reasonable for hyperspectral data.

---

## Pipeline overview

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

### 1. Continuum removal

Instead of PCA (which requires choosing how many components to keep), I use continuum removal as the preprocessing step.

```
CR(lambda) = spectrum(lambda) / continuum(lambda)
```

The continuum is the convex hull envelope of the reflectance spectrum. Dividing by it removes brightness variation and leaves absorption feature shape. The nice thing is it works on any band count without configuration and doesn't throw away information.

**Source:** `ghost/preprocessing/continuum_removal.py`

### 2. Spectral 3D convolution stack

3D convolutions with kernel `(7, 3, 3)` — 7 bands spectral depth, 3x3 spatial.

- Models spectral band adjacency (nearby bands tend to be correlated)
- `num_blocks` sequential layers (default: 3)
- Output channels: `num_filters x C`
- Handles arbitrary band counts at runtime

**Source:** `ghost/models/spectral_3d_block.py`

### 3. Squeeze-and-Excitation (SE) block

Standard channel attention:

1. Global average pool each channel to a scalar
2. Pass through a 2-layer MLP (reduce → expand)
3. Sigmoid → per-channel weight
4. Multiply each channel by its weight

The idea is to let the network learn which spectral features matter for a given input.

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

This is the part I find most interesting. Instead of training one model on all classes at once, the SPT splits classes into groups based on spectral similarity and trains separate ensembles for each group.

**Tree construction:**

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

Each tree node trains an independent ensemble of `num_ensembles` models:
- Global class IDs are remapped to local IDs per node
- Epoch budget scales with node complexity: `max(epochs//2, epochs x node_classes/total_classes)`
- Each ensemble member uses a different random seed
- Best model (by validation mIoU) is checkpointed per ensemble member

**Inference:**

Soft cascade — softmax probabilities are averaged across ensemble members, then propagated down the tree with routing weights. Final prediction is argmax of accumulated global-class probabilities.

**Source:** `ghost/rssp/sam_clustering.py`, `ghost/rssp/rssp_trainer.py`, `ghost/rssp/rssp_inference.py`

### 6. SSSR Router (experimental, not recommended)

An attempt at replacing hard argmax at each tree node with probabilistic soft routing using a spectral state-space model. It doesn't work well — ensemble routing beats it in every configuration I've tested. I'll either rework or remove it.

**Source:** `ghost/rssp/sssr_router.py`, `ghost/models/spectral_ssm.py`

---

## Training configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Loss | CrossEntropy | Options: `ce`, `dice`, `focal`, `squared_ce` |
| Optimiser | AdamW | `weight_decay=1e-4` |
| Scheduler | ReduceLROnPlateau | `patience=10, factor=0.5` |
| Early stopping | patience=50 | No improvement for N epochs → stop |
| Checkpoint | best val mIoU | Per node, per ensemble member |
| Splits | stratified | Every class appears in every split |
