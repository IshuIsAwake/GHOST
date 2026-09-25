# GHOST — Architecture

This is how the pipeline works. Nothing here is particularly novel — it's mostly standard components wired together in a way that seemed reasonable for hyperspectral data.

GHOST ships two architectures. **0.2.0** (the default, next section) classifies each pixel from its spectrum
alone. **0.1.7** (the rest of this document) uses a 3-D conv + U-Net with the Spectral Partition Tree.

---

## Architecture 0.2.0 — per-pixel

| Stage | What it does |
|-------|--------------|
| Loader | `.mat` (incl. v7.3), ENVI, TIFF/GeoTIFF, HDF5 → `(H, W, Bands)`. Pixels with a non-finite, all-zero or no-data spectrum are skipped |
| Continuum removal | Once per scene, on raw spectra. **full** (≥64 bands): Savitzky-Golay smoothing (window ≈5% of bands), Andrew's monotone-chain upper hull of the smoothed spectrum, then raw ÷ max(hull, raw), so output lies in (0, 1]. **simple** (3–63): raw ÷ line from first to last band, then ÷ pixel max. **off** (<3): ÷ pixel max. The hull uses real wavelengths when the file provides them |
| Encoder | 1-D dilated ResNet on `(pixels, 1, bands)`: stem conv (kernel 7, 64 channels), then residual blocks with dilation 1, 2, 4, … Width and length never change. A block is added only while the cumulative receptive field fits in the band count: 4 blocks at 200 bands (reach 187), 3 at 103, 2 at 61, 1 at 32 |
| Pooling | Average over bands (default) or flatten, which keeps band positions |
| Head | Linear → BatchNorm → ReLU → Dropout 0.3 → Linear |
| Training | AdamW (1e-3), batches of 256 pixels, early stop on validation mIoU |

Every operator is per pixel, so nothing learns scene layout. Continuum removal also divides out each pixel's
brightness; `--cr none` (v0.1's scene z-score) keeps brightness, for the ablation that tests whether that
matters.

---

## Pipeline overview (0.1.7)

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

Dividing by the continuum removes brightness variation and leaves the shape of absorption features, on any
band count and without configuration.

In 0.1.7, though, the continuum is not a hull. It is a straight line from each spectrum's minimum to its maximum
across the bands, which the code itself calls a simplification. It also runs after the dataset z-scores the
whole scene, so spectra take both signs and the line crosses zero. On Indian Pines every pixel's does: the
model's input reaches roughly ±800,000, and 81% of pixels have a value beyond ±100. Architecture 0.2.0 uses
a real upper hull on the raw spectra instead.

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

1. Compute each class's mean spectrum from the z-scored scene, then divide it by the same min-to-max line as
   in section 1
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

**Known problems (0.1.7, measured on Indian Pines):**
- **The tree sees test labels.** `ghost train_spt` builds it from every labelled pixel, test pixels
  included (`train_rssp.py:191` passes the full ground truth), so class means and pixel counts come partly
  from the test set. `benchmarks/v01_split.py --tree-from-train` builds it from training pixels only.
- **Its angles are not spectral.** The z-scored class means take both signs, so step 1's division blows
  up, to values from −1,795 to 794. 86 of the 120 class pairs then sit above π/2, up to 2.09 rad, an angle
  two non-negative spectra cannot make.
- **Small classes stop it at the root.** Built from training pixels alone on the 20% split, the tree is a
  single node: Oats has 4 training pixels, below the 10-pixel stop.

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
| Optimiser | AdamW | `lr=1e-4`, `weight_decay=1e-4` |
| Epoch | one step | The whole scene is one sample (batch size 1), so each epoch is a single optimiser step |
| Scheduler | ReduceLROnPlateau | `patience=10, factor=0.5`, stepped on the training loss |
| Early stopping | `train_spt` only | Up to 400 epochs; stops after 50 without improvement, never before epoch 40, validating every 20. Flat `ghost train --arch 0.1.7` has none: it runs all 300 epochs, validating every 10 |
| Checkpoint | best val mIoU | Flat: one model. SPT: per node, per ensemble member |
| Splits | stratified | Every class appears in every split |
