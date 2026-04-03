# GHOST v2 — Spectral-First Architecture

*Complete design specification for implementation.*

---

## Why the overhaul

GHOST v0.1.x uses a 3D Conv (7,3,3) + SE + U-Net backbone. The U-Net dominates learning — spatial features (shapes, regions, neighbor context) minimize loss faster than subtle spectral dips, so the network learns to classify by location rather than by chemistry. This produces strong numbers on single-scene benchmarks but the model has not learned what materials actually are. It cannot generalize to unseen scenes, different sensors, or different domains. The fix: remove all spatial convolutions from the classification loop and force the model to classify from spectral evidence alone.

---

## Design Principles

1. **One architecture fits all.** Same code, same hyperparameters, retrain on new data with zero config changes. Indian Pines, LUSC histopathology, Mars CRISM — all use the same pipeline. You retrain, but you change nothing.
2. **Spectral-only classification.** Every pixel is classified independently from its spectrum alone. No neighboring pixels, no 2D convolutions, no skip connections across spatial dimensions. If the model cannot identify a material from its spectrum, it has not learned the material.
3. **Band-count agnostic.** The architecture auto-adapts at construction time to any band count — from 3 bands to 400+. No hardcoded dimensions, no manual configuration.
4. **Format agnostic.** Accepts `.mat`, ENVI (`.hdr`), GeoTIFF (`.tif`), and HDF5 (`.h5`). Auto-detects format by extension, normalizes to `(H, W, Bands)` internally. The pipeline never knows what format the file was.
5. **Chemistry is the invariant (within a sensor).** Spectral fingerprints are determined by molecular bonds — chlorophyll absorbs at the same wavelengths whether the corn is in Indiana or Brazil. The model learns absorption chemistry, which transfers across scenes from the same sensor with similar atmospheric/calibration conditions. Cross-sensor transfer is a longer-term goal that requires handling sensor response function differences (deferred to post-SSL). Spatial patterns never transfer.
6. **Multi-scene native.** Supports both single-scene and multi-scene training from the ground up. Train on 2 scenes, run inference on 50. The headline experiment.
7. **Tunability.** Every component has sensible defaults that work out of the box. Power users can override any setting via CLI flags.

---

## Pipeline Overview

```
Input: any supported HSI file(s)
    |
    v
Universal Data Loader
    |      Auto-detects format (.mat, .hdr, .tif, .h5)
    |      Normalizes to (H, W, Bands) + (H, W) ground truth
    |      Extracts wavelength metadata when available (optional)
    |
    v
Stage 1: Continuum Removal (CR)
    |      Per-pixel, deterministic, zero parameters
    |      Three modes: full (convex hull), simple (slope removal), off
    |      Auto-selects by band count: >=64 full, <64 simple, <3 off
    |      Override: --cr full|simple|off
    |      Output: (H, W, Bands), values in (0, 1]
    |
    v
Stage 2: 1D Dilated ResNet Encoder
    |      Per-pixel, learned
    |      Processes full spectrum, outputs fixed-size embedding
    |      Independent encoder per SPT node, per ensemble member
    |      Auto-adapts depth and kernel to band count
    |
    v
Stage 3: MLP Classification Head
    |      Per SPT node, per ensemble member
    |      embed_dim -> hidden -> n_local_classes
    |
    v
Stage 4: SPT Ensemble Inference
    |      Unchanged from current GHOST
    |      Ensemble routing, probability accumulation down tree
    |
    v
Stage 5: Dense CRF Post-Processing (optional)
    |      Spatial smoothing on output probabilities only
    |      The model's classification is 100% spectral;
    |      CRF only cleans noise after the fact
    |
    v
Output: Prediction Map (H, W)
```

---

## Universal Data Loader

**Purpose:** Accept any common HSI format, produce a consistent `(H, W, Bands)` array.

### Supported formats

| Extension | Library | Raw shape | Transpose needed | Wavelength metadata |
|-----------|---------|-----------|------------------|---------------------|
| `.mat` | `scipy.io.loadmat` / `h5py` (v7.3+) | Variable | Maybe | Unreliable |
| `.hdr` + `.raw`/`.bsq`/`.bil`/`.bip` | `spectral` | (H, W, Bands) | No | Yes (in .hdr) |
| `.tif` / `.tiff` | `rasterio` | **(Bands, H, W)** | Yes | Rare |
| `.h5` / `.hdf5` | `h5py` | Variable | Maybe | Unreliable |

### Auto-detection logic

1. Detect format by file extension
2. Open file with the appropriate library
3. For `.mat` and `.h5` (where array names and shapes are unpredictable):
   - List all arrays in the file
   - The largest 3D array = data cube
   - The 2D array with matching H,W = ground truth
   - If shape is `(Bands, H, W)` (smallest dim first), transpose to `(H, W, Bands)`
4. For ENVI: `spectral` library returns `(H, W, Bands)` directly
5. For GeoTIFF: `rasterio` returns `(Bands, H, W)`, always transpose
6. Extract wavelength metadata when available (stored as optional attribute, not required by pipeline)

### Dependencies

Heavy libraries are optional extras to keep the base install light:

```bash
pip install ghost-hsi              # .mat and .h5 support (scipy, h5py)
pip install ghost-hsi[envi]        # + ENVI support (spectral)
pip install ghost-hsi[geo]         # + GeoTIFF support (rasterio/GDAL)
pip install ghost-hsi[all]         # everything
```

### Implementation

New module: `ghost_new/datasets/loader.py`. Contains a single `load_hsi(data_path, gt_path)` function that returns `(data_array, gt_array, metadata_dict)`. The dataset class calls this and doesn't care how the file was loaded. Current `.mat`-specific code in `hyperspectral_dataset.py` is replaced.

---

## Stage 1 — Continuum Removal (CR)

**Input:** Raw reflectance spectrum per pixel, shape `(Bands,)`

**Output:** Normalized spectrum, shape `(Bands,)`, values in `(0, 1]`

### What it does

Every raw spectrum sits on a sloped baseline caused by illumination angle, atmospheric scattering, and sensor response. This slope is not a material property. Continuum Removal strips it out.

### Three CR modes

GHOST provides three CR modes, selectable via `--cr auto|full|simple|off`:

**1. Full (convex hull CR)** — builds the upper convex hull envelope of the spectrum (a ceiling touching every local reflectance peak) and divides the raw spectrum by this ceiling at every band:

```
CR(lambda) = spectrum(lambda) / hull(lambda)
```

Every local peak normalizes to 1.0. Every absorption feature becomes a fractional dip below 1.0. Dip position, depth, and width are determined entirely by molecular bond chemistry. This is the most powerful mode — it makes absorption feature depths directly comparable across any material, sensor, or illumination condition.

**2. Simple (slope removal)** — draws a straight line from the first band to the last band and divides. This is what GHOST v0.1.x uses. It removes overall spectral tilt but does not normalize absorption features relative to their local peak context. A deep dip next to a tall peak and the same dip next to a short peak appear to have different depths even though the chemistry is identical. Still useful — removes illumination slope, which is the dominant non-chemical variation.

**3. Off** — no CR. Raw spectra are normalized to [0, 1] by dividing by the per-pixel max. The encoder receives reflectance values directly. Useful as a baseline to measure CR's contribution, or for data where CR is inappropriate.

### Auto mode (default)

`--cr auto` selects the mode based on band count:

```
bands >= 64:   full   (enough resolution for hull to be meaningful)
bands < 64:    simple (hull is too coarse, slope removal still helps)
bands < 3:     off    (degenerate, just normalize by max)
```

**The thresholds (64, 3) are provisional.** They will be validated empirically on Indian Pines (200 bands), OHID-1 (32 bands), and synthetic low-band cases. If full CR works well at 32 bands, the threshold drops. The auto logic is a sensible default, not a hard rule — users can override with `--cr full` or `--cr simple` for any dataset.

### What the output represents (full CR mode)

After CR, the spectrum is a direct chemical fingerprint:
- Values at 1.0 = no absorption at that wavelength
- Values below 1.0 = material absorbs light at that wavelength
- Dip position = which molecular bonds are present
- Dip depth = concentration / abundance
- Dip width = type of transition (narrow = vibrational/bond-specific, broad = electronic)

### Full CR implementation

**Upper hull algorithm:** Use Andrew's monotone chain (upper hull only), not `scipy.spatial.ConvexHull`. O(n) on pre-sorted points (band indices are already sorted), handles all degenerate inputs natively (collinear, 2 points), no scipy dependency, no lower hull computation wasted.

```python
def upper_hull(points):
    """Andrew's monotone chain - upper hull only.
    points: list of (band_index, reflectance), sorted by band_index.
    Returns list of (band_index, reflectance) hull vertices."""
    hull = []
    for p in points:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) >= 0:
            hull.pop()
        hull.append(p)
    return hull

def cross(O, A, B):
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])
```

**Noise robustness via Savitzky-Golay smoothing:** Noise spikes become hull vertices, creating false absorption features. To prevent this, smooth the spectrum before hull computation, but divide the RAW spectrum by the hull. This preserves real narrow features in the output while preventing noise from distorting the hull shape.

```
Raw spectrum → smooth → compute hull on smoothed → interpolate hull → divide RAW by hull
```

NOT: `smooth → hull → divide smoothed by hull` (destroys narrow features).

**Smoothing window auto-scaling:**

```python
window = max(5, int(0.05 * band_count))  # ~5% of band count, minimum 5
window = min(window, band_count)          # can't exceed spectrum length
if window % 2 == 0:
    window -= 1                           # must be odd for savgol
if window < 3:
    window = 3                            # savgol minimum
# polyorder = 2 always (preserves peak shapes)
```

| Bands | Window | Window as % of spectrum |
|-------|--------|------------------------|
| 200   | 11     | 5.5%                   |
| 103   | 7      | 6.8%                   |
| 64    | 5      | 7.8%                   |

**Known tradeoff:** Smoothing can attenuate genuinely narrow absorption features (3-4 bands wide) in the hull computation, making them appear ~10-20% shallower in the CR output. This is acceptable — a slightly weakened real feature is far less damaging than a completely fabricated feature from noise. Features this narrow are at the resolution limit of most HSI sensors. The encoder sees thousands of spectra during training and learns to work with slightly attenuated features. The alternative (no smoothing) creates entirely fabricated features at wrong positions, which teaches the model something false.

**Complete per-pixel algorithm:**

```python
def continuum_removal_full(spectrum):
    """Full convex hull CR for a single pixel.
    Input:  spectrum shape (Bands,), raw reflectance
    Output: cr_spectrum shape (Bands,), values in (0, 1]
    """
    bands = len(spectrum)

    # 1. Floor clip — prevent division by zero from negative/zero reflectance
    spectrum = np.maximum(spectrum, 1e-6)

    # 2. Savitzky-Golay smoothing for hull computation only
    window = max(5, int(0.05 * bands))
    window = min(window, bands)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        window = 3
    smoothed = savgol_filter(spectrum, window_length=window, polyorder=2)

    # 3. Upper hull on smoothed spectrum
    points = [(i, smoothed[i]) for i in range(bands)]
    hull_vertices = upper_hull(points)

    # 4. Interpolate hull to all band positions
    hull_x = [v[0] for v in hull_vertices]
    hull_y = [v[1] for v in hull_vertices]
    hull_at_all_bands = np.interp(np.arange(bands), hull_x, hull_y)

    # 5. Divide RAW spectrum by hull (not smoothed)
    hull_at_all_bands = np.maximum(hull_at_all_bands, 1e-6)  # safety floor
    cr = spectrum / hull_at_all_bands

    # 6. Warn + clip
    if np.any(cr > 1.0 + 1e-4):
        warnings.warn(f"CR produced values > 1.0 (max={cr.max():.4f}), possible hull error")
    cr = np.clip(cr, 0.0, 1.0)

    return cr
```

### Simple CR implementation

```python
def continuum_removal_simple(spectrum):
    """Slope removal CR — straight line from first to last band.
    Input:  spectrum shape (Bands,), raw reflectance
    Output: cr_spectrum shape (Bands,), values in (0, 1]
    """
    bands = len(spectrum)
    spectrum = np.maximum(spectrum, 1e-6)

    # Straight line from first to last reflectance value
    line = np.linspace(spectrum[0], spectrum[-1], bands)
    line = np.maximum(line, 1e-6)

    cr = spectrum / line
    cr = np.clip(cr, 0.0, 1.0)
    return cr
```

### Off mode implementation

```python
def continuum_removal_off(spectrum):
    """No CR — normalize to [0, 1] by dividing by max."""
    spectrum = np.maximum(spectrum, 1e-6)
    return spectrum / spectrum.max()
```

### Dispatcher

```python
def apply_cr(spectrum, mode='auto'):
    """Apply continuum removal to a single pixel.
    mode: 'auto', 'full', 'simple', 'off'
    """
    bands = len(spectrum)
    if mode == 'auto':
        if bands < 3:
            mode = 'off'
        elif bands < 64:
            mode = 'simple'
        else:
            mode = 'full'

    if mode == 'full':
        return continuum_removal_full(spectrum)
    elif mode == 'simple':
        return continuum_removal_simple(spectrum)
    else:
        return continuum_removal_off(spectrum)
```

**Vectorized wrapper:** The per-pixel functions above are for clarity. The actual implementation should vectorize over the spatial dimensions for performance: reshape `(H, W, Bands)` to `(H*W, Bands)`, process all pixels, reshape back. The hull computation is inherently per-pixel (each pixel has a different hull), but the floor clipping, smoothing, interpolation, and division can be partially vectorized. For full CR, the main loop is over pixels — use joblib or a simple for-loop; the per-pixel cost is O(Bands) so even 100K pixels at 200 bands takes seconds.

### Edge cases handled

**Noise spikes:** Smoothing before hull computation prevents noise from becoming hull vertices. The raw/hull division preserves real narrow features. (See "Known tradeoff" above.)

**Short spectra (< 64 bands):** Auto mode falls back to simple CR. The hull has too few vertices at low band counts for meaningful local peak normalization. Simple CR still removes tilt, which is the dominant non-chemical variation.

**Flat spectra (no absorption):** Hull equals spectrum. CR output is all ~1.0. This is correct — "no absorption features" is valid spectral information. The encoder receives a near-flat vector and classifies accordingly.

**Monotonic spectra:** Hull is a straight line. CR removes the slope and output is flat (~1.0 everywhere). Equivalent to simple CR. Any deviations from the monotonic trend are preserved as absorption features.

**Negative/zero reflectance:** Floor-clipped to 1e-6 before any computation. Negative reflectance is physically meaningless (calibration artifact). The floor is negligible relative to real values (0.01-1.0).

**Mixed pixels (material boundaries):** A pixel's spectrum is a linear mixture of two materials. The hull is valid, CR output shows absorption features from both materials superimposed. Not a bug — the pixel genuinely contains both materials. The encoder classifies as the dominant material or reports low confidence. Both correct.

**Collinear points / degenerate hulls:** Andrew's monotone chain handles these natively. Collinear points produce a 2-vertex hull (straight line), which is correct for a linear spectrum.

---

## Stage 2 — 1D Dilated ResNet Encoder

**Input:** CR-normalized spectrum, shape `(batch, 1, bands)` — single-channel 1D signal

**Output:** Spectral embedding, shape `(batch, embed_dim)`

This is the core learned component. Everything upstream is deterministic preprocessing. Everything downstream is classification logic. The encoder's job: compress a high-dimensional spectral fingerprint into a dense embedding that captures material identity.

### What the encoder learns

The dilated architecture creates a hierarchy of feature detection:

**Early layers (small dilation, small receptive field):** Learn to detect edges and slopes — "is this region flat or changing?", "how steep is the entry into a dip?", "is this a local minimum?" These are spectral first and second derivatives. A filter like [+1, +1, 0, -1, -1] is a slope detector. [+1, -2, +1] is a curvature detector that fires at dip bottoms.

**Middle layers (medium dilation):** See complete absorption features — the full shape of a single dip including entry slope, minimum, exit slope, and surrounding continuum. This is where chemically meaningful features emerge: a specific dip shape at a specific spectral position corresponds to a specific molecular bond.

**Deep layers (large dilation, full-spectrum receptive field):** See the combination of all absorption features simultaneously. Material identification happens here — vegetation = chlorophyll dip AND NIR plateau AND water absorption. It's the conjunction of multiple features that identifies a material, not any single one. Absence of a feature is also information.

**After Global Average Pool:** Each channel collapses to a scalar — "how strongly did this feature pattern appear anywhere in the spectrum." The resulting vector encodes which absorption patterns are present and how strongly. This is the spectral fingerprint embedding.

### Architecture

```
STEM
  Conv1d(1 -> C, kernel=K, padding=K//2, dilation=1)
  BatchNorm1d(C)
  ReLU

RESBLOCK 0 (dilation=1)
  Conv1d(C -> C, kernel=K, dilation=1, padding=1*K//2) -> BN -> ReLU
  Conv1d(C -> C, kernel=K, dilation=1, padding=1*K//2) -> BN
  + skip (identity)
  ReLU

RESBLOCK 1 (dilation=2)
  Conv1d(C -> C, kernel=K, dilation=2, padding=2*K//2) -> BN -> ReLU
  Conv1d(C -> C, kernel=K, dilation=2, padding=2*K//2) -> BN
  + skip (identity)
  ReLU

RESBLOCK 2 (dilation=4)
  ...same pattern...

  ...(blocks auto-pruned based on band count)...

RESBLOCK N (dilation=2^N)
  ...same pattern...

POOLING
  Global Average Pool across spectral axis -> (batch, C)

PROJECTION
  Linear(C -> embed_dim) -> (batch, embed_dim)
```

### Key design decisions

**Constant channels throughout.** All blocks use `C = encoder_channels`. No channel expansion between blocks. Standard image ResNets expand channels (64->128->256->512) because they simultaneously downsample spatially — more channels compensate for less spatial resolution. This encoder does NO downsampling; the spectral axis keeps full length through every block. Without downsampling, channel expansion just means more parameters for no structural reason. The increasing dilation already provides the "expansion" — deeper blocks see wider context with the same capacity. Constant channels also mean every skip connection is a clean identity addition (no 1x1 projection convs needed).

**Dilation schedule: powers of 2.** `dilations[i] = 2^i` for block `i`. Each dilation level doubles the receptive field. The full schedule for depth=5 is `[1, 2, 4, 8, 16]`.

**What each block sees** (kernel=7):

| Block | Dilation | Effective RF (single conv) | Cumulative RF (after block) |
|-------|----------|---------------------------|----------------------------|
| Stem  | 1        | 7                         | 7                          |
| 0     | 1        | 7                         | 19                         |
| 1     | 2        | 13                        | 43                         |
| 2     | 4        | 25                        | 91                         |
| 3     | 8        | 49                        | 187                        |
| 4     | 16       | 97                        | 379                        |
| 5     | 32       | 193                       | 763                        |

Default 5 blocks covers up to ~370 bands. A 6th block extends to ~760 bands for very high-resolution sensors. Users with 400+ band data set `--encoder_depth 6`.

**BatchNorm, not LayerNorm.** We're batching pixels — even a small SPT leaf with 200 training pixels gives full batches of 64. Batch statistics are stable. BN is the most tested normalization for 1D CNNs, provides slight regularization, and is fast. (LayerNorm would be appropriate if a Transformer backbone is added later.)

**Global Average Pool, not Attention Pooling.** GAP has zero learnable parameters — it cannot overfit, which matters for small SPT leaf nodes with limited training data. After ReLU, most feature map values are exactly 0. A well-trained "chlorophyll detector" channel has strong activation at ~10 positions and 0.0 everywhere else. GAP preserves the ratio between strong and weak features. The classification MLP can learn to rescale features as needed. (Attention Pooling is a valid future upgrade if narrow features prove to be under-represented in the embedding.)

**Padding preserves sequence length.** For dilated conv: `padding = dilation * (kernel_size // 2)`. Output length always equals input length. Required for identity skip connections.

### Band-count auto-adaptation

At model construction time, the encoder auto-configures based on the input band count. No user intervention required.

**The rule:** No convolutional layer's effective receptive field may exceed the input length. Effective RF = `(kernel_size - 1) * dilation + 1`.

**Auto-pruning algorithm:**
1. Start with full dilation schedule `[1, 2, 4, ..., 2^(depth-1)]` and configured kernel size
2. For each dilation level: if `(kernel - 1) * dilation + 1 > band_count`, drop that level and all higher
3. If `kernel_size > band_count`: shrink kernel to `band_count` (or `band_count - 1` if even, to keep it odd)
4. Build the network with whatever survived

**Behavior at different band counts:**

| Bands | Kernel | Blocks | Dilations       | Cumulative RF | Notes                                    |
|-------|--------|--------|-----------------|---------------|------------------------------------------|
| 3     | 3      | 1      | [1]             | 9             | Effectively a shallow nonlinear transform |
| 10    | 7      | 1      | [1]             | 19            | Single-scale feature detection            |
| 30    | 7      | 2      | [1, 2]          | 43            | Two-scale features                        |
| 61    | 7      | 4      | [1, 2, 4, 8]   | 187           | Covers full LUSC spectrum                 |
| 103   | 7      | 5      | [1, 2, 4, 8, 16] | 379         | Covers Pavia with room to spare           |
| 200   | 7      | 5      | [1, 2, 4, 8, 16] | 379         | Covers Indian Pines / Salinas             |
| 400+  | 7      | 6      | [1, 2, ..., 32] | 763          | User sets --encoder_depth 6               |

The architecture degrades gracefully at low band counts rather than crashing. A 3-band input gets a minimal encoder — which is all you can do with 3 bands.

### Parameter count

Typical config (C=64, K=7, 5 blocks, embed_dim=128):

- Stem: 1 * 64 * 7 + 64 (BN) = ~512
- Each ResBlock: 2 * (64 * 64 * 7) + 2 * 64 (BN) = ~57,500
- 5 blocks: ~287,500
- Projection: 64 * 128 = 8,192
- **Per encoder: ~296K parameters**

**System-level total:** With a typical SPT tree (depth 3, ~7 nodes) and 5 ensemble members per node, the full system trains 35 independent encoders + heads = ~10.4M parameters total. This is comparable to the current GHOST U-Net system in aggregate, but distributed across specialized models rather than concentrated in one monolithic backbone. Each individual encoder is lightweight enough to train quickly on small per-node datasets.

---

## Stage 3 — MLP Classification Head

**Input:** Spectral embedding from encoder, shape `(batch, embed_dim)`

**Output:** Logits over local class set for this SPT node, shape `(batch, n_local_classes)`

### Architecture

```
Linear(embed_dim -> head_hidden)
BatchNorm1d(head_hidden)
ReLU
Dropout(dropout_rate)
Linear(head_hidden -> n_local_classes)
```

### Why an MLP and not a single linear layer

Distinguishing spectrally similar materials (corn-notill vs corn-mintill, two similar minerals) from a fixed embedding through a single linear projection requires the embedding to be perfectly linearly separable for those classes. An MLP with one hidden layer provides non-linear decision boundaries — enough capacity for hard within-group separations without being so large that it memorizes.

### Independent per SPT node, per ensemble member

Each SPT node trains its own encoder AND its own classification head. Each ensemble member at that node is a fully independent (encoder + head) pair with a different random seed.

This is identical to how current GHOST works — each node trains a full independent model. The only change is what the model architecture is (1D ResNet + MLP instead of 3D Conv + U-Net + linear).

**Why not a shared encoder with per-node heads:** A shared encoder receives conflicting gradients from different nodes (root needs broad vegetation-vs-mineral features; leaf needs subtle chlorophyll-concentration features). The compromise representation is suboptimal for every node. Additionally, ensemble diversity collapses — if the encoder is shared, ensemble members only differ in their head initialization, producing near-identical models. Independent encoders have genuine diversity from different local optima.

---

## Stage 4 — Spectral Partition Tree (SPT) Ensemble Inference

**Unchanged from current GHOST.** This section documents how it works for completeness.

### Tree construction (before training)

1. Compute mean spectrum per class (after continuum removal)
2. Build pairwise SAM (Spectral Angle Mapper) distance matrix
3. Find the two most spectrally distant classes -> split seeds
4. Assign remaining classes to whichever seed they're closer to (spectrally similar classes group together)
5. Apply pixel balance correction
6. Recurse on each branch

**Stopping conditions:**
- Fewer than 3 classes in a node
- Any class has fewer than 10 pixels
- Depth >= 3
- Mean intra-node SAM distance < 0.05

### Why grouping similar classes is correct

The tree is a coarse-to-fine hierarchy. Routing between nodes handles easy separations (vegetation vs mineral). Models at each node focus entirely on hard within-group distinctions (soybean subtypes). If you grouped dissimilar classes together, the easy separations would consume model capacity while the hard ones never get a focused specialist.

### Training

Each tree node trains an ensemble of `--ensembles` independent models (encoder + head). Each member uses a different random seed. Best model per member is checkpointed by validation mIoU. Epoch budget scales with node complexity: `max(epochs//2, epochs * node_classes / total_classes)`.

### Inference

Ensemble routing: softmax probabilities are averaged across ensemble members at each node. Predictions are propagated down the tree, accumulating global-class probabilities. Final prediction is argmax over accumulated probabilities.

Source (to port from): `ghost/rssp/sam_clustering.py`, `ghost/rssp/rssp_trainer.py`, `ghost/rssp/rssp_inference.py`

---

## Stage 5 — Dense CRF Post-Processing (Optional)

**Input:** Per-pixel class probability map from SPT, shape `(H, W, n_classes)`

**Output:** Smoothed prediction map, shape `(H, W)`

### Why this exists

Per-pixel classification with zero spatial context produces salt-and-pepper noise. A single noisy pixel in a corn field might get classified as soybean due to sensor noise, atmospheric artifacts, or mixed-pixel effects. The classification is spectrally honest but visually noisy.

### What it does

A Dense CRF balances two signals:
- **Unary potential:** "The spectral model says this pixel is class X with probability 0.93" (from SPT output)
- **Pairwise potential:** "This pixel is spatially close to and spectrally similar to its neighbors, so they should probably have the same class"

When the unary is strong (high confidence), the CRF barely changes the prediction. When the unary is weak (uncertain pixel), the pairwise has more influence. This naturally protects confident rare-class predictions while smoothing uncertain boundaries — no explicit confidence threshold needed.

### Philosophical constraint

The CRF operates on OUTPUT probabilities, not on features. The model's classification decision is 100% spectral. The CRF cannot and does not change what the model learns. It is spatial smoothing of predictions, not spatial learning of features. This distinction is fundamental.

### Pairwise potentials

- **Spatial proximity:** Pixels that are close should tend to agree (Gaussian kernel on spatial distance)
- **Spectral similarity:** Pixels with similar spectra should tend to agree (Gaussian kernel on spectral distance between raw spectra)

The spectral similarity term is key — it prevents the CRF from smoothing across material boundaries where neighboring pixels have genuinely different spectra.

### Minority class protection

The CRF's spatial penalty weight (`--crf_weight`) should be kept low. A weak CRF cleans genuine noise (isolated misclassified pixels in homogeneous regions) without erasing legitimate small-area classes. The unary-vs-pairwise balance handles this naturally: if the spectral model is confident about a 2x2 oats patch, the strong unary protects it. No explicit gating mechanism needed.

### When to disable

The `--crf` flag defaults to off. CRF helps remote sensing (large homogeneous regions, spatial coherence is real). It may hurt on medical histopathology at cellular resolution where spatial heterogeneity is legitimate. User's choice.

### Implementation (`postprocessing/crf.py`)

Use `pydensecrf` library (wrapper around the Philipp Krahenbuhl dense CRF). Optional dependency — import inside function, raise ImportError with install instructions if missing.

```python
def apply_crf(probabilities, raw_image, crf_weight=3.0, spatial_sigma=3, spectral_sigma=10, n_iterations=5):
    """Apply dense CRF post-processing to probability map.

    Args:
        probabilities: (H, W, n_classes) float32, softmax probabilities from SPT
        raw_image: (H, W, Bands) float32, the RAW spectra (not CR'd) — used for
                   spectral similarity kernel. Raw is correct here because CRF
                   needs to detect actual material boundaries, not CR-normalized ones.
        crf_weight: pairwise penalty strength. Lower = less smoothing.
        spatial_sigma: std dev of spatial Gaussian kernel (in pixels)
        spectral_sigma: std dev of spectral Gaussian kernel (in reflectance units)
        n_iterations: CRF inference iterations (5 is standard)

    Returns:
        prediction: (H, W) int, refined class labels
    """
```

**pydensecrf usage pattern:**
1. Create `DenseCRF2D(W, H, n_classes)`
2. Set unary: `-log(probabilities)` transposed to `(n_classes, H*W)` C-contiguous float32
3. Add pairwise Gaussian (spatial only): `addPairwiseGaussian(sxy=spatial_sigma, compat=crf_weight)`
4. Add pairwise bilateral (spatial + spectral): `addPairwiseBilateral(sxy=spatial_sigma, srgb=spectral_sigma, rgbim=raw_image_uint8, compat=crf_weight)`
   - For HSI: `raw_image` has more than 3 channels. Use PCA to reduce to 3-5 components for the bilateral kernel, or use the first 3 principal components. This is only for the CRF kernel similarity measure, not for classification.
5. Inference: `d.inference(n_iterations)` → `(n_classes, H*W)` → argmax → reshape to `(H, W)`

**PCA reduction for bilateral kernel:**
```python
from sklearn.decomposition import PCA
raw_flat = raw_image.reshape(-1, bands)
pca = PCA(n_components=min(3, bands))
raw_reduced = pca.fit_transform(raw_flat).reshape(H, W, -1)
# Scale to 0-255 uint8 for pydensecrf
raw_uint8 = ((raw_reduced - raw_reduced.min()) / (raw_reduced.max() - raw_reduced.min()) * 255).astype(np.uint8)
```

**Fallback if pydensecrf not installed:** Simple morphological majority filter — for each pixel, replace prediction with the most common class in a `(k, k)` spatial window (k=3 or 5). Much simpler, no pairwise spectral kernel, but still cleans salt-and-pepper noise. Implement as `apply_majority_filter(prediction, kernel_size=3)`.

---

## Train / Val / Test Protocol

### The principle

Because the model classifies each pixel independently from its spectrum alone, it can learn chemistry from very few examples and generalize to the rest. 50 labeled pixels of corn teach the model what corn's spectral fingerprint looks like — and that fingerprint is the same whether it's the 51st pixel or the 10,000th. This is fundamentally different from spatial models that need to see enough spatial context to learn shapes/textures. Spectral models learn from chemistry, and chemistry is the same everywhere.

This means: train on 50 pixels per class, segment entire scenes. Train on 2 scenes, segment 8 more. The ratio of training data to inference data can be 1:100 or higher. The model doesn't need more data — it needs diverse spectra.

Since the model is per-pixel with no spatial context, there is no spatial leakage. A training pixel and a test pixel sitting next to each other are processed as two independent spectra. Random pixel sampling is fair.

### Mode 1: Single-scene (benchmark comparison)

For comparison with published papers. One image, small training set, large test set.

```
Per class:
  --samples_per_class 50 (default, user-configurable)

  50 pixels -> train
  50 pixels -> val (for early stopping / checkpointing)
  remainder -> test
```

For rare classes where `samples_per_class` exceeds class size: fall back to percentage split (40% train / 30% val / 30% test). Report the actual counts.

Report mean +/- std over 3-5 random seeds for robust estimates.

This matches the protocol used by S2Mamba, HS-Mamba, and other recent papers.

**Implementation (`datasets/splits.py`):** The split logic should be its own module — it's used by both `train.py` and `train_spt.py`.

```python
def split_single_scene(labels, samples_per_class=50, seed=42):
    """Split pixels from a single scene into train/val/test.

    Args:
        labels: (N,) integer labels. 0 = background (excluded).
        samples_per_class: pixels per class for train (same for val).
        seed: random seed for reproducibility.

    Returns:
        train_idx, val_idx, test_idx: arrays of pixel indices
    """
    # For each class:
    #   if class_count >= 2 * samples_per_class + 10:
    #     train: samples_per_class, val: samples_per_class, test: rest
    #   else (rare class):
    #     train: 40%, val: 30%, test: 30%
    # Shuffle within each class before splitting.

def split_multi_scene(scene_labels, train_scenes, val_scenes=None, seed=42):
    """Split pixels across scenes.

    Args:
        scene_labels: dict {scene_name: (N_scene,) labels}
        train_scenes: list of scene names for training
        val_scenes: list of scene names for validation (optional)
        seed: random seed

    Returns:
        train_idx, val_idx per scene, test scenes untouched
    """
    # If val_scenes provided: all pixels from val_scenes go to val
    # If val_scenes not provided: 80/20 random pixel split within train_scenes
    # Test scenes: all pixels are test (pure inference)
```

`--samples_per_class` only applies to single-scene mode. Ignored in multi-scene mode.

### Mode 2: Multi-scene (the real test)

Train and val on a small number of scenes. Run inference on the rest. Within training scenes, split pixels into train and val sets.

```
OHID-1 (10 scenes, 32 bands, ~10K pixels each):
  Scenes 1-2: 80% train, 20% val (from these scenes' pixels)
  Scenes 3-10: 100% test (pure inference, never seen during training)

LUSC (60+ scans):
  Scans 1-2: 80% train, 20% val
  Scans 3-60+: 100% test
```

The model trains on spectra from 2 scenes and must segment 8+ scenes it has never seen. If it works, it learned chemistry. If it doesn't, it didn't.

**Val split behavior:**
- If `--val` / `--val-gt` is provided: those scenes are used entirely for validation (user has full control)
- If `--val` is omitted: val is auto-split from `--train` pixels at 80% train / 20% val (random per-pixel split within training scenes)

**SPT tree construction in multi-scene mode:** Compute class mean spectra across ALL training pixels (from all training scenes). If corn appears in scenes 1 and 2, the mean corn spectrum is the average across all corn pixels from both scenes. More training scenes = more representative mean spectra.

**Validation:** Per-pixel val set, either from training scenes (auto-split) or from explicit val scenes. Early stopping watches val mIoU.

**Metrics:** Report per-scene AND aggregated. "Scene 3: 94.2% OA. Scene 4: 91.8% OA. ... Mean: 93.0% +/- 1.2%."

### Directory mode (`--data-dir`)

When using `--data-dir ./ohid1/ --split scene`, GHOST auto-discovers all scene files in the directory (sorted alphabetically by filename). Assignment rule:
- First 2 scenes (alphabetically) = training (with 80/20 train/val pixel split)
- Remaining scenes = test

The user can override with `--train_scenes 3` to use the first 3 scenes for training instead of 2. For full control over which specific scenes go where, use the explicit `--train`/`--test` flags instead of `--data-dir`.

**For published results:** Alphabetical assignment is a convenience default for quick testing only. Rigorous evaluation should use multiple random scene assignments — e.g., 5 runs with different randomly selected train scenes each time. Report mean +/- std across runs. This controls for confounds in scene ordering (acquisition date, location, sensor configuration).

### Why OHID-1 and LUSC are the key experiments

**OHID-1** (10 distinct scenes, 32 bands): Train on 2, test on 8. Different spatial layouts, same material types. At 32 bands, tests the architecture at low band count (auto-prunes to ~3 ResBlocks).

**LUSC** (60+ scans, 61 bands): Train on 2, test on 50+. Different patients, different tissue samples, different spatial arrangements. Same tissue types with same spectral signatures (hemoglobin, melanin, collagen). Proves cross-patient generalization.

Together: "GHOST segments unseen scenes in both remote sensing and medical imaging without any code or config changes between the two."

---

## CLI Interface

All defaults work out of the box. Power users can override anything.

### Data Input

```bash
# Single file (any supported format)
ghost train_spt --data scene.mat --gt labels.mat
ghost train_spt --data scene.hdr --gt labels.hdr
ghost train_spt --data scene.tif --gt labels.tif
ghost train_spt --data scene.h5 --gt labels.h5

# Multi-scene, explicit
ghost train_spt \
  --train s1.mat,s2.mat --train-gt g1.mat,g2.mat \
  --val s3.mat --val-gt g3.mat \
  --test s4.mat,s5.mat --test-gt g4.mat,g5.mat

# Multi-scene, directory (auto-assigns scenes)
ghost train_spt --data-dir ./ohid1/ --split scene
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data` / `--gt` | — | Single-scene data and ground truth files |
| `--train` / `--train-gt` | — | Training scene(s) for multi-scene mode (comma-separated) |
| `--val` / `--val-gt` | — | Optional: explicit validation scene(s). If omitted, val is auto-split from train at 80/20 |
| `--test` / `--test-gt` | — | Test scene(s) for multi-scene mode (comma-separated) |
| `--data-dir` | — | Directory of scene files for auto-discovery mode |
| `--train_scenes` | `2` | Number of scenes for training in `--data-dir` mode (sorted alphabetically, rest = test) |
| `--samples_per_class` | `50` | Training pixels per class (single-scene mode only, ignored in multi-scene) |

### Encoder

| Flag                | Default    | Description                                            |
|---------------------|------------|--------------------------------------------------------|
| `--backbone`        | `resnet1d` | Encoder architecture. Future: `transformer`, `mamba`   |
| `--embed_dim`       | `128`      | Embedding size. Larger = more capacity, more overfit risk |
| `--encoder_channels`| `64`       | Channel width in encoder                               |
| `--encoder_depth`   | `5`        | Max ResBlocks. Auto-pruned by band count               |
| `--kernel_size`     | `7`        | Conv kernel size. Auto-shrunk for short spectra        |

### Classification Head

| Flag            | Default | Description                          |
|-----------------|---------|--------------------------------------|
| `--head_hidden` | `128`   | Hidden layer size in MLP head        |
| `--dropout`     | `0.3`   | Dropout rate in head                 |

### SPT (unchanged)

| Flag               | Default | Description                     |
|--------------------|---------|---------------------------------|
| `--ensembles`      | `5`     | Ensemble members per node       |
| `--leaf_ensembles` | `3`     | Ensemble members at leaf nodes  |

### Preprocessing

| Flag   | Default | Description                                              |
|--------|---------|----------------------------------------------------------|
| `--cr` | `auto`  | CR mode: `auto` (by band count), `full`, `simple`, `off` |

### Post-Processing

| Flag           | Default | Description                              |
|----------------|---------|------------------------------------------|
| `--crf`        | `off`   | Enable CRF spatial smoothing             |
| `--crf_weight` | `3.0`   | Pairwise penalty strength. Lower = less smoothing |

### Training (unchanged from v0.1.x unless noted)

| Flag          | Default | Description          |
|---------------|---------|----------------------|
| `--loss`      | `dice`  | Loss function: `ce`, `dice`, `focal`, `squared_ce` |
| `--epochs`    | `400`   | Max epochs           |
| `--patience`  | `50`    | Early stopping (by val mIoU) |
| `--lr`        | `1e-3`  | Learning rate        |
| `--min_epochs`| `40`    | Minimum before early stop |

**Optimizer:** AdamW, weight_decay=1e-4 (same as v0.1.x).

**Scheduler:** ReduceLROnPlateau, monitoring val loss, patience=10, factor=0.5 (same as v0.1.x).

---

## Preprocessing and Data Flow

**CR is applied in the dataset class, not in the model.** The dataset loads raw data via the universal loader, applies CR (full, simple, or off — based on `--cr` flag and band count) to every pixel, and stores the normalized spectra. When a batch is requested, it returns CR-normalized spectra as `(batch, 1, bands)` tensors. The encoder receives already-normalized input and never sees raw reflectance.

This means:
- CR is computed once per pixel at dataset creation time (not every epoch)
- The encoder module has no preprocessing logic — it's a pure neural network
- The same CR output is used for SPT tree construction (class mean spectra) and for training
- The CR mode used is stored in the saved model metadata so `predict.py` applies the same mode at inference

### Dataset class design (`datasets/dataset.py`)

The dataset class is a PyTorch `Dataset` that serves per-pixel spectra. It handles both single-scene and multi-scene modes.

**Construction:**
1. Load scene(s) via `loader.load_hsi()` → `(H, W, Bands)` arrays
2. Apply CR to all pixels (vectorized: reshape to `(N, Bands)`, apply, reshape back)
3. Flatten to pixel list: `spectra = (N_total, Bands)`, `labels = (N_total,)`
4. Filter out background pixels (label == 0)
5. Split into train/val/test sets based on mode:
   - Single-scene: `--samples_per_class` per class for train, same for val, rest for test
   - Multi-scene: scenes are pre-assigned, pixel split within train scenes at 80/20
6. Store as numpy arrays (or torch tensors). Each `__getitem__` returns `(spectrum, label)` where spectrum is `(1, Bands)` float32.

**Multi-scene handling:** Each scene is loaded and CR'd independently. Pixels from all training scenes are pooled into one training set. Scene boundaries are not tracked after pooling — the model sees pixels, not scenes. For test scenes, keep scene identity for per-scene metric reporting.

**Key attributes the dataset must expose:**
- `band_count: int` — number of spectral bands
- `n_classes: int` — number of classes (excluding background)
- `class_names: list[str] or None` — class names if available
- `cr_mode: str` — the CR mode actually used ('full', 'simple', 'off')
- `spectra: np.ndarray` — all CR'd spectra for this split, shape `(N, Bands)`
- `labels: np.ndarray` — corresponding labels, shape `(N,)`
- `spatial_info: dict` — for test set: `{scene_name: (H, W, pixel_indices)}` for reconstructing prediction maps

**Flat training mode (`train.py`):** A baseline mode with no SPT. One encoder + one head trained on all classes simultaneously. Useful for debugging the encoder, for datasets with few classes where SPT is unnecessary, and as a baseline to measure SPT's contribution. Same CLI flags minus the SPT-specific ones.

## Model Save / Load Format

The trained model is saved as a single `.pkl` file (same as v0.1.x for CLI compatibility) containing:

```python
{
    'tree': {
        'structure': <SPT tree structure (node IDs, class mappings, split info)>,
        'class_means': <per-class mean spectra used for tree construction>,
    },
    'models': {
        node_id: [
            {
                'encoder_state': <encoder state_dict>,
                'head_state': <head state_dict>,
                'config': {band_count, encoder_channels, embed_dim, ...},
            }
            for each ensemble member
        ]
        for each node_id in tree
    },
    'metadata': {
        'version': '0.2.0',
        'band_count': <int>,
        'n_classes': <int>,
        'class_names': <list or None>,
        'cr_mode': <str: 'full', 'simple', or 'off' — what was actually used>,
    }
}
```

The `config` per ensemble member stores the architecture parameters needed to reconstruct the encoder and head at inference time. `band_count` in metadata tells `predict` how many bands to expect.

Default filename: `spt_models.pkl` (unchanged from v0.1.x).

---

## Project Structure — `ghost_new/`

The v2 implementation lives in `ghost_new/`, completely separate from the v0.1.x codebase. This allows:
- Development without breaking the existing published package
- Side-by-side testing (v0.1.x vs v2 on the same data)
- No Cython build needed during development (`pip install -e .`)
- Clean merge into `ghost/` when v2 is validated and ready for v0.2.0 release

```
ghost_new/
├── __init__.py                     # Version: 0.2.0-dev
├── cli.py                          # CLI entry point
├── train.py                        # Flat training (single model, no SPT)
├── train_spt.py                    # Full SPT pipeline training
├── predict.py                      # Inference + metrics
├── visualize.py                    # Visualization
├── losses.py                       # CE, dice, focal (port from ghost/)
│
├── datasets/
│   ├── loader.py                   # Universal format loader (.mat, .hdr, .tif, .h5)
│   ├── dataset.py                  # Per-pixel dataset: loads scene(s), applies CR, batches spectra
│   └── splits.py                   # Train/val/test split logic (single-scene + multi-scene)
│
├── preprocessing/
│   └── continuum_removal.py        # Three-tier CR: full (hull), simple (slope), off
│
├── models/
│   ├── encoder.py                  # 1D Dilated ResNet (+ future backbone interface)
│   ├── head.py                     # MLP classification head
│   └── ghost_model.py              # Encoder + Head combined (what SPT nodes train)
│
├── spt/                            # Renamed from rssp/ for clarity
│   ├── tree.py                     # SPT tree construction (SAM clustering)
│   ├── trainer.py                  # Per-node ensemble training
│   └── inference.py                # Ensemble routing + probability accumulation
│
├── postprocessing/
│   └── crf.py                      # Dense CRF (optional)
│
└── utils/
    └── display.py                  # ASCII art, progress bars (port from ghost/)
```

### What gets ported from `ghost/`

| From `ghost/` | To `ghost_new/` | Changes |
|----------------|-----------------|---------|
| `rssp/sam_clustering.py` | `spt/tree.py` | Rename only, logic unchanged |
| `rssp/rssp_trainer.py` | `spt/trainer.py` | Update to instantiate new model (encoder + head) |
| `rssp/rssp_inference.py` | `spt/inference.py` | Minimal changes, routing logic unchanged |
| `losses.py` | `losses.py` | Port as-is |
| `utils/display.py` | `utils/display.py` | Port as-is |
| `cli.py` | `cli.py` | Update for new flags + multi-scene |
| `predict.py` | `predict.py` | Update for new model + add CRF option |
| `visualize.py` | `visualize.py` | Port, update for new model output |

### What is NEW (not ported)

| Module | Why new |
|--------|---------|
| `datasets/loader.py` | Universal format support (replaces .mat-only loading) |
| `datasets/dataset.py` | Per-pixel batching + multi-scene support (replaces image-level dataset) |
| `datasets/splits.py` | Proper per-class sampling + multi-scene split logic |
| `preprocessing/continuum_removal.py` | Three-tier CR: full hull, simple slope, off (replaces straight-line-only CR) |
| `models/encoder.py` | 1D Dilated ResNet (replaces 3D Conv + SE + U-Net) |
| `models/head.py` | MLP head (replaces linear head) |
| `models/ghost_model.py` | Encoder + Head wrapper |
| `postprocessing/crf.py` | Dense CRF (entirely new) |

---

## What This Architecture Does NOT Have

**No spatial convolutions.** No 2D or 3D kernels. No U-Net. No skip connections between pixels.

**No fixed band count.** Handles 3 bands to 400+ with automatic adaptation.

**No fixed file format.** Accepts .mat, ENVI, GeoTIFF, HDF5.

**No scene-specific priors.** The model cannot learn "pixels near corn are probably corn" because it never sees neighboring pixels.

**No shared encoder across SPT nodes.** Each node trains its own independent model, preserving ensemble diversity and allowing per-node specialization.

**No spatial leakage in evaluation.** Per-pixel model means random pixel splits are already fair. Multi-scene splits are the real generalization test.

---

## Discarded Alternatives (brief)

- **Optical depth transform (tau = -ln F):** Incorrect physics for reflectance spectroscopy. Applies to transmission spectroscopy only.
- **Wavelength positional embedding:** Requires wavelength metadata that most formats don't guarantee. Redundant for CNN.
- **Multi-scale parallel branches (Inception-style):** Redundant with dilated convolutions. Triples per-block compute.
- **Shallow Transformer after CNN:** SPT already handles global context. Anti-zero-config. Overkill for leaf nodes.
- **Attention Pooling replacing GAP:** Adds learnable parameters that risk overfitting on small SPT nodes. Valid future upgrade.
- **Explicit absorption feature extraction:** Introduces hyperparameters. The encoder's job is to find dips — let it. Deferred.
- **Confidence-gated CRF:** A weak CRF's unary-vs-pairwise balance already does this naturally.
- **Shared encoder with per-node heads:** Gradient conflicts, ensemble diversity collapse.
- **Mamba SSM backbone:** `mamba-ssm` CUDA-only, breaks cross-platform builds. Valid future `--backbone mamba` option.
- **Disjoint spatial block splits:** Unnecessary when model is per-pixel with no spatial context. Random pixel sampling is already fair.
- **Minimum hull vertex spacing (CR noise defense):** Considered as a second noise filter alongside Savitzky-Golay smoothing. Dropped — smoothing alone handles noise, and vertex spacing risks merging legitimate closely-spaced absorption doublets (some minerals have features 8-10 bands apart). One defense layer is cleaner than two competing heuristics.
- **scipy.spatial.ConvexHull for CR:** Computes full 2D hull (upper + lower), has QHull degenerate-input errors, unnecessary dependency. Replaced with Andrew's monotone chain (upper hull only, O(n), handles all edge cases natively).

---

## Open Design Questions (to resolve during validation)

1. **CR auto threshold:** The band count threshold for full vs simple CR (currently 64) needs empirical validation. Test full CR on OHID-1 (32 bands) — if it helps, lower the threshold. If it hurts, the threshold is correct or should be raised.
2. **CRF parameter selection:** How to set spatial and spectral sigma values that work across datasets without tuning.
3. **SSL integration (future):** Masked spectral modeling for unsupervised spectral representation learning. Required before cross-sensor/cross-dataset transfer is viable.

## Honest Scope of v0.2.0

**What v0.2.0 claims:** One architecture, zero config, retrain on any HSI data. Multi-scene generalization within the same sensor (train on 2 scenes, segment the rest). Demonstrated on remote sensing (OHID-1) and medical imaging (LUSC) without code changes.

**What v0.2.0 does NOT claim:** Cross-sensor transfer (different instruments measuring the same materials). Cross-domain zero-shot (train on crops, segment minerals). These require handling sensor response function differences and are deferred to post-SSL work.

**What is novel vs assembled:** The individual components (convex hull CR, 1D dilated ResNet, CRF, ensembles) are known techniques. The contribution is the specific combination, the zero-config philosophy, the SPT-driven spectral partitioning, and the multi-scene evaluation protocol that no other data-agnostic HSI toolkit provides.
