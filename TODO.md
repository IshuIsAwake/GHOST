# GHOST — Roadmap & Known Limitations

> Transparency builds trust. This document lists what works, what doesn't, and what's planned.

---

## Known Limitations

### Spectral State-Space Model (SSM)

The SSM encoder (`--routing hybrid` / `--routing soft`) is **experimental**. It was designed to learn spectral fingerprints for probabilistic routing through the RSSP tree. In practice:

- Forest routing (`--routing forest`) consistently outperforms hybrid and soft routing
- SSM pretraining adds 30-60 minutes of training time with marginal or negative returns
- The SSM architecture assumes `num_bands >= 50` (wide kernel branch). Fails silently on low-band datasets
- **Recommendation:** Use `--routing forest` for all production runs. SSM routing is disabled by default in the recommended configs

The SSM module will be reworked or replaced in a future version.

### File Format Support

Currently supports only `.mat` (MATLAB/HDF5) files. This requires users to convert from other common hyperspectral formats before using GHOST:

- ENVI (.raw + .hdr) — the most common format in remote sensing
- GeoTIFF (.tif / .tiff) — standard for geospatial data
- HDF5 (.h5 / .he5) — used by NASA instruments (e.g., AVIRIS-NG, Hyperion)

Workaround: Use `scipy.io.loadmat()` or `spectral` Python library to convert to `.mat` before running GHOST.

### Single Image Training

GHOST currently trains on one hyperspectral scene at a time. There is no support for:

- Training across multiple images from the same sensor
- Patient-level or scene-level cross-validation
- Dataset-level data loaders that iterate over multiple files

This means results on medical datasets (LUSC) or multi-scene campaigns are not directly comparable to published benchmarks that use proper cross-validation.

### No Transfer Learning

Each training run starts from scratch. There is no support for:

- Loading pretrained weights from one dataset and fine-tuning on another
- Pre-trained backbone weights (ImageNet, etc.)
- Domain adaptation between sensors

### Spatial Context

- Spectral context is local only (7-band convolutional kernel). Long-range spectral dependencies are not modelled
- No spatial regularisation — predictions are per-pixel, homogeneous regions can show speckle noise
- No CRF or post-processing for boundary refinement

### Class Imbalance

- RSSP tree structure helps (splits rare classes into their own nodes)
- Dice loss (`--loss dice`) mitigates imbalance at the loss level
- But classes with <30 training pixels (e.g., Indian Pines Class 7, 9) still underperform significantly
- No oversampling, SMOTE, or class-weighted sampling implemented

### Platform Support

- **Linux only** (compiled wheels are platform-specific)
- No Windows or macOS wheels yet
- Workaround: Use Google Colab (free GPU, Linux-based)

---

## Planned Features

### Near-term

- [ ] **ENVI file support** — native `.raw` + `.hdr` loading without manual conversion
- [ ] **GeoTIFF support** — `.tif` / `.tiff` loading via `rasterio`
- [ ] **HDF5 support** — `.h5` / `.he5` loading via `h5py`
- [ ] **Windows / macOS wheels** — CI-built wheels for all major platforms
- [ ] **Bundled sample data** — Indian Pines included with `pip install` for instant demo

### Medium-term

- [ ] **Multi-image training** — train across multiple scenes from the same sensor
- [ ] **Transfer learning** — load pretrained GHOST weights and fine-tune on new data
- [ ] **SSM rework** — replace current SSM with a more effective spectral encoder, or remove it entirely
- [ ] **Spatial post-processing** — CRF or morphological cleanup for smoother predictions
- [ ] **Class-weighted sampling** — address extreme class imbalance at the data loader level

### Long-term

- [ ] **Patch-based training** — enable training on scenes larger than GPU memory
- [ ] **Multi-GPU / distributed** — for large-scale campaigns
- [ ] **ONNX export** — deploy trained models without PyTorch dependency
- [ ] **Streaming inference** — process tiles of large scenes incrementally
- [ ] **Python API** — programmatic `GHOST()` class interface alongside CLI

---

## Completed

- [x] RSSP tree with forest ensemble training
- [x] Continuum removal preprocessing (replaces PCA)
- [x] Dice loss, focal loss, squared CE loss options
- [x] Early stopping with patience and minimum epochs
- [x] Leaf forest count control (`--leaf_forests`)
- [x] Compiled binary distribution via PyPI (`pip install ghost-hsi`)
- [x] CLI with ASCII art, progress bars, GPU monitoring
- [x] Three-panel visualisation (false colour / ground truth / prediction)
- [x] Tested on: Indian Pines, Salinas, Pavia University, LUSC (medical), Mars CRISM, Asteroid Ryugu
