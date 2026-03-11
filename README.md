# GHOST — General Hyperspectral Observation & Segmentation Toolkit

A deep learning framework for semantic segmentation of hyperspectral imaging data.
Designed to be sensor-agnostic, domain-agnostic, and configurable for any hyperspectral dataset.

> Think of it as the nnU-Net of hyperspectral imaging.
> The framework is the contribution. The demonstrations are the proof.

---

## Motivation

Hyperspectral imaging captures rich material-specific spectral signatures across hundreds of wavelength bands. It is used in agriculture, geology, planetary science, and medical imaging. However, segmentation of hyperspectral data remains fragmented — most solutions are tightly coupled to a single dataset, sensor, or domain.

Generic RGB segmentation architectures treat spectral bands as interchangeable channels, discarding the physical structure of spectra. GHOST is built from first principles around the physics of hyperspectral data.

---

## Architecture

```
Input Hyperspectral Cube (B × C × H × W)
↓
Continuum Removal
Normalizes spectra across sensors by removing the baseline illumination envelope.
Strips instrument response, preserves material absorption chemistry.
↓
Spectral 3D Convolutional Stack (3 blocks)
3D kernels slide across spectral and spatial dimensions simultaneously.
Learns the shape of absorption features as they evolve across adjacent wavelength bands.
Kernel: (7 spectral × 3 spatial × 3 spatial)
Output: (B, num_filters, C, H, W) — full spectral resolution preserved for transformer.
↓
Spectral Transformer
Each spatial pixel becomes a sequence of C band-tokens of dimension num_filters.
Self-attention learns cross-band relationships (co-occurring absorption features).
Learned positional embeddings are interpolated dynamically → any band count works.
Global average pool over spectral dimension → fixed (B, num_filters, H, W) output.
↓
2D Convolutional Encoder (4 levels, GroupNorm throughout)
Reasons spatially over the spectral features.
Produces skip connections at each resolution level.
Filters: 32 → 64 → 128 → 256 → 512
↓
2D U-Net Decoder (4 levels)
Reconstructs full resolution segmentation map using skip connections.
Bilinear interpolation handles arbitrary input resolutions.
↓
Output Segmentation Map (B × num_classes × H × W)
```

**Key design principles:**
- 3D convolutions capture absorption features that span ranges of wavelengths, not single bands.
- Continuum removal strips sensor-specific response — material identity lives in relative spectral shape.
- The Spectral Transformer collapses variable C bands into a fixed representation via global pooling — this is what makes the model band-count agnostic.
- GroupNorm replaces BatchNorm everywhere — stable at batch size 1, across datasets with different statistics.

---

## Dataset Agnosticism

GHOST accepts any number of spectral bands at inference without retraining. A model trained on Indian Pines (200 bands) can run inference on Pavia University (103 bands) directly. The Spectral Transformer handles this via interpolated positional embeddings.

**Training:** GPU (patches, B=4)
**Val / Test / Full-image inference:** CPU (full image creates too many transformer sequences for 6GB GPU)

---

## Input / Output Format

### Input
A hyperspectral cube of shape `(H × W × C)` where C is sensor-dependent.

Currently supported loaders:
- `.mat` files (MATLAB format) — Indian Pines, Pavia University

Planned loaders:
- `.hdr` / `.img` (ENVI) — AVIRIS, Hyperion
- `.nc` (NetCDF) — satellite missions
- `.h5` / `.hdf5` — EnMAP, EMIT
- Custom binary — NIRS3 (Hayabusa-2)

### Output
Segmentation map `(H × W)` — every pixel assigned a class label.

---

## Demonstrated On

### Indian Pines (Earth Remote Sensing)
- Sensor: AVIRIS
- Scene: Agricultural land, Indiana USA
- Dimensions: 145 × 145 × 200 bands
- Classes: 16 crop and vegetation types
- Format: `.mat`

Current results (real SpectralTransformer, B=4, 16×16 patches):
```
In progress — training run underway.
```

### Pavia University (Zero-Shot Transfer)
- Sensor: ROSIS
- Dimensions: 610 × 340 × 103 bands
- Classes: 9 urban material types
- Zero-shot: model trained on Indian Pines (200 bands), inference on Pavia (103 bands) with no retraining.
- Status: agnosticism confirmed, fine-tuning pending.

### Ryugu Asteroid (Planetary Science)
- Sensor: NIRS3 (Hayabusa-2, JAXA)
- Status: preprocessing pipeline in development.

---

## Reproducibility

```python
torch.manual_seed(42)
torch.cuda.manual_seed(42)
numpy.random.seed(42)
```

---

## Configuration

```yaml
dataset:
  name: indian_pines
  num_bands: 200
  num_classes: 17

model:
  num_filters: 8
  num_blocks: 3
  base_filters: 32

training:
  epochs: 300
  lr: 1e-4
  batch_size: 4
  patch_size: 16
```

---

## Usage

```bash
# Train on Indian Pines
python train.py

# Sanity check all components
python helper.py

# Zero-shot transfer test (requires best_model.pth from training)
python test_pavia_zeroshot.py
```

Outputs:
- `best_model.pth` — best checkpoint by val mIoU
- `training_log.csv` — per epoch metrics
- `test_results.csv` — final test evaluation

---

## Metrics

| Metric | Description |
|--------|-------------|
| OA | Overall Accuracy — fraction of correctly classified labeled pixels |
| mIoU | Mean Intersection over Union — primary segmentation metric |
| Dice | F1 score per class, averaged |
| Precision | Per class, macro averaged |
| Recall | Per class, macro averaged |

---

## Project Structure

```
GHOST/
├── data/                        # datasets (not tracked by git)
├── datasets/                    # dataset loaders
│   ├── indian_pines.py
│   └── pavia_university.py
├── models/                      # architecture components
│   ├── hyperspectral_net.py     # full pipeline
│   ├── spectral_3d_block.py     # 3D conv stack
│   ├── real_spectral_transformer.py  # spectral self-attention
│   ├── encoder_2d.py            # U-Net encoder
│   └── decoder_2d.py            # U-Net decoder
├── preprocessing/
│   └── continuum_removal.py
├── configs/
│   └── indian_pines.yaml
├── train.py
├── helper.py                    # component sanity checks
├── test_pavia_zeroshot.py       # band-agnosticism test
├── README.md
└── TODO.md
```

---

## Hardware Notes

Training verified on RTX 3050 6GB (laptop):
- Training: B=4, 16×16 patches → ~4.4GB peak VRAM
- Val/Test: CPU inference (full image too large for 6GB GPU)
- Kill any zombie GPU processes before training: `nvidia-smi` to check, `kill -9 <PID>` to clear.

---

## Authors
- IshuIsAwake
- Abhishek