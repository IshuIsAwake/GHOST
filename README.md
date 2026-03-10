# GHOST — General Hyperspectral Observation & Segmentation Toolkit

A deep learning framework for semantic segmentation of hyperspectral imaging data.
Designed to be sensor-agnostic, domain-agnostic, and configurable for any hyperspectral dataset.

> Think of it as the nnU-Net of hyperspectral imaging.
> The framework is the contribution. The demonstrations are the proof.

---

## Motivation

Hyperspectral imaging captures rich material-specific spectral signatures across
hundreds of wavelength bands. It is used in agriculture, geology, planetary science,
and medical imaging. However, segmentation of hyperspectral data remains fragmented:
most solutions are tightly coupled to a single dataset, sensor, or domain.

Generic RGB segmentation architectures treat spectral bands as interchangeable channels,
discarding the physical structure of spectra. GHOST is built from first principles
around the physics of hyperspectral data.

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
↓
Squeeze and Excitation Block
Dynamically re-weights feature channels based on spectral informativeness.
Learns which wavelength regions matter for the current input.
↓
2D Convolutional Encoder (4 levels)
Reasons spatially over spectral features baked into feature maps.
Produces skip connections at each resolution level.
Filters: 32 → 64 → 128 → 256 → 512
↓
2D U-Net Decoder (4 levels)
Reconstructs full resolution segmentation map using skip connections.
Bilinear interpolation handles arbitrary input resolutions.
↓
Output Segmentation Map (B × num_classes × H × W)
```

**Key design principle:** Every architectural decision is physically motivated.
The 3D convolutions exist because spectral absorption features span ranges of
wavelengths, not single bands. The continuum removal exists because material
identity lives in relative spectral shape, not absolute intensity.

---

## Input / Output Format

### Input
A hyperspectral cube of shape `(H × W × C)` where:
- `H`, `W` are spatial dimensions
- `C` is the number of spectral bands (sensor-dependent)

Currently supported loaders:
- `.mat` files (MATLAB format) — used by Indian Pines, Pavia University, etc.

Planned loaders (see TODO.md):
- `.hdr` / `.img` (ENVI format) — used by AVIRIS, Hyperion
- `.nc` (NetCDF) — used by satellite missions
- `.h5` / `.hdf5` — used by EnMAP, EMIT
- Custom CSV/binary — for instruments like NIRS3 (Hayabusa-2)

### Output
A segmentation map of shape `(H × W)` where every pixel is assigned a class label.

---

## Demonstrated On

### Indian Pines (Earth Remote Sensing)
- Sensor: AVIRIS
- Scene: Agricultural land in Indiana, USA
- Dimensions: 145 × 145 × 200 bands
- Classes: 16 crop and vegetation types
- Format: `.mat`

Results on 20/10/70 stratified pixel split:
```
Test OA:   0.894
Test mIoU: 0.523
Test Dice: (logged in test_results.csv)
```

### Ryugu Asteroid (Planetary Science) — In Progress
- Sensor: NIRS3 (Hayabusa-2 mission, JAXA)
- Scene: Near-Earth asteroid 162173 Ryugu
- Classes: Olivine, pyroxene, carbon-rich surface regions
- Format: Raw spectrometer point data → reprojected onto 3D shape model
- Status: Preprocessing pipeline in development

---

## Reproducibility

All results are reproducible with fixed seeds:
```python
torch.manual_seed(42)
torch.cuda.manual_seed(42)
numpy.random.seed(42)
```

---

## Configuration

Every parameter is defined in a YAML config file.
To adapt this framework to a new dataset, only the config needs to change.
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
```

---

## Training
```bash
python train.py
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

All metrics 

---

## Project Structure
```
GHOST/
├── data/                           # datasets (not tracked by git)
├── datasets/                       # dataset loaders
├── models/                         # architecture components
├── preprocessing/                  # continuum removal and future preprocessors
├── configs/                        # yaml configuration files
├── baseline_version1_results/      # baseline results csv to improve on
├── train.py                        # training and evaluation script
├── README.md
├── TODO.md
└── requirements.txt
```

---

## Authors
- IshuIsAwake
- Abhishek
