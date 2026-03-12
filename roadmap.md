# GHOST Roadmap

What's been done, what's coming next, and what the long-term vision looks like.

---

## Current State (v0.1 — Hackathon Build)

- [x] Universal `.mat` loader with auto-key detection
- [x] Physics-informed Continuum Removal
- [x] Spectral 3D Conv Stack
- [x] Squeeze-and-Excitation channel attention
- [x] 2D U-Net encoder/decoder with skip connections
- [x] RSSP tree construction via SAM clustering
- [x] Per-node HyperspectralNet forest training
- [x] SSSR spectral fingerprint encoder (parallel multi-scale)
- [x] Soft probabilistic cascade inference
- [x] Mixed precision (fp16) support for flat training
- [x] Fully data-agnostic — no hardcoded dataset assumptions

---

## Near-Term (Post-Hackathon)

### Input Format Support
- [ ] ENVI `.hdr` / `.raw` file support
- [ ] GeoTIFF / COG support
- [ ] HDF5 support (common in planetary science)

### SSM Improvements
- [ ] Masked Spectral Modelling pretraining — train SSM to reconstruct randomly masked bands rather than classify pixels. Produces a truly universal spectral encoder that does not require per-dataset pretraining. Zero labelled pixels required for SSM training

### Architecture
- [ ] Adaptive kernel sizes in SSM encoder based on input band count C (currently assumes C ≥ 50)
- [ ] Optional patch-based training for very large scenes that exceed RAM
- [ ] Learned tree topology — differentiable splitting criteria that improve jointly with node model training

### Usability
- [ ] `pip install ghost-hsi` package
- [ ] Single-command inference on a trained model
- [ ] Visualisation utilities (false colour map, per-class accuracy breakdown, tree diagram with node metrics)
- [ ] Config file support (YAML) as alternative to argparse

---

## Long-Term Vision

### Multi-Sensor Generalisation
GHOST currently assumes consistent band ordering across a dataset. Different sensors (AVIRIS, HyMap, CRISM) have different band ranges and resolutions. A band-agnostic representation (learning from wavelength values as metadata rather than band indices) would make GHOST truly sensor-agnostic.

### Self-Supervised Foundation Model
Pretrain SpectralSSMEncoder on a large unlabelled hyperspectral corpus (publicly available Earth observation data). Fine-tune routing heads on any new dataset. This is the "ImageNet pretraining" moment for hyperspectral AI.

### Interactive Relabelling
After initial segmentation, allow users to correct boundary errors on a small number of pixels and retrigger targeted node retraining — without retraining the full tree.

---

## Known Issues to Fix

| Issue | Priority | Notes |
|---|---|---|
| Conv kernel sizes assume C ≥ 50 | Medium | Fix: make kernels proportional to C |
| `rssp_models.pkl` can be very large for deep trees | Low | Compress state dicts or save per-node |
| No inference script — test runs from training script only | Medium | Add `predict.py` |
| Router training uses only train_coords — val routing not evaluated | Low | Add router val BCE to printed metrics |