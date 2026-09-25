# GHOST — Planned Features

## 0.2.1

- [ ] SPT on the per-pixel model (`train_spt` for architecture 0.2.x)
- [ ] CRF post-processing on predicted probabilities
- [ ] Compiled wheels for Python 3.13/3.14 and ARM Linux; no Python source shipped
- [ ] GeoTIFF prediction output that keeps the input's georeference
- [ ] Continuum removal mode chosen by validation score when the user does not pick one
- [ ] `--arch` accepts `x` and `x.y`, resolving to the newest installed match

## Medium-term

- [ ] Multi-image training — train across multiple scenes from the same sensor
- [ ] Transfer learning — load pretrained GHOST weights and fine-tune on new data
- [ ] Patch-based preprocessing — scenes larger than memory

## Long-term

- [ ] ONNX export — deploy without PyTorch dependency
- [ ] Python API — programmatic `GHOST()` class interface alongside CLI

## Done in 0.2.0

- [x] ENVI, GeoTIFF and HDF5 loading without conversion
- [x] Per-pixel architecture with true continuum removal
- [x] Prediction without ground truth
