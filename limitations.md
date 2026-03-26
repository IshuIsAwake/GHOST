# GHOST — Limitations

An honest list of what GHOST does not do well or does not support.

---

## Spectral Context

- Spectral context is local only (7-band convolutional kernel). Long-range spectral dependencies are not modelled.
- The SSM encoder (`--routing hybrid` / `--routing soft`) was designed to address this but currently underperforms ensemble routing. It will be reworked or removed.

## Spatial Predictions

- Predictions are per-pixel. No CRF, morphological cleanup, or spatial regularisation is applied.
- Homogeneous regions can show speckle noise in the prediction map.

## Class Imbalance

- The SPT tree structure helps (splits rare classes into their own nodes).
- Dice loss mitigates imbalance at the loss level.
- Classes with <30 training pixels still underperform significantly (e.g., Indian Pines Class 7, 9).
- No oversampling, SMOTE, or class-weighted sampling is implemented.

## Single Image Training

- GHOST trains on one hyperspectral scene at a time.
- No support for multi-image training, scene-level cross-validation, or dataset-level data loaders.
- This means results on medical datasets (LUSC) are not comparable to published benchmarks that use patient-level cross-validation.

## No Transfer Learning

- Each training run starts from scratch. No pretrained weights, no fine-tuning, no domain adaptation between sensors.

## File Format

- Only `.mat` (MATLAB/HDF5) files are supported.
- ENVI (`.raw` + `.hdr`), GeoTIFF, and HDF5 (`.h5`) require manual conversion before use.

## SSM Routing

- `--routing hybrid` and `--routing soft` are experimental. `--routing forest` (ensemble) is the only recommended mode.
- The SSM encoder assumes `num_bands >= 50`. Fails silently on low-band datasets.
