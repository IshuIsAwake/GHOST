# GHOST — Planned Features

## Near-term

- [ ] ENVI file support (`.raw` + `.hdr`) — native loading without manual conversion
- [ ] GeoTIFF support (`.tif`) via `rasterio`
- [ ] HDF5 support (`.h5`) via `h5py`

## Medium-term

- [ ] Multi-image training — train across multiple scenes from the same sensor
- [ ] Transfer learning — load pretrained GHOST weights and fine-tune on new data
- [ ] SSM rework — replace current spectral encoder with something that actually helps, or remove it
- [ ] Spatial post-processing — CRF or morphological cleanup for smoother predictions

## Long-term

- [ ] Patch-based training — handle scenes larger than GPU memory
- [ ] ONNX export — deploy without PyTorch dependency
- [ ] Python API — programmatic `GHOST()` class interface alongside CLI
