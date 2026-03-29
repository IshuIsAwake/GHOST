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
- Native support for ENVI, TIFF, and other common hyperspectral formats is planned for future versions.

## LUSC (Lung Squamous Cell Carcinoma)

LUSC results demonstrate data-agnosticity (GHOST works on non-remote-sensing HSI), but carry significant caveats:

- **Single image only** — trained on 1 of 62 images (a 512×512 crop), not the full dataset
- **Spatial leakage** — train/test split is pixel-level within the same tissue region, not patient-level cross-validation as in published benchmarks
- **Class distribution** — the tumor class forms a large contiguous blob in this crop, making it spatially easy to segment
- **No patient generalisation tested** — results do not indicate clinical applicability

**Preprocessing required:** The raw HMI-LUSC data is in ENVI format (`.raw` + `.hdr`) with PNG label masks. GHOST only accepts `.mat` files, so a one-off conversion is needed:

```bash
python3 -c "
import spectral.io.envi as envi
import numpy as np
from PIL import Image
import scipy.io as sio

img = envi.open('data/lusc/LUSC-3-8/Raw.hdr', 'data/lusc/LUSC-3-8/Raw')
data = np.array(img.load())
label_img = np.array(Image.open('data/lusc/LUSC-3-8/Cell-level Label.png'))

gt = np.zeros((label_img.shape[0], label_img.shape[1]), dtype=np.uint8)
gt[(label_img[:,:,0] == 255) & (label_img[:,:,1] == 0)] = 1  # red  = non-tumor cell
gt[(label_img[:,:,1] == 255) & (label_img[:,:,0] == 0)] = 2  # green = tumor cell
gt[(label_img[:,:,2] == 255) & (label_img[:,:,0] == 0)] = 3  # blue  = non-cell tissue

# 512x512 crop with all 3 classes
y, x, s = 448, 2560, 256
sio.savemat('data/lusc/lusc_512_data.mat', {'data': data[y-s:y+s, x-s:x+s, :]}, do_compression=True)
sio.savemat('data/lusc/lusc_512_gt.mat',   {'gt':   gt[y-s:y+s, x-s:x+s]},      do_compression=True)
print('Saved 512x512 crops')
"
```

## SSM Routing

- `--routing hybrid` and `--routing soft` are experimental. `--routing forest` (ensemble) is the only recommended mode.
- The SSM encoder assumes `num_bands >= 50`. Fails silently on low-band datasets.

## Model Weight Portability (Kaggle / Cloud)

- **Warning:** Downloading model weights (`spt_models.pkl`) from Kaggle via zip download has been observed to silently corrupt the pickle file. The model loads without errors, but produces degraded predictions (e.g., 0.78 mIoU instead of the correct 0.96 mIoU).
- This was confirmed by rerunning `ghost predict` on the same Kaggle environment where `train_spt` ran — both produced identical, correct results. The corruption occurs during the Kaggle download/zip/unzip process, not in the code.
- **Workaround:** Always run `ghost predict` in the same environment where the model was trained to verify results. If you must transfer weights, compare the file size and checksum (`md5sum`) of the original and downloaded pkl to ensure integrity.
