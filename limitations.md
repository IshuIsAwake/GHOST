# GHOST — Limitations

Things that don't work well or aren't supported. I'm trying to be straightforward here.

---

## Evaluation methodology

This is probably the biggest caveat. All results use pixel-level train/test splits on a single scene. Nearby pixels are spatially correlated, and the train/test sets are interleaved — so the numbers are optimistic compared to what you'd get with spatially disjoint splits or cross-scene evaluation. This is standard practice for these benchmarks (most published papers do the same), but it's still a real limitation.

I haven't done scene-level cross-validation or tested on held-out scenes from the same sensor.

## Spectral context

- Spectral context is local only (7-band convolutional kernel). Long-range spectral dependencies are not modelled.
- The SSM encoder (`--routing hybrid` / `--routing soft`) was supposed to address this but it underperforms ensemble routing in every test. Needs a rethink or removal.

## Spatial predictions

- Predictions are per-pixel. No CRF, morphological cleanup, or spatial regularisation.
- Homogeneous regions can show speckle noise in the prediction map.

## Class imbalance

- The SPT tree structure helps somewhat (splits rare classes into their own nodes).
- Dice loss helps at the loss level.
- Classes with <30 training pixels still do badly (e.g., Indian Pines Class 7, 9).
- No oversampling, SMOTE, or class-weighted sampling.

## Single image training

- GHOST trains on one hyperspectral scene at a time.
- No multi-image training, scene-level cross-validation, or dataset-level data loaders.
- This is the main reason the LUSC results are not meaningful — proper medical HSI evaluation needs patient-level cross-validation.

## No transfer learning

- Every training run starts from scratch. No pretrained weights, no fine-tuning.

## File format

- Only `.mat` (MATLAB/HDF5) files are supported.
- ENVI (`.raw` + `.hdr`), GeoTIFF, and HDF5 (`.h5`) need manual conversion.
- I want to add native support for these eventually.

## LUSC specifics

The LUSC results show that the pipeline handles non-remote-sensing HSI, but that's about all they show:

- **Single image only** — 1 of 62 images (a 512×512 crop)
- **Spatial leakage** — pixel-level split within the same tissue region
- **Easy geometry** — tumor is a big contiguous blob in this crop
- **No patient generalization tested**

**Preprocessing required:** Raw HMI-LUSC data is ENVI format. Conversion to `.mat`:

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

## SSM routing

- `--routing hybrid` and `--routing soft` are experimental and worse than `--routing forest` in all tests.
- The SSM encoder assumes `num_bands >= 50`. Fails silently on low-band datasets.

## Model weight portability (Kaggle / cloud)

- Downloading model weights (`spt_models.pkl`) from Kaggle via zip download can silently corrupt the pickle file. The model loads fine but predictions are degraded.
- Workaround: always run `ghost predict` in the same environment where you trained. If transferring weights, check the file size and md5sum.
