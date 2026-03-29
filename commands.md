# GHOST — Quick Commands

---

## Version & Info

```bash
ghost --version
ghost version
ghost demo
```

---

## Indian Pines

```bash
# Train (Dice loss, SPT + ensembles)
ghost train_spt \
  --data data/indian_pines/Indian_pines_corrected.mat \
  --gt   data/indian_pines/Indian_pines_gt.mat \
  --loss dice \
  --base_filters 32 --num_filters 8 --num_blocks 3 \
  --ensembles 5 --leaf_ensembles 3 \
  --epochs 400 --patience 75 --min_epochs 40 \
  --val_interval 20 \
  --out-dir runs/local_runs/indian_ce_dice

# Predict
ghost predict \
  --data  data/indian_pines/Indian_pines_corrected.mat \
  --gt    data/indian_pines/Indian_pines_gt.mat \
  --model runs/indian_pines_dice/spt_models.pkl \
  --routing forest --out-dir runs/localc_runs/indian_ce_dice

# Visualize
ghost visualize \
  --data  data/indian_pines/Indian_pines_corrected.mat \
  --gt    data/indian_pines/Indian_pines_gt.mat \
  --model runs/indian_pines_dice/spt_models.pkl \
  --routing forest --dataset indian_pines \
  --out-dir runs/localc_runs/indian_ce_dice
```

---

## Pavia University

```bash
# Train
ghost train_spt \
  --data data/pavia/PaviaU.mat \
  --gt   data/pavia/PaviaU_gt.mat \
  --loss dice \
  --base_filters 16 --num_filters 4 --num_blocks 3 \
  --ensembles 5 --leaf_ensembles 3 \
  --epochs 400 --patience 75 --min_epochs 40 \
  --val_interval 20 \
  --out-dir runs/pavia_dice

# Predict
ghost predict \
  --data  data/pavia/PaviaU.mat \
  --gt    data/pavia/PaviaU_gt.mat \
  --model runs/kaggle/pavia/runs/pavia_dice/spt_models.pkl  


# Visualize
ghost visualize \
  --data  data/pavia/PaviaU.mat \
  --gt    data/pavia/PaviaU_gt.mat \
  --model runs/kaggle/salinas/runs/salinas_dice/spt_models.pkl \ 
  --out-dir runs/pavia_dice_kaggle
```

---

## Salinas

```bash
# Train
ghost train_spt \
  --data data/salinas/Salinas_corrected.mat \
  --gt   data/salinas/Salinas_gt.mat \
  --loss dice \
  --base_filters 16 --num_filters 4 --num_blocks 3 \
  --ensembles 5 --leaf_ensembles 3 \
  --epochs 400 --patience 50 --min_epochs 40 \
  --val_interval 20 \
  --out-dir runs/salinas_dice

# Predict
ghost predict \
  --data  data/salinas/Salinas_corrected.mat \
  --gt    data/salinas/Salinas_gt.mat \
  --model runs/salinas_ce_dice/rssp_models.pkl \
  --routing forest --out-dir runs/salinas_ce_dice

# Visualize
ghost visualize \
  --data  data/salinas/Salinas_corrected.mat \
  --gt    data/salinas/Salinas_gt.mat \
  --model runs/salinas_ce_dice/rssp_models.pkl \
  --out-dir runs/salinas_ce_dice
```

---

## LUSC (Lung Squamous Cell Carcinoma)

> **Preprocessing required:** The raw data is ENVI format. Convert to .mat first.

```bash
# One-off conversion: ENVI + PNG labels → .mat (run once)
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

# Train
ghost train_spt \
  --data data/lusc/lusc_512_data.mat \
  --gt   data/lusc/lusc_512_gt.mat \
  --loss dice \
  --base_filters 32 --num_filters 8 --num_blocks 3 \
  --ensembles 5 --leaf_ensembles 3 \
  --epochs 400 --patience 50 --min_epochs 40 \
  --val_interval 20 \
  --out-dir runs/lusc_dice

# Predict
ghost predict \
  --data  data/lusc/lusc_512_data.mat \
  --gt    data/lusc/lusc_512_gt.mat \
  --model runs/lusc_dice/spt_models.pkl \
  --routing forest --out-dir runs/lusc_dice

# Visualize
ghost visualize \
  --data  data/lusc/lusc_512_data.mat \
  --gt    data/lusc/lusc_512_gt.mat \
  --model runs/lusc_dice/spt_models.pkl \
  --routing forest \
  --out-dir runs/lusc_dice
```

---

## Output Files

| File | Description |
|------|-------------|
| `spt_models.pkl` | Trained model checkpoint (all ensembles + router weights) |
| `ssm_pretrained.pt` | Pre-trained SSM encoder weights |
| `training_history.csv` | Per-epoch metrics: OA, mIoU, Dice, Precision, Recall, AA, Kappa |
| `test_results.csv` | Final test metrics summary |
| `class_report_{routing}.csv` | Per-class IoU, Precision, Recall with pixel counts |
| `prediction_map.npy` | Full prediction map (from `predict`) |
| `*.png` | Visualization outputs (from `visualize`) |
