import numpy as np
import scipy.io
import torch
from pathlib import Path
from torch.utils.data import Dataset


# ── Format auto-dispatcher ────────────────────────────────────────────────────

def load_hyperspectral(data_path: str, gt_path: str = None,
                       data_key: str = None, labels_key: str = None):
    """
    Load any supported hyperspectral format and return:
        data   — (C, H, W) float32 numpy array (already normalized)
        labels — (H, W) int64 numpy array (0 = background / unlabeled)
        meta   — dict with format-specific metadata (wavelengths, dims, etc.)

    Supported formats (auto-detected from file extension):
        .mat          — MATLAB / Indian Pines / Pavia style (legacy)
        .hdr          — ENVI format (CRISM exported via CAT, airborne sensors)
        .img / .lbl   — PDS3 format (raw CRISM from NASA PDS)
        .fits / .fit  — FITS format (Ryugu NIRS3, astronomical data)
        .h5 / .hdf5   — HDF5 format (EMIT, various planetary missions)
    """
    from datasets import pds3_reader, envi_reader, fits_reader, hdf5_reader

    ext = Path(data_path).suffix.lower()

    if ext == '.mat':
        # Legacy path — load data and labels from separate .mat files
        data_mat   = scipy.io.loadmat(data_path)
        labels_mat = scipy.io.loadmat(gt_path) if gt_path else {}

        if data_key is None or labels_key is None:
            all_mat = {**data_mat, **(labels_mat if gt_path else {})}
            data_key, labels_key = _find_keys(all_mat)

        data   = data_mat[data_key].astype(np.float32)   # (H, W, C)
        labels = labels_mat[labels_key].astype(np.int64) if gt_path else \
                 np.zeros(data.shape[:2], dtype=np.int64)

        # Normalize (same as before)
        data = (data - data.mean()) / (data.std() + 1e-8)
        # Transpose to (C, H, W)
        data = data.transpose(2, 0, 1)
        meta = {'format': 'mat', 'data_key': data_key, 'labels_key': labels_key}
        return data, labels, meta

    elif ext == '.hdr':
        return envi_reader.read(data_path, gt_path)

    elif ext in ('.img', '.lbl'):
        # PDS3: .img is the data, .lbl is the label.
        # Accept either file as input and auto-find the other.
        if ext == '.lbl':
            lbl_path = data_path
            img_path = str(Path(data_path).with_suffix('.img'))
            if not Path(img_path).exists():
                img_path = str(Path(data_path).with_suffix('.IMG'))
        else:
            img_path = data_path
            lbl_path = gt_path  # may be None, pds3_reader will auto-find
        return pds3_reader.read(img_path, lbl_path)

    elif ext in ('.fits', '.fit'):
        return fits_reader.read(data_path, gt_path)

    elif ext in ('.h5', '.hdf5'):
        return hdf5_reader.read(data_path, gt_path, data_key, labels_key)

    else:
        raise ValueError(
            f"Unsupported file format: '{ext}'\n"
            f"Supported: .mat  .hdr  .img/.lbl  .fits/.fit  .h5/.hdf5"
        )


def _find_keys(mat: dict):
    """
    Auto-detect data and label keys from a .mat file.
    Data key: array with 3 dimensions (H, W, C)
    Labels key: array with 2 dimensions (H, W)
    """
    candidates = {k: v for k, v in mat.items() if not k.startswith('__')}

    data_key   = None
    labels_key = None

    for k, v in candidates.items():
        arr = np.array(v)
        if arr.ndim == 3:
            data_key = k
        elif arr.ndim == 2:
            labels_key = k

    if data_key is None or labels_key is None:
        raise ValueError(
            f"Could not auto-detect keys.\n"
            f"Available: {list(candidates.keys())}\n"
            f"Pass manually: HyperspectralDataset(..., data_key='...', labels_key='...')"
        )

    return data_key, labels_key


class HyperspectralDataset(Dataset):
    """
    Universal loader for hyperspectral .mat datasets.
    Works with Indian Pines, Pavia, Salinas, Houston, Botswana, KSC, and any other .mat dataset.

    Usage:
        dataset = HyperspectralDataset('data.mat', 'labels.mat', split='train')
        model = HyperspectralNet(num_bands=dataset.num_bands, num_classes=dataset.num_classes)
    """

    def __init__(self, data_path, gt_path, split='train',
                 train_ratio=0.2, val_ratio=0.1,
                 data_key=None, labels_key=None,
                 seed=42 , use_fp16=False):

        # ── Load (auto-detects format: .mat, .hdr, .img, .fits, .h5) ─────────
        data_np, labels_np, self.meta = load_hyperspectral(
            data_path, gt_path, data_key, labels_key
        )
        # data_np is already (C, H, W) float32, normalized by the reader

        C, H, W = data_np.shape

        if use_fp16:
            # Min-Max rescale to [0,1] to avoid fp16 overflow in ContinuumRemoval
            d_min, d_max = data_np.min(), data_np.max()
            data_np = (data_np - d_min) / (d_max - d_min + 1e-8)

        self.num_bands   = C
        self.num_classes = int(labels_np.max()) + 1  # +1 to include background 0

        self.data   = torch.tensor(data_np).float()        # (C, H, W)
        self.labels = torch.tensor(labels_np).long()       # (H, W)

        # ── Stratified split ──────────────────────────────────────────────────
        labeled_coords = np.argwhere(labels_np > 0)
        np.random.seed(seed)

        train_coords, val_coords, test_coords = [], [], []

        for class_id in range(1, self.num_classes):
            class_coords = labeled_coords[
                labels_np[labeled_coords[:, 0], labeled_coords[:, 1]] == class_id
            ]
            np.random.shuffle(class_coords)

            n       = len(class_coords)
            n_train = max(1, int(n * train_ratio))
            n_val   = max(1, int(n * val_ratio))

            train_coords.append(class_coords[:n_train])
            val_coords.append(class_coords[n_train:n_train + n_val])
            test_coords.append(class_coords[n_train + n_val:])

        self.train_coords = np.concatenate(train_coords)
        self.val_coords   = np.concatenate(val_coords)
        self.test_coords  = np.concatenate(test_coords)

        split_map = {
            'train': self.train_coords,
            'val':   self.val_coords,
            'test':  self.test_coords
        }
        if split not in split_map:
            raise ValueError(f"split must be 'train', 'val', or 'test'. Got: '{split}'")
        self.coords = split_map[split]

        # ── Split mask ────────────────────────────────────────────────────────
        self.split_mask = torch.zeros(self.labels.shape, dtype=torch.long)
        for r, c in self.coords:
            self.split_mask[r, c] = self.labels[r, c]

        print(f"{split}: {len(self.coords)} pixels across {self.num_classes - 1} classes")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data, self.split_mask