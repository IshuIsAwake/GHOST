import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset


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
                 seed=42):

        # ── Load ─────────────────────────────────────────────────────────────
        data_mat   = scipy.io.loadmat(data_path)
        labels_mat = scipy.io.loadmat(gt_path)

        if data_key is None or labels_key is None:
            data_key, labels_key = _find_keys({**data_mat, **labels_mat})

        data   = data_mat[data_key].astype(np.float32)    # (H, W, C)
        labels = labels_mat[labels_key].astype(np.int64)  # (H, W)

        # ── Fingerprint ───────────────────────────────────────────────────────
        H, W, C          = data.shape
        self.num_bands   = C
        self.num_classes = int(labels.max()) + 1  # +1 to include background class 0

        # ── Normalize ─────────────────────────────────────────────────────────
        data = (data - data.mean()) / (data.std() + 1e-8)

        self.data   = torch.tensor(data).permute(2, 0, 1)  # (C, H, W)
        self.labels = torch.tensor(labels).long()          # (H, W)

        # ── Stratified split ──────────────────────────────────────────────────
        labeled_coords = np.argwhere(labels > 0)
        np.random.seed(seed)

        train_coords, val_coords, test_coords = [], [], []

        for class_id in range(1, self.num_classes):
            class_coords = labeled_coords[
                labels[labeled_coords[:, 0], labeled_coords[:, 1]] == class_id
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