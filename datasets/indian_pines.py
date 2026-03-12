import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

class IndianPinesDataset(Dataset):
    def __init__(self, data_path, gt_path, split='train', train_ratio=0.1, val_ratio=0.1, seed=42):
        data = scipy.io.loadmat(data_path)['indian_pines_corrected']
        labels = scipy.io.loadmat(gt_path)['indian_pines_gt']

        data = data.astype(np.float32)
        data = (data - data.mean()) / (data.std() + 1e-8)

        self.data = torch.tensor(data).permute(2, 0, 1)  # (200, 145, 145)
        self.labels = torch.tensor(labels).long()         # (145, 145)

        # Get labeled pixel coordinates
        labeled_coords = np.argwhere(labels > 0)  # (N, 2)

        # Stratified split - ensure every class is represented in train/val
        np.random.seed(seed)
        train_coords, val_coords, test_coords = [], [], []

        for class_id in range(1, 17):
            class_coords = labeled_coords[labels[labeled_coords[:,0], labeled_coords[:,1]] == class_id]
            np.random.shuffle(class_coords)

            n = len(class_coords)
            n_train = max(1, int(n * train_ratio))
            n_val   = max(1, int(n * val_ratio))

            train_coords.append(class_coords[:n_train])
            val_coords.append(class_coords[n_train:n_train+n_val])
            test_coords.append(class_coords[n_train+n_val:])

        self.train_coords = np.concatenate(train_coords)
        self.val_coords   = np.concatenate(val_coords)
        self.test_coords  = np.concatenate(test_coords)

        if split == 'train':
            self.coords = self.train_coords
        elif split == 'val':
            self.coords = self.val_coords
        else:
            self.coords = self.test_coords

        # Build split mask for full image evaluation
        self.split_mask = torch.zeros(145, 145, dtype=torch.long)
        for r, c in self.coords:
            self.split_mask[r, c] = self.labels[r, c]

        print(f"{split}: {len(self.coords)} pixels across 16 classes")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data, self.split_mask