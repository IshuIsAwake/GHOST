import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

# --- TUNABLE DATASET PARAMETERS ---
DEFAULT_TRAIN_RATIO = 0.2
DEFAULT_VAL_RATIO = 0.1
DEFAULT_PATCH_SIZE = 32
DEFAULT_STRIDE = 8

class PaviaUniversityDataset(Dataset):
    def __init__(self, data_path, gt_path, split='train', 
                 train_ratio=DEFAULT_TRAIN_RATIO, val_ratio=DEFAULT_VAL_RATIO, 
                 patch_size=DEFAULT_PATCH_SIZE, stride=DEFAULT_STRIDE,
                 use_patches=False, seed=42):
        
        # Load Pavia University (.mat keys are usually 'paviaU' and 'paviaU_gt')
        data = scipy.io.loadmat(data_path)['paviaU']
        labels = scipy.io.loadmat(gt_path)['paviaU_gt']

        data = data.astype(np.float32)
        data = (data - data.mean()) / (data.std() + 1e-8)

        self.data = torch.tensor(data).permute(2, 0, 1)  # (103, 610, 340)
        self.labels = torch.tensor(labels).long()        # (610, 340)
        
        self.use_patches = use_patches
        self.patch_size = patch_size

        labeled_coords = np.argwhere(labels > 0)

        np.random.seed(seed)
        train_coords, val_coords, test_coords = [], [], []
        class_counts = np.zeros(10) # 9 classes + background

        # Stratified Split
        for class_id in range(1, 10):
            class_coords = labeled_coords[labels[labeled_coords[:,0], labeled_coords[:,1]] == class_id]
            np.random.shuffle(class_coords)

            n = len(class_coords)
            n_train = max(1, int(n * train_ratio))
            n_val   = max(1, int(n * val_ratio))

            train_coords.append(class_coords[:n_train])
            val_coords.append(class_coords[n_train:n_train+n_val])
            test_coords.append(class_coords[n_train+n_val:])
            
            class_counts[class_id] = n_train 

        self.train_coords = np.concatenate(train_coords)
        self.val_coords   = np.concatenate(val_coords)
        self.test_coords  = np.concatenate(test_coords)

        if split == 'train':
            self.coords = self.train_coords
        elif split == 'val':
            self.coords = self.val_coords
        else:
            self.coords = self.test_coords

        self.split_mask = torch.zeros(self.labels.shape, dtype=torch.long)
        for r, c in self.coords:
            self.split_mask[r, c] = self.labels[r, c]

        print(f"PAVIA {split.upper()}: {len(self.coords)} labeled pixels active.")

        if self.use_patches:
            self.patches = []
            self.patch_weights = []
            
            inv_class_weights = np.zeros(10)
            for c in range(1, 10):
                if class_counts[c] > 0:
                    inv_class_weights[c] = 1.0 / class_counts[c]
            
            if inv_class_weights.sum() > 0:
                inv_class_weights = inv_class_weights / inv_class_weights.max()
            
            h, w = self.labels.shape
            for r in range(0, h - patch_size + 1, stride):
                for c in range(0, w - patch_size + 1, stride):
                    patch_mask = self.split_mask[r:r+patch_size, c:c+patch_size]
                    
                    if patch_mask.sum() > 0:
                        self.patches.append((r, c))
                        pixel_classes = patch_mask[patch_mask > 0].numpy()
                        patch_weight = np.sum(inv_class_weights[pixel_classes])
                        self.patch_weights.append(patch_weight + 1e-4)
                        
            print(f"--> Extracted {len(self.patches)} training patches (Size: {patch_size}, Stride: {stride})")

    def __len__(self):
        return len(self.patches) if self.use_patches else 1

    def __getitem__(self, idx):
        if self.use_patches:
            r, c = self.patches[idx]
            p_data = self.data[:, r:r+self.patch_size, c:c+self.patch_size]
            p_mask = self.split_mask[r:r+self.patch_size, c:c+self.patch_size]
            return p_data, p_mask
        
        return self.data, self.split_mask