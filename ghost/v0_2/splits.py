"""Train/val/test splits over labelled pixels, returned as flat (row · W + col) index arrays."""
from __future__ import annotations

import warnings

import numpy as np

SPLIT_MODES = ('ratio', 'fixed', 'disjoint')


def _flat(coords: np.ndarray, width: int) -> np.ndarray:
    return (coords[:, 0] * width + coords[:, 1]).astype(np.int64)


def _class_coords(labels: np.ndarray):
    coords = np.argwhere(labels > 0)
    if len(coords) == 0:
        raise ValueError("The ground truth has no labelled pixels")
    values = labels[coords[:, 0], coords[:, 1]]
    for c in range(1, int(labels.max()) + 1):
        yield c, coords[values == c]


def _stack(parts: list, width: int) -> np.ndarray:
    parts = [p for p in parts if len(p)]
    return _flat(np.concatenate(parts), width) if parts else np.empty(0, dtype=np.int64)


def ratio_split(labels, train_ratio=0.2, val_ratio=0.1, seed=42):
    """v0.1's stratified split, pixel for pixel: per class in order, shuffle, take max(1, int(n · ratio))."""
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    train, val, test = [], [], []
    for _, cc in _class_coords(labels):
        rng.shuffle(cc)
        n = len(cc)
        n_train, n_val = max(1, int(n * train_ratio)), max(1, int(n * val_ratio))
        train.append(cc[:n_train])
        val.append(cc[n_train:n_train + n_val])
        test.append(cc[n_train + n_val:])
    W = labels.shape[1]
    return _stack(train, W), _stack(val, W), _stack(test, W)


def fixed_split(labels, samples_per_class=50, minority_samples=15, val_ratio=0.1, seed=42):
    """Literature protocol: a fixed number of training pixels per class (fewer for small classes);
    val_ratio of the rest validates, the remainder tests."""
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    train, val, test = [], [], []
    for c, cc in _class_coords(labels):
        if len(cc) == 0:
            continue
        rng.shuffle(cc)
        n = len(cc)
        if n >= samples_per_class:
            n_train = samples_per_class
        elif minority_samples is not None and n >= minority_samples:
            n_train = minority_samples
        else:
            n_train = n
            warnings.warn(f"class {c} has only {n} labelled pixels (fewer than "
                          f"{minority_samples or samples_per_class}); all of them go to training", UserWarning)
        rest = cc[n_train:]
        n_val = max(1, int(len(rest) * val_ratio)) if val_ratio > 0 and len(rest) > 1 else 0
        train.append(cc[:n_train])
        val.append(rest[:n_val])
        test.append(rest[n_val:])
    W = labels.shape[1]
    return _stack(train, W), _stack(val, W), _stack(test, W)


def disjoint_split(labels, train_ratio=0.2, val_ratio=0.1, seed=42, block_size=None):
    """Spatially disjoint split: the scene is cut into square blocks and whole blocks go to one split."""
    labels = np.asarray(labels)
    H, W = labels.shape
    block_size = block_size or max(1, min(H, W) // 10)
    coords = np.argwhere(labels > 0)
    if len(coords) == 0:
        raise ValueError("The ground truth has no labelled pixels")
    n_blocks_w = (W + block_size - 1) // block_size
    block_ids = (coords[:, 0] // block_size) * n_blocks_w + coords[:, 1] // block_size
    blocks = np.unique(block_ids)
    np.random.RandomState(seed).shuffle(blocks)
    n_train, n_val = max(1, int(len(blocks) * train_ratio)), max(1, int(len(blocks) * val_ratio))
    in_train = np.isin(block_ids, blocks[:n_train])
    in_val = np.isin(block_ids, blocks[n_train:n_train + n_val])
    splits = (coords[in_train], coords[in_val], coords[~in_train & ~in_val])

    all_classes = set(np.unique(labels[labels > 0]).tolist())
    for name, cc in zip(('train', 'val', 'test'), splits):
        missing = all_classes - set(labels[cc[:, 0], cc[:, 1]].tolist())
        if missing:
            warnings.warn(f"{name} split has no pixels of classes {sorted(missing)}", UserWarning)
    return tuple(_flat(cc, W) for cc in splits)


def make_split(mode, labels, seed=42, train_ratio=0.2, val_ratio=0.1, samples_per_class=50,
               minority_samples=15, block_size=None):
    if mode == 'ratio':
        return ratio_split(labels, train_ratio, val_ratio, seed)
    if mode == 'fixed':
        return fixed_split(labels, samples_per_class, minority_samples, val_ratio, seed)
    if mode == 'disjoint':
        return disjoint_split(labels, train_ratio, val_ratio, seed, block_size)
    raise ValueError(f"Unknown split mode '{mode}'. Choose from: {', '.join(SPLIT_MODES)}")
