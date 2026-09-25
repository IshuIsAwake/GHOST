"""Run the unmodified v0.1.7 trainers on v0.2's train/val/test split.

    python benchmarks/v01_split.py --split fixed -- train_spt --data <cube> --gt <labels> [v0.1 flags]

v0.1 builds its own ratio split; this replaces it, right after, with v0.2's split for the same seed, so both
architectures are scored on identical pixels. With --split ratio nothing changes. --tree-from-train builds the
SPT tree from training pixels only; shipped v0.1.7 builds it from every labelled pixel, test pixels included.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ghost.datasets.hyperspectral_dataset import HyperspectralDataset  # noqa: E402
from ghost.v0_2.splits import SPLIT_MODES, make_split  # noqa: E402

_STATE = {}


def make_init(original, mode, samples_per_class=50, minority_samples=15, block_size=None):
    """A HyperspectralDataset.__init__ that runs v0.1's own and then swaps in v0.2's split."""

    def init(self, data_path, gt_path, split='train', train_ratio=0.2, val_ratio=0.1,
             data_key=None, labels_key=None, seed=42, use_fp16=False):
        original(self, data_path, gt_path, split=split, train_ratio=train_ratio, val_ratio=val_ratio,
                 data_key=data_key, labels_key=labels_key, seed=seed, use_fp16=use_fp16)
        labels = np.ascontiguousarray(self.labels.numpy())
        W = labels.shape[1]
        flat = make_split(mode, labels, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio,
                          samples_per_class=samples_per_class, minority_samples=minority_samples,
                          block_size=block_size)
        self.train_coords, self.val_coords, self.test_coords = (np.stack([i // W, i % W], axis=1) for i in flat)
        self.coords = {'train': self.train_coords, 'val': self.val_coords, 'test': self.test_coords}[split]
        rows, cols = torch.from_numpy(self.coords[:, 0]), torch.from_numpy(self.coords[:, 1])
        self.split_mask = torch.zeros(self.labels.shape, dtype=torch.long)
        self.split_mask[rows, cols] = self.labels[rows, cols]
        _STATE['train_flat'] = flat[0]
        print(f"  v0.2 '{mode}' split in use → {split}: {len(self.coords)} pixels")

    return init


def make_tree_builder(original):
    """build_rssp_tree that only sees the labels of training pixels (flat indices, so array order can't bite)."""

    def build(data, labels, **kwargs):
        flat_labels = np.ascontiguousarray(labels).reshape(-1)
        masked = np.zeros_like(flat_labels)
        idx = _STATE['train_flat']
        masked[idx] = flat_labels[idx]
        return original(data, masked.reshape(np.shape(labels)), **kwargs)

    return build


def install(mode, samples_per_class=50, minority_samples=15, block_size=None, tree_from_train=False):
    HyperspectralDataset.__init__ = make_init(HyperspectralDataset.__init__, mode, samples_per_class,
                                              minority_samples, block_size)
    if tree_from_train:
        import ghost.train_rssp as train_rssp
        train_rssp.build_rssp_tree = make_tree_builder(train_rssp.build_rssp_tree)


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit("usage: v01_split.py --split MODE [options] -- (train|train_spt) [v0.1 flags]")
    cut = argv.index('--')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split', required=True, choices=SPLIT_MODES)
    p.add_argument('--samples_per_class', type=int, default=50)
    p.add_argument('--minority_samples', type=int, default=15)
    p.add_argument('--block_size', type=int, default=None)
    p.add_argument('--tree-from-train', dest='tree_from_train', action='store_true')
    args = p.parse_args(argv[:cut])
    command, rest = argv[cut + 1], argv[cut + 2:]

    install(args.split, args.samples_per_class, args.minority_samples, args.block_size, args.tree_from_train)
    if command == 'train':
        from ghost.train import main as run
    elif command in ('train_spt', 'train_rssp'):
        from ghost.train_rssp import main as run
    else:
        raise SystemExit(f"Unknown v0.1 command '{command}': use train or train_spt")
    sys.argv = [f'ghost {command}'] + rest
    run()


if __name__ == '__main__':
    main()
