"""Load a scene once: the raw cube, which pixels are usable, checked labels and content fingerprints."""
from __future__ import annotations

import os

import numpy as np

from ghost.datasets.loader import load_cube, load_labels
from ghost.v0_2.checkpoint import fingerprint
from ghost.v0_2.preprocessing import validity_mask


def _check_labels(raw, hw: tuple, path: str) -> np.ndarray:
    gt = np.squeeze(np.asarray(raw))
    if gt.ndim != 2:
        raise ValueError(f"Ground truth in {path} must be a 2-D label map, got shape {gt.shape} "
                         f"(an RGB image has to be mapped to class ids first)")
    if gt.shape != tuple(hw):
        raise ValueError(f"Spatial mismatch: the data is {tuple(hw)} but the ground truth in {path} is {gt.shape}")
    if np.issubdtype(gt.dtype, np.floating) and not np.array_equal(gt, np.round(gt)):
        raise ValueError(f"Ground truth in {path} holds non-integer values")
    gt = gt.astype(np.int64)
    if (gt < 0).any():
        raise ValueError(f"Ground truth in {path} holds negative labels")
    return gt


def load_scene(data_path: str, gt_path: str | None = None) -> dict:
    """Invalid pixels (NaN, all-zero, no-data) are unlabelled in the returned ground truth."""
    cube, meta = load_cube(data_path)
    valid = validity_mask(cube, meta.get('nodata'))
    if not valid.any():
        raise ValueError(f"{data_path} has no usable pixels")
    scene = {'cube': cube, 'meta': meta, 'valid': valid, 'fingerprint': fingerprint(cube),
             'wavelengths': meta.get('wavelengths'), 'data_path': os.path.abspath(data_path),
             'gt': None, 'label_fingerprint': None, 'invalid_labelled': 0}
    if gt_path is not None:
        gt = _check_labels(load_labels(gt_path)[0], cube.shape[:2], gt_path)
        scene['label_fingerprint'] = fingerprint(gt)
        scene['invalid_labelled'] = int(((gt > 0) & ~valid).sum())
        gt[~valid] = 0
        scene['gt'] = gt
    return scene


def class_ids_of(gt: np.ndarray) -> list:
    return [int(c) for c in np.unique(gt[gt > 0])]


def remap_labels(gt: np.ndarray, class_ids: list) -> np.ndarray:
    """Original ids → 0..K−1 in class_ids order; unlabelled pixels become −1."""
    lookup = np.full(max(int(gt.max()), max(class_ids)) + 1, -1, dtype=np.int64)
    lookup[class_ids] = np.arange(len(class_ids))
    return lookup[gt]
