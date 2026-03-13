"""
HDF5 reader for generic planetary/multi-mission hyperspectral data.

What is HDF5?
    HDF5 (Hierarchical Data Format 5) is a container format — think of it
    like a ZIP file, but for numerical arrays. Inside one .h5 file you can
    have multiple arrays organized in a folder-like hierarchy.
    
    It's used by: ESA missions, some planetary datasets, EMIT (Earth Surface
    Mineral Dust Source Investigation), various university research datasets.

Requires: h5py (pip install h5py)

Output:
    data   — (C, H, W) float32 numpy array, normalized
    labels — (H, W) int64 (all zeros unless a 2D dataset is found)
    meta   — dict with dataset names and wavelengths if present
"""

import numpy as np
from pathlib import Path

try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False


def _find_datasets(hf) -> dict:
    """Walk HDF5 hierarchy and collect all datasets with their shapes."""
    datasets = {}

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets[name] = obj.shape

    hf.visititems(visitor)
    return datasets


def read(h5_path: str, lbl_path: str = None,
         data_key: str = None, labels_key: str = None) -> tuple:
    """
    Read an HDF5 hyperspectral file.

    Parameters
    ----------
    h5_path : str
        Path to the .h5 or .hdf5 file.
    lbl_path : str, optional
        Ignored if labels are embedded in the same HDF5 file.
    data_key : str, optional
        Explicit HDF5 dataset key for the spectral cube.
        If None, auto-detects the largest 3D dataset.
    labels_key : str, optional
        Explicit HDF5 dataset key for labels.
        If None, auto-detects any 2D integer dataset.

    Returns
    -------
    data   : (C, H, W) float32
    labels : (H, W) int64
    meta   : dict
    """
    if not _H5PY_AVAILABLE:
        raise ImportError(
            "h5py is required to read HDF5 files.\n"
            "Install it with:  pip install h5py"
        )

    h5_path = Path(h5_path)

    with h5py.File(str(h5_path), 'r') as hf:
        all_datasets = _find_datasets(hf)

        # ── Auto-detect data key ──────────────────────────────────────────────
        if data_key is None:
            # Find the largest 3D dataset
            candidates_3d = {k: v for k, v in all_datasets.items()
                             if len(v) == 3}
            if not candidates_3d:
                raise ValueError(
                    f"No 3D dataset found in {h5_path}.\n"
                    f"Available datasets: {list(all_datasets.keys())}\n"
                    f"Pass data_key='...' explicitly."
                )
            data_key = max(candidates_3d, key=lambda k: np.prod(candidates_3d[k]))

        # ── Auto-detect labels key ────────────────────────────────────────────
        if labels_key is None:
            candidates_2d = {k: v for k, v in all_datasets.items()
                             if len(v) == 2 and k != data_key}
            if candidates_2d:
                # Prefer keys containing 'label', 'gt', 'class', 'mask'
                for hint in ('label', 'gt', 'class', 'mask', 'ground'):
                    for k in candidates_2d:
                        if hint in k.lower():
                            labels_key = k
                            break
                    if labels_key:
                        break
                if labels_key is None:
                    labels_key = list(candidates_2d.keys())[0]

        # ── Load data ─────────────────────────────────────────────────────────
        raw = np.array(hf[data_key], dtype=np.float32)

        # ── Detect spectral axis ──────────────────────────────────────────────
        # Same logic as FITS reader: largest axis is likely spectral
        if raw.ndim == 3:
            if raw.shape[2] > raw.shape[0] and raw.shape[2] > raw.shape[1]:
                # (H, W, C) → (C, H, W)
                raw = raw.transpose(2, 0, 1)
            # else assume already (C, H, W) or (H, C, W) — we take first dim as C

        elif raw.ndim == 2:
            raw = raw[np.newaxis, :, :]  # (1, H, W)

        C, H, W = raw.shape

        # ── Load labels ───────────────────────────────────────────────────────
        if labels_key and labels_key in hf:
            labels = np.array(hf[labels_key], dtype=np.int64)
        else:
            labels = np.zeros((H, W), dtype=np.int64)

        # ── Try to find wavelength array ──────────────────────────────────────
        wavelengths = []
        for wname in ('wavelength', 'wavelengths', 'lambda', 'wave', 'Wavelength'):
            if wname in hf:
                wavelengths = list(np.array(hf[wname], dtype=float))
                break

        print(f"[HDF5] {h5_path.name}")
        print(f"  Data key:   '{data_key}'")
        print(f"  Dims:       {H} lines × {W} samples × {C} bands")
        if labels_key:
            print(f"  Labels key: '{labels_key}'")
        if wavelengths:
            print(f"  Wavelengths: {wavelengths[0]:.1f} – {wavelengths[-1]:.1f}")

    # ── Clean and normalize (Per-band Z-score) ────────────────────────────────
    invalid = (raw < -1e10) | (raw > 1e10) | np.isinf(raw) | np.isnan(raw)

    for b in range(C):
        band = raw[b]
        band_invalid = invalid[b]
        valid_pixels = band[~band_invalid]
        if valid_pixels.size > 0:
            mu  = valid_pixels.mean()
            std = valid_pixels.std()
            raw[b] = (band - mu) / (std + 1e-8)
        raw[b][band_invalid] = 0.0

    if invalid.any():
        print(f"  Invalid pixels: {invalid.sum()} — normalized and zeroed.")

    meta = {
        'bands': C, 'lines': H, 'line_samples': W,
        'data_key': data_key, 'labels_key': labels_key,
        'wavelengths': wavelengths, 'h5_path': str(h5_path),
        'all_datasets': list(all_datasets.keys()),
    }

    print(f"  Loaded:      cube {raw.shape}  labels {labels.shape}")
    return raw, labels, meta
