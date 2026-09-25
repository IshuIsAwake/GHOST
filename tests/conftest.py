"""Shared fixtures: a synthetic scene whose classes differ only by where their absorption dip sits."""
import os
import sys

import numpy as np
import pytest
import scipy.io


def make_scene(H=24, W=20, B=64, n_classes=4, seed=0, unlabelled_frac=0.2, noise=0.01):
    """Sloped continuum, one Gaussian dip per class, random per-pixel brightness, class stripes."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, B)
    continuum = 0.3 + 0.6 * x
    centers = np.linspace(0.2, 0.8, n_classes)
    templates = np.stack([continuum * (1.0 - 0.5 * np.exp(-((x - c) / 0.05) ** 2)) for c in centers])
    classes = np.broadcast_to((np.arange(W) * n_classes) // W, (H, W))
    brightness = rng.uniform(0.5, 2.0, size=(H, W, 1))
    cube = templates[classes] * brightness + rng.normal(0.0, noise, size=(H, W, B))
    cube = np.clip(cube, 1e-3, None).astype(np.float32)
    gt = (classes + 1).astype(np.int64)
    gt[rng.random((H, W)) < unlabelled_frac] = 0
    return cube, gt


def write_mat(path, cube, gt=None, cube_key="data", gt_key="gt"):
    scipy.io.savemat(str(path), {cube_key: cube})
    if gt is not None:
        gt_path = str(path).replace(".mat", "_gt.mat")
        scipy.io.savemat(gt_path, {gt_key: gt})
        return str(path), gt_path
    return str(path)


def write_tif(path, array):
    """Band-major GeoTIFF; a 2-D array becomes a single band."""
    rasterio = pytest.importorskip("rasterio")
    arr = array[:, :, None] if array.ndim == 2 else array
    H, W, B = arr.shape
    with rasterio.open(str(path), "w", driver="GTiff", height=H, width=W, count=B,
                       dtype=arr.dtype.name) as dst:
        for b in range(B):
            dst.write(arr[:, :, b], b + 1)
    return str(path)


def write_h5(path, array, key="data", attrs=None):
    h5py = pytest.importorskip("h5py")
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset(key, data=array)
        for k, v in (attrs or {}).items():
            ds.attrs[k] = v
    return str(path)


def write_envi(path, array, wavelengths=None):
    envi = pytest.importorskip("spectral.io.envi")
    arr = array[:, :, None] if array.ndim == 2 else array
    meta = {"wavelength": [str(w) for w in wavelengths]} if wavelengths is not None else {}
    envi.save_image(str(path), arr, dtype=arr.dtype, force=True, interleave="bsq", metadata=meta)
    return str(path)


def write_mat_v73(path, array, key="data"):
    """MATLAB v7.3 is HDF5 with the axes reversed; scipy.io.loadmat refuses it."""
    h5py = pytest.importorskip("h5py")
    with h5py.File(str(path), "w", userblock_size=512) as f:
        f.create_dataset(key, data=np.ascontiguousarray(array.T))
    text = b"MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: test HDF5 schema 1.00 ."
    header = text.ljust(116, b" ") + b"\x00" * 8 + b"\x00\x02" + b"IM"
    with open(str(path), "r+b") as fh:
        fh.write(header)
    return str(path)


@pytest.fixture
def scene():
    return make_scene()


@pytest.fixture
def scene_files(tmp_path):
    cube, gt = make_scene(H=16, W=12, B=24, n_classes=3, seed=1)
    data_path, gt_path = write_mat(tmp_path / "scene.mat", cube, gt)
    return {"cube": cube, "gt": gt, "data": data_path, "gt_path": gt_path, "dir": tmp_path}


def run_train(out_dir, data, gt, *flags, epochs=3):
    """Train v0.2 in-process on CPU with tiny settings; returns the summary dict."""
    from ghost.v0_2.train import main
    argv = ["--data", data, "--gt", gt, "--out-dir", str(out_dir), "--epochs", str(epochs),
            "--min_epochs", "1", "--device", "cpu", "--batch_size", "32"] + [str(f) for f in flags]
    return main(argv)


def run_cli(*args, cwd=None, timeout=600):
    """Run `python -m ghost.cli` on CPU; returns the CompletedProcess."""
    import subprocess
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=os.getcwd())
    return subprocess.run([sys.executable, "-m", "ghost.cli", *map(str, args)], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=timeout)
