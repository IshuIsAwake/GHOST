"""The benchmark shim runs v0.1's own trainers on v0.2's split without touching v0.1's code."""
import contextlib
import importlib.util
import io
import os
import sys

import numpy as np
import pytest

from conftest import make_scene, write_mat

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def shim():
    spec = importlib.util.spec_from_file_location("v01_split", os.path.join(REPO, "benchmarks", "v01_split.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _datasets(path, gt_path, seed=0):
    from ghost.datasets.hyperspectral_dataset import HyperspectralDataset
    with contextlib.redirect_stdout(io.StringIO()):
        return {s: HyperspectralDataset(path, gt_path, split=s, seed=seed) for s in ("train", "val", "test")}


def test_ratio_mode_changes_nothing(shim, tmp_path, monkeypatch):
    from ghost.datasets.hyperspectral_dataset import HyperspectralDataset
    cube, gt = make_scene(H=20, W=16, B=10, n_classes=4, seed=3)
    data, gt_path = write_mat(tmp_path / "s.mat", cube, gt)
    before = _datasets(data, gt_path)
    monkeypatch.setattr(HyperspectralDataset, "__init__", shim.make_init(HyperspectralDataset.__init__, "ratio"))
    after = _datasets(data, gt_path)
    for s in before:
        np.testing.assert_array_equal(before[s].coords, after[s].coords)
        assert np.array_equal(before[s].split_mask.numpy(), after[s].split_mask.numpy())


def test_fixed_mode_gives_the_literature_split_on_indian_pines(shim, monkeypatch):
    from ghost.data import indian_pines_path
    from ghost.datasets.hyperspectral_dataset import HyperspectralDataset
    monkeypatch.setattr(HyperspectralDataset, "__init__", shim.make_init(HyperspectralDataset.__init__, "fixed"))
    ds = _datasets(*indian_pines_path())
    assert [len(ds[s].coords) for s in ("train", "val", "test")] == [695, 950, 8604]
    assert int((ds["train"].split_mask > 0).sum()) == 695


def test_tree_builder_sees_only_training_labels(shim):
    # v0.1's labels come out of scipy in column-major order; writing through a reshape of them silently fails.
    labels = np.asfortranarray(np.arange(1, 13).reshape(3, 4) % 3 + 1)
    shim._STATE["train_flat"] = np.array([0, 5, 11])
    seen = {}
    shim.make_tree_builder(lambda data, lbl, **kw: seen.setdefault("labels", lbl))(None, labels, num_classes=4)
    expected = np.zeros(12, dtype=labels.dtype)
    expected[[0, 5, 11]] = np.ascontiguousarray(labels).reshape(-1)[[0, 5, 11]]
    np.testing.assert_array_equal(seen["labels"].reshape(-1), expected)


@pytest.mark.slow
def test_v01_trainer_runs_on_the_fixed_split(tmp_path):
    import subprocess
    cube, gt = make_scene(H=32, W=32, B=16, n_classes=3, seed=2)
    data, gt_path = write_mat(tmp_path / "s.mat", cube, gt)
    out = tmp_path / "out"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, os.path.join(REPO, "benchmarks", "v01_split.py"), "--split", "fixed",
                        "--samples_per_class", "20", "--minority_samples", "5", "--", "train", "--data", data,
                        "--gt", gt_path, "--epochs", "20", "--lr", "1e-3", "--out-dir", str(out)],
                       capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "v0.2 'fixed' split in use → train: 60 pixels" in r.stdout
    assert (out / "test_results.csv").exists()
