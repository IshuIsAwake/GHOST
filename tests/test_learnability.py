"""The pipeline learns a problem it should find easy, and cannot learn from shuffled labels."""
import numpy as np
import pytest

from conftest import make_scene, run_train, write_mat

pytestmark = pytest.mark.slow


def test_learns_absorption_positions(tmp_path):
    cube, gt = make_scene(H=40, W=40, B=64, n_classes=4, seed=5)
    data, gt_path = write_mat(tmp_path / "s.mat", cube, gt)
    summary = run_train(tmp_path / "run", data, gt_path, "--patience", 100, epochs=40)
    assert summary["cr_mode"] == "full"
    assert summary["test_metrics"]["OA"] >= 0.95


def test_shuffled_labels_stay_near_chance(tmp_path):
    cube, gt = make_scene(H=40, W=40, B=64, n_classes=4, seed=5)
    rng = np.random.default_rng(0)
    labelled = gt > 0
    shuffled = gt.copy()
    shuffled[labelled] = rng.permutation(gt[labelled])
    data, gt_path = write_mat(tmp_path / "s.mat", cube, shuffled)
    summary = run_train(tmp_path / "run", data, gt_path, "--patience", 100, epochs=40)
    assert summary["test_metrics"]["OA"] <= summary["majority_baseline"] + 0.15
