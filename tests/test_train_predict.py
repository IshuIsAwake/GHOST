"""Train and predict end to end on a tiny synthetic scene, on CPU."""
import csv
import json

import numpy as np
import pytest
import torch

from conftest import make_scene, run_train, write_mat

V01_COLUMNS = ["best_epoch", "test_oa", "test_miou", "test_dice", "test_precision",
               "test_recall", "test_aa", "test_kappa"]
METRIC_COLUMNS = V01_COLUMNS[1:]


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def _predict(*argv):
    from ghost.v0_2.predict import main
    return main([str(a) for a in argv] + ["--device", "cpu"])


def test_training_writes_its_outputs(scene_files, tmp_path):
    out = tmp_path / "run"
    summary = run_train(out, scene_files["data"], scene_files["gt_path"], epochs=3)
    for name in ("ghost_model.pt", "training_log.csv", "test_results.csv", "class_report.csv", "run_config.json"):
        assert (out / name).exists(), name
    assert len(_rows(out / "training_log.csv")) == summary["epochs_run"] + 1
    assert _rows(out / "test_results.csv")[0] == V01_COLUMNS
    config = json.loads((out / "run_config.json").read_text())
    assert config["arch"] == "0.2.0"
    assert config["cr_mode"] == "simple"  # 24 bands
    assert config["split_sizes"]["test"] == summary["split_sizes"]["test"] > 0
    assert 0 < config["majority_baseline"] < 1


def test_predict_on_the_training_scene_reproduces_the_test_score(scene_files, tmp_path):
    out = tmp_path / "run"
    summary = run_train(out, scene_files["data"], scene_files["gt_path"], epochs=3)
    result = _predict("--model", out / "ghost_model.pt", "--data", scene_files["data"],
                      "--gt", scene_files["gt_path"], "--out-dir", tmp_path / "pred")
    assert result["mode"] == "same_scene"
    assert result["n_eval"] == summary["split_sizes"]["test"]
    trained = dict(zip(*_rows(out / "test_results.csv")))
    predicted = dict(zip(*_rows(tmp_path / "pred" / "predict_results.csv")))
    assert {k: trained[k] for k in METRIC_COLUMNS} == {k: predicted[k] for k in METRIC_COLUMNS}
    assert result["metrics"]["OA"] == summary["test_metrics"]["OA"]


def test_predict_without_ground_truth_writes_a_map(tmp_path):
    cube, gt = make_scene(H=16, W=12, B=24, n_classes=3, seed=4)
    cube[0, 0, :] = 0.0  # a no-data pixel
    data, gt_path = write_mat(tmp_path / "s.mat", cube, gt)
    out = tmp_path / "run"
    run_train(out, data, gt_path, epochs=2)
    result = _predict("--model", out / "ghost_model.pt", "--data", data, "--out-dir", tmp_path / "pred")
    assert result["mode"] is None and result["metrics"] is None
    pred = np.load(tmp_path / "pred" / "prediction.npy")
    assert pred.shape == (16, 12) and pred.dtype == np.int32
    assert pred[0, 0] == 0
    assert set(np.unique(pred[1:])) <= {1, 2, 3}
    assert (tmp_path / "pred" / "prediction.png").exists()
    assert not (tmp_path / "pred" / "predict_results.csv").exists()


def test_predict_on_a_new_scene_scores_every_labelled_pixel(scene_files, tmp_path):
    out = tmp_path / "run"
    run_train(out, scene_files["data"], scene_files["gt_path"], epochs=2)
    cube = scene_files["cube"].copy()
    cube[3, 3, 3] *= 1.01
    other, _ = write_mat(tmp_path / "other.mat", cube, scene_files["gt"])
    result = _predict("--model", out / "ghost_model.pt", "--data", other, "--gt", scene_files["gt_path"],
                      "--out-dir", tmp_path / "pred")
    assert result["mode"] == "new_scene"
    assert result["n_eval"] == int((scene_files["gt"] > 0).sum())


def test_band_mismatch_is_a_clear_error(scene_files, tmp_path, capsys):
    out = tmp_path / "run"
    run_train(out, scene_files["data"], scene_files["gt_path"], epochs=1)
    other = write_mat(tmp_path / "narrow.mat", scene_files["cube"][:, :, :20])
    with pytest.raises(SystemExit) as exc:
        _predict("--model", out / "ghost_model.pt", "--data", other, "--out-dir", tmp_path / "pred")
    assert exc.value.code == 2
    message = capsys.readouterr()
    text = message.out + message.err
    assert "24" in text and "20" in text


def test_non_contiguous_labels_round_trip(tmp_path):
    cube, gt = make_scene(H=16, W=12, B=24, n_classes=3, seed=6)
    gt = np.choose(gt, [0, 2, 5, 9])
    data, gt_path = write_mat(tmp_path / "s.mat", cube, gt)
    out = tmp_path / "run"
    run_train(out, data, gt_path, epochs=2)
    ckpt = torch.load(out / "ghost_model.pt", weights_only=True)
    assert ckpt["class_ids"] == [2, 5, 9]
    _predict("--model", out / "ghost_model.pt", "--data", data, "--out-dir", tmp_path / "pred")
    assert set(np.unique(np.load(tmp_path / "pred" / "prediction.npy"))) <= {2, 5, 9}


def test_same_seed_is_reproducible(scene_files, tmp_path):
    a = run_train(tmp_path / "a", scene_files["data"], scene_files["gt_path"], "--seed", 3, epochs=3)
    b = run_train(tmp_path / "b", scene_files["data"], scene_files["gt_path"], "--seed", 3, epochs=3)
    assert a["test_metrics"] == b["test_metrics"]
    sa = torch.load(tmp_path / "a" / "ghost_model.pt", weights_only=True)["state_dict"]
    sb = torch.load(tmp_path / "b" / "ghost_model.pt", weights_only=True)["state_dict"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


@pytest.mark.parametrize("cr,resolved", [("auto", "simple"), ("full", "full"), ("simple", "simple"),
                                         ("off", "off"), ("none", "none")])
def test_every_cr_mode_trains(scene_files, tmp_path, cr, resolved):
    summary = run_train(tmp_path / cr, scene_files["data"], scene_files["gt_path"], "--cr", cr, epochs=1)
    assert summary["cr_mode"] == resolved
    ckpt = torch.load(tmp_path / cr / "ghost_model.pt", weights_only=True)
    assert ckpt["preprocessing"]["cr_mode"] == resolved


@pytest.mark.parametrize("split", ["ratio", "fixed", "disjoint"])
def test_every_split_trains(scene_files, tmp_path, split):
    summary = run_train(tmp_path / split, scene_files["data"], scene_files["gt_path"], "--split", split,
                        "--samples_per_class", 10, "--minority_samples", 5, epochs=1)
    assert summary["split_sizes"]["train"] > 0


def test_flatten_pool_trains(scene_files, tmp_path):
    summary = run_train(tmp_path / "flat", scene_files["data"], scene_files["gt_path"], "--pool", "flatten", epochs=1)
    assert 0.0 <= summary["test_metrics"]["OA"] <= 1.0
