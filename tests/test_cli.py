"""CLI: --arch picks the pipeline, checkpoints pick it for predict, v0.1 still runs untouched."""
import pytest

from conftest import make_scene, run_cli, run_train, write_mat, write_tif

pytestmark = pytest.mark.slow


@pytest.fixture
def tiny(tmp_path):
    cube, gt = make_scene(H=16, W=12, B=24, n_classes=3, seed=1)
    data, gt_path = write_mat(tmp_path / "s.mat", cube, gt)
    return data, gt_path


def _train_args(data, gt_path, out):
    return ["--data", data, "--gt", gt_path, "--out-dir", out, "--epochs", 2, "--min_epochs", 1,
            "--device", "cpu"]


def test_version_lists_architectures():
    r = run_cli("version")
    assert r.returncode == 0
    assert "0.1.7" in r.stdout and "0.2.0 (default)" in r.stdout


@pytest.mark.parametrize("arch", [[], ["--arch", "0.2.0"], ["--arch", "v0.2.0"], ["--arch=0.2.0"]])
def test_train_runs_v02(tiny, tmp_path, arch):
    out = tmp_path / "run"
    r = run_cli("train", *_train_args(*tiny, out), *arch)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "ghost_model.pt").exists()


@pytest.mark.parametrize("arch", ["9.9.9", "0.2"])
def test_unknown_arch_is_rejected(tiny, tmp_path, arch):
    r = run_cli("train", "--arch", arch, *_train_args(*tiny, tmp_path / "run"))
    assert r.returncode != 0
    text = r.stdout + r.stderr
    assert "0.1.7" in text and "0.2.0" in text


def test_train_spt_is_v01_only(tiny):
    r = run_cli("train_spt", "--arch", "0.2.0", "--data", tiny[0], "--gt", tiny[1])
    assert r.returncode != 0
    assert "0.1.7" in r.stdout + r.stderr


def test_predict_follows_the_checkpoint(tiny, tmp_path):
    run_train(tmp_path / "run", *tiny, epochs=1)
    model = tmp_path / "run" / "ghost_model.pt"
    r = run_cli("predict", "--model", model, "--data", tiny[0], "--out-dir", tmp_path / "pred", "--device", "cpu")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "pred" / "prediction.npy").exists()
    r = run_cli("predict", "--model", model, "--data", tiny[0], "--arch", "0.1.7")
    assert r.returncode != 0
    assert "0.2.0" in r.stdout + r.stderr


def test_visualize_follows_the_checkpoint(tiny, tmp_path):
    run_train(tmp_path / "run", *tiny, epochs=1)
    r = run_cli("visualize", "--model", tmp_path / "run" / "ghost_model.pt", "--data", tiny[0],
                "--gt", tiny[1], "--out-dir", tmp_path / "vis", "--device", "cpu")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "vis" / "segmentation.png").exists()


def test_v01_trainer_still_runs(tmp_path):
    cube, gt = make_scene(H=32, W=32, B=16, n_classes=3, seed=2)
    data, gt_path = write_mat(tmp_path / "s.mat", cube, gt)
    out = tmp_path / "v01"
    r = run_cli("train", "--arch", "0.1.7", "--data", data, "--gt", gt_path, "--epochs", 10, "--out-dir", out)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "best_model.pth").exists() and (out / "test_results.csv").exists()


def test_help_follows_the_arch():
    assert "--cr" in run_cli("train", "--help").stdout
    assert "--base_filters" in run_cli("train", "--arch", "0.1.7", "--help").stdout


def test_demo_prints_a_v02_command():
    r = run_cli("demo")
    assert r.returncode == 0
    assert "ghost train" in r.stdout and "Indian_pines_corrected.mat" in r.stdout


def test_convert_to_mat_still_works(tmp_path):
    cube, _ = make_scene(H=6, W=5, B=4)
    src = write_tif(tmp_path / "c.tif", cube)
    r = run_cli("convert_to_mat", "--img", src, "--out-dir", tmp_path / "out")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "out" / "data.mat").exists()
