"""Universal loader: every format reaches the pipeline as (H, W, Bands) with its values intact."""
import sys

import numpy as np
import pytest
import scipy.io

from conftest import make_scene, write_envi, write_h5, write_mat, write_mat_v73, write_tif

H, W, B = 8, 5, 3


@pytest.fixture
def cube():
    rng = np.random.default_rng(0)
    return rng.uniform(0.1, 1.0, size=(H, W, B)).astype(np.float32)


@pytest.fixture
def labels():
    return (np.arange(H * W).reshape(H, W) % 4).astype(np.int32)


def _hide(monkeypatch, dependency):
    for name in [m for m in list(sys.modules) if m == dependency or m.startswith(dependency + ".")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, dependency, None)


@pytest.mark.parametrize("fmt", ["mat", "tif", "h5", "envi", "mat_v73"])
def test_cube_loads_as_hwb_with_source_values(tmp_path, cube, fmt):
    from ghost.datasets.loader import load_cube
    writers = {
        "mat": lambda: write_mat(tmp_path / "c.mat", cube),
        "tif": lambda: write_tif(tmp_path / "c.tif", cube),
        "h5": lambda: write_h5(tmp_path / "c.h5", cube),
        "envi": lambda: write_envi(tmp_path / "c.hdr", cube),
        "mat_v73": lambda: write_mat_v73(tmp_path / "c73.mat", cube),
    }
    data, meta = load_cube(writers[fmt]())
    assert data.shape == (H, W, B)
    assert data.dtype == np.float32
    np.testing.assert_allclose(data, cube, rtol=1e-6)


@pytest.mark.parametrize("fmt", ["mat", "tif", "h5", "envi", "mat_v73"])
def test_labels_load_as_hw(tmp_path, labels, fmt):
    from ghost.datasets.loader import load_labels
    def savemat():
        scipy.io.savemat(str(tmp_path / "g.mat"), {"gt": labels})
        return str(tmp_path / "g.mat")

    writers = {
        "mat": savemat,
        "tif": lambda: write_tif(tmp_path / "g.tif", labels),
        "h5": lambda: write_h5(tmp_path / "g.h5", labels, key="gt"),
        "envi": lambda: write_envi(tmp_path / "g.hdr", labels),
        "mat_v73": lambda: write_mat_v73(tmp_path / "g73.mat", labels, key="gt"),
    }
    gt, _ = load_labels(writers[fmt]())
    gt = np.squeeze(gt)
    assert gt.shape == (H, W)
    np.testing.assert_array_equal(gt.astype(np.int64), labels.astype(np.int64))


def test_envi_wavelengths_reach_meta(tmp_path, cube):
    from ghost.datasets.loader import load_cube
    wl = [450.0, 550.0, 900.0]
    _, meta = load_cube(write_envi(tmp_path / "w.hdr", cube, wavelengths=wl))
    np.testing.assert_allclose(meta["wavelengths"], wl)


def test_hdf5_wavelength_attribute_reaches_meta(tmp_path, cube):
    from ghost.datasets.loader import load_cube
    _, meta = load_cube(write_h5(tmp_path / "w.h5", cube, attrs={"wavelength": [1.0, 2.0, 4.0]}))
    np.testing.assert_allclose(meta["wavelengths"], [1.0, 2.0, 4.0])


def test_unusable_wavelengths_are_dropped(tmp_path, cube):
    from ghost.datasets.loader import load_cube
    _, meta = load_cube(write_h5(tmp_path / "w.h5", cube, attrs={"wavelength": [3.0, 2.0, 1.0]}))
    assert meta.get("wavelengths") is None


@pytest.mark.parametrize("dependency,fmt", [("rasterio", "tif"), ("h5py", "h5"), ("spectral", "envi")])
def test_missing_optional_dependency_raises_import_error(tmp_path, cube, monkeypatch, dependency, fmt):
    from ghost.datasets.loader import load_cube
    path = {"tif": lambda: write_tif(tmp_path / "c.tif", cube),
            "h5": lambda: write_h5(tmp_path / "c.h5", cube),
            "envi": lambda: write_envi(tmp_path / "c.hdr", cube)}[fmt]()
    _hide(monkeypatch, dependency)
    with pytest.raises(ImportError, match=dependency):
        load_cube(path)


def test_unknown_extension_raises_value_error(tmp_path):
    from ghost.datasets.loader import load_cube
    junk = tmp_path / "scene.xyz"
    junk.write_bytes(b"not a cube")
    with pytest.raises(ValueError):
        load_cube(str(junk))


def test_two_dimensional_cube_raises_value_error(tmp_path, labels):
    from ghost.datasets.loader import load_cube
    path = tmp_path / "flat.mat"
    scipy.io.savemat(str(path), {"only": labels.astype(np.float32)})
    with pytest.raises(ValueError):
        load_cube(str(path))


def test_convert_to_mat_still_converts_a_tif(tmp_path, cube, monkeypatch):
    from ghost import convert
    src = write_tif(tmp_path / "c.tif", cube)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["ghost convert_to_mat", "--img", src, "--out-dir", str(out)])
    convert.main()
    data = scipy.io.loadmat(str(out / "data.mat"))["data"]
    np.testing.assert_allclose(data, cube, rtol=1e-6)
    assert (out / "metadata.json").exists()


def test_convert_to_mat_exits_with_install_hint(tmp_path, cube, monkeypatch, capsys):
    from ghost import convert
    src = write_tif(tmp_path / "c.tif", cube)
    _hide(monkeypatch, "rasterio")
    monkeypatch.setattr(sys, "argv", ["ghost convert_to_mat", "--img", src, "--out-dir", str(tmp_path / "o")])
    with pytest.raises(SystemExit) as exc:
        convert.main()
    assert exc.value.code == 1
    assert "pip install rasterio" in capsys.readouterr().out


def test_synthetic_scene_round_trips_through_mat(tmp_path):
    from ghost.datasets.loader import load_cube, load_labels
    cube, gt = make_scene(H=6, W=7, B=10)
    data_path, gt_path = write_mat(tmp_path / "s.mat", cube, gt)
    np.testing.assert_allclose(load_cube(data_path)[0], cube)
    np.testing.assert_array_equal(np.squeeze(load_labels(gt_path)[0]), gt)
