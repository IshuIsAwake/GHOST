"""Continuum removal: runs once on raw spectra, lands in (0, 1], keeps absorption dips."""
import time
import warnings

import numpy as np
import pytest

from ghost.v0_2.preprocessing import (build_settings, continuum_removal, preprocess,
                                      resolve_cr_mode, validity_mask, zscore_stats)

MODES = ("full", "simple", "off")


def gaussian(B, center, sigma):
    i = np.arange(B)
    return np.exp(-0.5 * ((i - center) / sigma) ** 2)


@pytest.mark.parametrize("bands,expected", [(200, "full"), (64, "full"), (63, "simple"),
                                            (3, "simple"), (2, "off"), (1, "off")])
def test_auto_mode_follows_band_count(bands, expected):
    assert resolve_cr_mode("auto", bands) == expected


def test_explicit_mode_overrides_auto():
    assert resolve_cr_mode("simple", 200) == "simple"
    assert resolve_cr_mode("none", 5) == "none"


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        resolve_cr_mode("convex", 200)


def test_full_stays_in_unit_interval_on_noisy_spectra():
    # The hull comes from the smoothed spectrum, so noise peaks rise above it; the envelope guard caps them at 1.
    rng = np.random.default_rng(0)
    base = 0.5 + 0.3 * np.sin(np.linspace(0, 6, 200))
    spectra = base * rng.uniform(0.5, 2.0, (100, 1)) + rng.normal(0, 0.02, (100, 200))
    out = continuum_removal(np.clip(spectra, 0.05, None), "full")
    assert np.isfinite(out).all()
    assert out.min() > 0
    assert out.max() <= 1.0 + 1e-6
    assert out.max(axis=1).min() >= 0.85


def test_full_preserves_a_wide_dip():
    s = 1.0 - 0.4 * gaussian(200, 100, 6)
    out = continuum_removal(s[None], "full")[0]
    assert 0.55 <= out.min() <= 0.65
    np.testing.assert_allclose(out[:60], 1.0, atol=0.02)
    np.testing.assert_allclose(out[140:], 1.0, atol=0.02)


def test_full_flattens_a_linear_ramp():
    out = continuum_removal(np.linspace(0.2, 1.0, 200)[None], "full")
    np.testing.assert_allclose(out, 1.0, atol=1e-5)


def test_full_interior_peak_touches_one():
    s = 0.5 + 0.4 * gaussian(200, 100, 10)
    out = continuum_removal(s[None], "full")[0]
    assert out[100] == pytest.approx(1.0, abs=1e-6)
    assert out.max() <= 1.0 + 1e-6


@pytest.mark.parametrize("mode", MODES)
def test_constant_spectrum_gives_ones(mode):
    np.testing.assert_allclose(continuum_removal(np.full((3, 50), 0.7), mode), 1.0, atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_brightness_is_removed(mode):
    s = (0.4 + 0.5 * gaussian(80, 30, 8) + 0.2 * np.linspace(0, 1, 80))[None]
    np.testing.assert_allclose(continuum_removal(3.7 * s, mode), continuum_removal(s, mode), rtol=1e-5)


def test_simple_straight_line_gives_ones():
    np.testing.assert_allclose(continuum_removal(np.linspace(0.3, 0.9, 40)[None], "simple"), 1.0, atol=1e-6)


def test_simple_preserves_a_dip_under_the_chord():
    line = np.linspace(0.3, 0.9, 40)
    out = continuum_removal((line * (1.0 - 0.3 * gaussian(40, 20, 3)))[None], "simple")[0]
    assert out.min() == pytest.approx(0.7, abs=0.01)
    assert out.max() <= 1.0 + 1e-6


def test_simple_bulge_above_the_chord_is_rescaled():
    line = np.linspace(0.3, 0.9, 40)
    out = continuum_removal((line * (1.0 + 0.5 * gaussian(40, 20, 3)))[None], "simple")[0]
    assert out.max() == pytest.approx(1.0, abs=1e-6)


def test_off_is_raw_over_max():
    s = np.array([[0.2, 0.4, 0.8, 0.4]])
    np.testing.assert_allclose(continuum_removal(s, "off"), s / 0.8, rtol=1e-6)


def test_none_is_identity():
    s = np.array([[0.2, -0.4, 3.0]])
    np.testing.assert_allclose(continuum_removal(s, "none"), s)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name,spectrum", [
    ("zeros", np.zeros(30)),
    ("negatives", -np.linspace(0.1, 1.0, 30)),
    ("mixed_sign", np.linspace(-0.5, 0.5, 30)),
])
def test_degenerate_spectra_stay_finite_and_warn(mode, name, spectrum):
    with pytest.warns(UserWarning, match="clamped"):
        out = continuum_removal(spectrum[None], mode)
    assert np.isfinite(out).all()
    assert out.max() <= 1.0 + 1e-6


@pytest.mark.parametrize("mode", MODES)
def test_single_band_gives_one(mode):
    np.testing.assert_allclose(continuum_removal(np.array([[0.4], [2.0]]), mode), 1.0)


def test_clean_input_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        continuum_removal(np.linspace(0.2, 1.0, 64)[None], "full")


def test_wavelength_axis_changes_the_hull():
    s = np.array([[1.0, 0.8, 0.2]])
    assert continuum_removal(s, "full")[0, 1] == pytest.approx(1.0)
    assert continuum_removal(s, "full", x=[0.0, 1.0, 10.0])[0, 1] == pytest.approx(0.8 / 0.92, rel=1e-5)


def test_validity_mask_flags_nan_zero_and_nodata_pixels():
    cube = np.full((2, 3, 4), 0.5, dtype=np.float32)
    cube[0, 0, 2] = np.nan
    cube[0, 1] = 0.0
    cube[0, 2] = -9999.0
    cube[1, 0, 1] = 0.0  # one zero band is still a valid pixel
    valid = validity_mask(cube, nodata=-9999.0)
    np.testing.assert_array_equal(valid, [[False, False, False], [True, True, True]])


def test_zscore_stats_ignore_invalid_pixels():
    cube = np.ones((2, 2, 3), dtype=np.float32)
    cube[0, 0] = 1e6
    valid = np.array([[False, True], [True, True]])
    mean, std = zscore_stats(cube, valid)
    assert mean == pytest.approx(1.0)


def test_none_mode_uses_the_training_scene_statistics():
    rng = np.random.default_rng(0)
    train = rng.uniform(0, 10, (4, 4, 6)).astype(np.float32)
    valid = np.ones((4, 4), dtype=bool)
    settings = build_settings("none", train, valid)
    other = train + 100.0
    feats = preprocess(other, valid, settings)
    expected = (other.reshape(-1, 6) - settings["zscore_mean"]) / settings["zscore_std"]
    np.testing.assert_allclose(feats, expected, rtol=1e-5)


def test_preprocess_resolves_auto_and_zeroes_invalid_rows():
    cube = np.random.default_rng(1).uniform(0.1, 1.0, (3, 3, 70)).astype(np.float32)
    valid = np.ones((3, 3), dtype=bool)
    valid[1, 1] = False
    settings = build_settings("auto", cube, valid)
    assert settings["cr_mode"] == "full"
    feats = preprocess(cube, valid, settings)
    assert feats.shape == (9, 70) and feats.dtype == np.float32
    assert not feats[4].any()
    assert (feats[valid.reshape(-1)] > 0).all()


def test_settings_record_wavelength_axis():
    cube = np.random.default_rng(2).uniform(0.1, 1.0, (2, 2, 3)).astype(np.float32)
    valid = np.ones((2, 2), dtype=bool)
    assert build_settings("full", cube, valid)["x_axis"] == "index"
    s = build_settings("full", cube, valid, wavelengths=[400.0, 500.0, 900.0])
    assert s["x_axis"] == "wavelength" and s["wavelengths"] == [400.0, 500.0, 900.0]


@pytest.mark.slow
def test_full_mode_is_fast_enough():
    spectra = np.random.default_rng(0).uniform(0.1, 1.0, (5000, 200))
    t0 = time.time()
    continuum_removal(spectra, "full")
    assert time.time() - t0 < 15.0
