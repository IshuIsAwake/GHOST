"""Continuum removal on raw spectra, computed once per scene before any normalisation."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import savgol_filter

EPS = 1e-6
CR_MODES = ('auto', 'full', 'simple', 'off', 'none')
FULL_MIN_BANDS = 64
SIMPLE_MIN_BANDS = 3


def resolve_cr_mode(mode: str, num_bands: int) -> str:
    """'auto' picks full at ≥64 bands, simple at 3–63, off below 3; other modes pass through."""
    if mode not in CR_MODES:
        raise ValueError(f"Unknown continuum removal mode '{mode}'. Choose from: {', '.join(CR_MODES)}")
    if mode != 'auto':
        return mode
    if num_bands >= FULL_MIN_BANDS:
        return 'full'
    if num_bands >= SIMPLE_MIN_BANDS:
        return 'simple'
    return 'off'


def savgol_window(num_bands: int) -> int | None:
    """Odd window of about 5% of the bands (at least 5); None when the spectrum is too short to smooth."""
    if num_bands < 5:
        return None
    window = min(max(5, int(0.05 * num_bands)), num_bands)
    return window if window % 2 else window - 1


def upper_hull(x: np.ndarray, y: np.ndarray) -> tuple[list, list]:
    """Andrew's monotone chain, upper half only; x must be increasing. Collinear points are dropped."""
    hx, hy = [], []
    for px, py in zip(x.tolist(), y.tolist()):
        while len(hx) >= 2 and (hx[-1] - hx[-2]) * (py - hy[-2]) - (hy[-1] - hy[-2]) * (px - hx[-2]) >= 0:
            hx.pop()
            hy.pop()
        hx.append(px)
        hy.append(py)
    return hx, hy


def _axis(x, num_bands: int) -> np.ndarray:
    if x is None:
        return np.arange(num_bands, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (num_bands,) or not (np.diff(x) > 0).all():
        raise ValueError(f"x must hold {num_bands} strictly increasing values")
    return x


def continuum_removal(spectra, mode: str, x=None) -> np.ndarray:
    """(N, B) raw spectra → (N, B) float32 in (0, 1]. Values below EPS are clamped, with a warning.

    full: divide by max(upper hull of the Savitzky-Golay-smoothed spectrum, raw spectrum).
    simple: divide by the line from the first to the last band, then by the pixel's maximum.
    off: divide by the pixel's maximum. none: return the spectra unchanged.
    """
    spectra = np.asarray(spectra, dtype=np.float64)
    if spectra.ndim != 2:
        raise ValueError(f"Expected (N, B) spectra, got shape {spectra.shape}")
    if mode == 'none':
        return spectra.astype(np.float32)
    if mode not in ('full', 'simple', 'off'):
        raise ValueError(f"Unknown continuum removal mode '{mode}'")

    low = spectra < EPS
    if low.any():
        warnings.warn(f"{int(low.sum())} values in {int(low.any(axis=1).sum())} pixels were below {EPS:g} "
                      f"and were clamped to it before continuum removal", UserWarning, stacklevel=2)
        spectra = np.maximum(spectra, EPS)
    N, B = spectra.shape
    xs = _axis(x, B)

    if mode == 'off' or B == 1:
        out = spectra / spectra.max(axis=1, keepdims=True)
    elif mode == 'simple':
        slope = (spectra[:, -1:] - spectra[:, :1]) / (xs[-1] - xs[0])
        line = np.maximum(spectra[:, :1] + slope * (xs - xs[0]), EPS)
        out = spectra / line
        out = out / out.max(axis=1, keepdims=True)
    else:
        window = savgol_window(B)
        smoothed = savgol_filter(spectra, window, 2, axis=1) if window else spectra
        out = np.empty_like(spectra)
        for i in range(N):
            hx, hy = upper_hull(xs, smoothed[i])
            envelope = np.maximum(np.interp(xs, hx, hy), spectra[i])
            out[i] = spectra[i] / envelope
    return out.astype(np.float32)


def validity_mask(cube: np.ndarray, nodata=None) -> np.ndarray:
    """(H, W) True for usable pixels; any non-finite band, an all-zero spectrum or all-nodata is invalid."""
    valid = np.isfinite(cube).all(axis=-1) & (cube != 0).any(axis=-1)
    if nodata is not None and np.isfinite(nodata):
        valid &= ~(cube == nodata).all(axis=-1)
    return valid


def zscore_stats(cube: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    """Scene-wide mean and std over valid pixels (v0.1's normalisation, kept for the 'none' ablation)."""
    values = cube[valid].astype(np.float64)
    return float(values.mean()), float(values.std())


def build_settings(cr: str, cube: np.ndarray, valid: np.ndarray, wavelengths=None) -> dict:
    """Everything predict needs to repeat training's preprocessing, as plain Python values."""
    num_bands = cube.shape[-1]
    mode = resolve_cr_mode(cr, num_bands)
    settings = {
        'cr_requested': cr,
        'cr_mode': mode,
        'eps': EPS,
        'savgol_window': savgol_window(num_bands) if mode == 'full' else None,
        'savgol_polyorder': 2,
        'x_axis': 'wavelength' if wavelengths is not None else 'index',
        'wavelengths': [float(w) for w in wavelengths] if wavelengths is not None else None,
        'zscore_mean': None,
        'zscore_std': None,
    }
    if mode == 'none':
        settings['zscore_mean'], settings['zscore_std'] = zscore_stats(cube, valid)
    return settings


def preprocess(cube: np.ndarray, valid: np.ndarray, settings: dict) -> np.ndarray:
    """(H, W, B) cube → (H·W, B) float32 features; invalid pixels stay all-zero rows."""
    H, W, B = cube.shape
    flat_valid = valid.reshape(-1)
    feats = np.zeros((H * W, B), dtype=np.float32)
    spectra = cube.reshape(-1, B)[flat_valid]
    if settings['cr_mode'] == 'none':
        spectra = (spectra.astype(np.float64) - settings['zscore_mean']) / (settings['zscore_std'] + 1e-8)
        feats[flat_valid] = spectra.astype(np.float32)
    else:
        feats[flat_valid] = continuum_removal(spectra, settings['cr_mode'], x=settings['wavelengths'])
    return feats
