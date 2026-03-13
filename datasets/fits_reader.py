"""
FITS reader for Hayabusa2 NIRS3 (Ryugu) and astronomical hyperspectral data.

What is FITS?
    FITS (Flexible Image Transport System) is the standard file format for
    astronomy. Every telescope, space probe spectrometer, and astronomical
    instrument uses FITS. It's been around since 1981.
    
    A FITS file is a single file containing:
      - One or more HDUs (Header Data Units)
      - Each HDU has a header (key=value metadata) + a data array
    
    The Hayabusa2 NIRS3 instrument measured near-infrared spectra of the
    Ryugu asteroid surface. Each FITS file contains a spectral data cube.

Requires: astropy (pip install astropy)

Output:
    data   — (C, H, W) float32 numpy array, normalized
    labels — (H, W) zeros (Ryugu data has no labelled ground truth)
    meta   — dict with wavelengths and FITS header info
"""

import numpy as np
from pathlib import Path

try:
    from astropy.io import fits as astropy_fits
    _ASTROPY_AVAILABLE = True
except ImportError:
    _ASTROPY_AVAILABLE = False


def read(fits_path: str, lbl_path: str = None) -> tuple:
    """
    Read a FITS hyperspectral file.

    Parameters
    ----------
    fits_path : str
        Path to the .fits or .fit file.
    lbl_path : str, optional
        Ignored for FITS — Ryugu data has no separate labels file.

    Returns
    -------
    data   : (C, H, W) float32
    labels : (H, W) int64 (all zeros)
    meta   : dict with wavelengths and header info
    """
    if not _ASTROPY_AVAILABLE:
        raise ImportError(
            "astropy is required to read FITS files.\n"
            "Install it with:  pip install astropy"
        )

    fits_path = Path(fits_path)

    with astropy_fits.open(str(fits_path), memmap=True) as hdul:
        # Find the primary data HDU and any wavelength extension
        data_hdu  = None
        wave_data = None

        for hdu in hdul:
            if hdu.data is None:
                continue
            if data_hdu is None and hdu.data.ndim >= 2:
                data_hdu = hdu
            # Check for wavelength table
            if hasattr(hdu, 'columns') and hdu.data is not None:
                for col in hdu.columns.names:
                    if 'WAVE' in col.upper() or 'LAM' in col.upper():
                        wave_data = hdu.data[col]

        if data_hdu is None:
            raise ValueError(f"No image data found in FITS file: {fits_path}")

        raw    = np.array(data_hdu.data, dtype=np.float32)
        header = dict(data_hdu.header)

    # ── Detect shape ─────────────────────────────────────────────────────────
    # NIRS3 FITS files are typically stored as (C, H, W) or (H, W, C)
    # FITS NAXIS convention: NAXIS1=fastest, NAXIS3=slowest
    # So a cube stored as (H, W, C) in numpy comes out as NAXIS1=C, NAXIS3=H
    # We need to detect which axis is the spectral axis.

    if raw.ndim == 2:
        # Single spectrum or spectral image — treat as (C, 1, W)
        raw = raw[np.newaxis, :, :]        # (1, H, W)

    if raw.ndim == 3:
        # Determine spectral axis: assume it's the axis with the most elements
        # for a typical CRISM/NIRS3 file this is unambiguous
        shape = raw.shape
        # FITS NAXIS1 = last numpy axis. If NAXIS3 >> NAXIS1 and NAXIS2, spectral is dim 0
        # If NAXIS1 >> others, spectral is dim 2
        # General rule: smallest spatial dims → shape should be H×W, largest → C
        # For NIRS3: typically (C=128, H=varies, W=varies) or (H, W, C)
        if header.get('NAXIS1', 0) > 100 and header.get('NAXIS3', 1) > 100:
            # Ambiguous — go by header CTYPE if present
            ctype1 = str(header.get('CTYPE1', '')).upper()
            if 'WAVE' in ctype1 or 'SPEC' in ctype1:
                # NAXIS1 is spectral → shape is (H, W, C) → transpose
                raw = raw.transpose(2, 0, 1)   # (C, H, W)
            # else already (C, H, W)
        elif shape[2] > shape[0] and shape[2] > shape[1]:
            # Last axis is largest → spectral last → (H, W, C)
            raw = raw.transpose(2, 0, 1)       # (C, H, W)
        # else assume already (C, H, W)

    C, H, W = raw.shape

    # ── Recover wavelengths from header (CRVAL1/CDELT1 convention) ──────────
    wavelengths = []
    if wave_data is not None:
        wavelengths = list(wave_data.astype(float))
    elif 'CRVAL1' in header and 'CDELT1' in header:
        crval = float(header['CRVAL1'])
        cdelt = float(header['CDELT1'])
        naxis = int(header.get('NAXIS1', C))
        wavelengths = [crval + i * cdelt for i in range(naxis)]

    print(f"[FITS] {fits_path.name}")
    print(f"  Dims:       {H} lines × {W} samples × {C} bands")
    if wavelengths:
        print(f"  Wavelengths: {wavelengths[0]:.1f} – {wavelengths[-1]:.1f} nm")

    # ── Clean up ─────────────────────────────────────────────────────────────
    invalid = (raw < -1e10) | (raw > 1e10) | np.isinf(raw) | np.isnan(raw)
    
    # ── Normalize (Per-band Z-score) ──────────────────────────────────────────
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

    labels = np.zeros((H, W), dtype=np.int64)

    meta = {
        'bands': C, 'lines': H, 'line_samples': W,
        'wavelengths': wavelengths,
        'fits_path': str(fits_path),
    }

    print(f"  Loaded:     cube {raw.shape}")
    return raw, labels, meta
