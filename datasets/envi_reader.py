"""
ENVI hyperspectral file reader.

What is ENVI?
    ENVI (Environment for Visualizing Images) is a commercial remote sensing
    software. Its file format has become the de-facto standard for sharing
    hyperspectral data because it's extremely simple:
      - .hdr  — plain text header describing dimensions, data type, interleave
      - .raw / .img / .dat / .bin  — flat binary data, no compression

    CRISM data exported from the CAT tool, many airborne sensors (AVIRIS,
    HyMap), and most downloaded datasets come in this format.

Output:
    data   — (C, H, W) float32 numpy array, normalized
    labels — (H, W) zeros (or loaded from a separate file if provided)
    meta   — dict with wavelengths (if in header) and scene info
"""

import re
import numpy as np
from pathlib import Path


# Map ENVI data type codes → numpy dtypes
_ENVI_DTYPE = {
    1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32,
    5: np.float64, 6: np.complex64, 9: np.complex128,
    12: np.uint16, 13: np.uint32, 14: np.int64, 15: np.uint64,
}


def _parse_hdr(hdr_path: str) -> dict:
    """Parse ENVI .hdr key=value pairs into a dict."""
    text = Path(hdr_path).read_text(errors='replace')

    # Remove comments (lines starting with ;)
    lines = [l for l in text.splitlines() if not l.strip().startswith(';')]
    text  = '\n'.join(lines)

    parsed = {}
    # Multi-line values are enclosed in { ... }
    # Single-line: key = value
    pattern = re.compile(r'(\w[\w\s]*?)\s*=\s*(\{[^}]*\}|[^\n]+)', re.MULTILINE)
    for m in pattern.finditer(text):
        key = m.group(1).strip().lower().replace(' ', '_')
        val = m.group(2).strip()
        parsed[key] = val

    return parsed


def _parse_list(s: str) -> list:
    """Parse an ENVI list like {1.2, 3.4, 5.6} into a Python list of floats."""
    s = s.strip('{}').strip()
    try:
        return [float(x.strip()) for x in s.split(',') if x.strip()]
    except ValueError:
        return []


def _find_data_file(hdr_path: Path) -> Path:
    """Find the binary data file paired with a .hdr."""
    stem = hdr_path.stem
    parent = hdr_path.parent
    for ext in ('.img', '.raw', '.dat', '.bin', '.sli', ''):
        candidate = parent / (stem + ext)
        if candidate.exists() and candidate != hdr_path:
            return candidate
    # Try without stripping .hdr suffix
    bare = parent / stem
    if bare.exists():
        return bare
    raise FileNotFoundError(
        f"Cannot find binary data file for {hdr_path}.\n"
        f"Expected one of: {stem}.img / .raw / .dat / .bin in {parent}"
    )


def read(hdr_path: str, lbl_path: str = None) -> tuple:
    """
    Read an ENVI hyperspectral file.

    Parameters
    ----------
    hdr_path : str
        Path to the .hdr header file.
    lbl_path : str, optional
        Path to ground-truth labels (.hdr or .mat). If None, returns zeros.

    Returns
    -------
    data   : (C, H, W) float32
    labels : (H, W) int64
    meta   : dict
    """
    hdr_path = Path(hdr_path)
    hdr      = _parse_hdr(str(hdr_path))

    H         = int(hdr.get('lines', 0))
    W         = int(hdr.get('samples', 0))
    C         = int(hdr.get('bands', 0))
    dtype_key = int(hdr.get('data_type', 4))
    interleave = hdr.get('interleave', 'bil').upper()
    byte_order = int(hdr.get('byte_order', 0))  # 0=little, 1=big
    header_offset = int(hdr.get('header_offset', 0))

    if H == 0 or W == 0 or C == 0:
        raise ValueError(f"Failed to parse ENVI header dims. Got H={H}, W={W}, C={C}")

    endian = '<' if byte_order == 0 else '>'
    base_dtype = _ENVI_DTYPE.get(dtype_key, np.float32)
    dtype = np.dtype(f'{endian}{base_dtype().dtype.kind}{base_dtype().itemsize}')

    wavelengths = _parse_list(hdr.get('wavelength', '{}'))

    data_file = _find_data_file(hdr_path)
    print(f"[ENVI] {data_file.name}")
    print(f"  Dims:       {H} lines × {W} samples × {C} bands")
    print(f"  Interleave: {interleave}  |  Dtype: {dtype}")
    if wavelengths:
        print(f"  Wavelengths: {wavelengths[0]:.1f} – {wavelengths[-1]:.1f} nm")

    raw = np.memmap(str(data_file), dtype=dtype, mode='r',
                    offset=header_offset)

    n_image_samples = H * C * W
    cube_flat = raw[:n_image_samples]

    if interleave in ('BIL', 'LINE_INTERLEAVED'):
        cube = cube_flat.reshape(H, C, W).transpose(1, 0, 2)  # (C, H, W)
    elif interleave in ('BSQ', 'BAND_SEQUENTIAL'):
        cube = cube_flat.reshape(C, H, W)
    elif interleave in ('BIP', 'PIXEL_INTERLEAVED'):
        cube = cube_flat.reshape(H, W, C).transpose(2, 0, 1)  # (C, H, W)
    else:
        cube = cube_flat.reshape(H, C, W).transpose(1, 0, 2)

    cube = cube.astype(np.float32)

    invalid = (cube < -1e10) | (cube > 1e10) | np.isinf(cube) | np.isnan(cube)

    # ── Normalize (Per-band Z-score) ──────────────────────────────────────────
    for b in range(C):
        band = cube[b]
        band_invalid = invalid[b]
        valid_pixels = band[~band_invalid]
        if valid_pixels.size > 0:
            mu  = valid_pixels.mean()
            std = valid_pixels.std()
            cube[b] = (band - mu) / (std + 1e-8)
        cube[b][band_invalid] = 0.0

    labels = np.zeros((H, W), dtype=np.int64)

    meta = {'lines': H, 'line_samples': W, 'bands': C,
            'wavelengths': wavelengths, 'hdr_path': str(hdr_path)}

    print(f"  Loaded:     cube {cube.shape}")
    return cube, labels, meta
