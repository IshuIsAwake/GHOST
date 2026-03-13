"""
PDS3 reader for CRISM hyperspectral data from NASA PDS.

What is PDS3?
    PDS3 is NASA's Planetary Data System version 3 format, used for archiving
    all planetary science data since the 1980s. Every file comes in a pair:
      - .IMG  — the raw binary data (the actual pixels/values)
      - .LBL  — a plain text "label" file describing how to read the binary

Why do we need our own reader?
    There's no standard Python library that handles PDS3 CRISM files cleanly.
    The label tells us everything we need: dimensions, data type, layout.
    We parse the label, then use numpy to read the binary directly.

CRISM Data Layout (BIL = Band Interleaved by Line):
    Imagine stacking the image like this:
      [all 107 band values for row 0] → [all 107 band values for row 1] → ...
    In memory this looks like: (H, B, W) which we transpose to (B, H, W) = (C, H, W)

Output:
    data   — (C, H, W) float32 numpy array, normalized to zero mean / unit std
    labels — (H, W) zeros (CRISM has no hand-labelled ground truth)
    meta   — dict with scene info (bands, spatial dims, unit, etc.)
"""

import re
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Label parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_lbl(lbl_path: str) -> dict:
    """
    Parse a PDS3 .lbl file into a flat key→value dict.

    The label is a text file with entries like:
        KEY = VALUE
    Some values span multiple lines or are wrapped in quotes.
    We only care about a handful of keys — extract them with regex.
    """
    text = Path(lbl_path).read_text(errors='replace')

    def get(key):
        m = re.search(rf'^\s*{key}\s*=\s*(.+?)(?=\n\S|\Z)', text,
                      re.MULTILINE | re.DOTALL)
        return m.group(1).strip().strip('"').split('\n')[0].strip() if m else None

    return {
        'lines':       int(get('LINES') or 0),
        'line_samples': int(get('LINE_SAMPLES') or 0),
        'bands':       int(get('BANDS') or 0),
        'sample_bits': int(get('SAMPLE_BITS') or 32),
        'sample_type': (get('SAMPLE_TYPE') or 'PC_REAL').upper(),
        'interleave':  (get('BAND_STORAGE_TYPE') or 'LINE_INTERLEAVED').upper(),
        'unit':        (get('UNIT') or 'UNKNOWN').strip('"'),
        'record_bytes': int(get('RECORD_BYTES') or 0),
    }


def _sample_type_to_dtype(sample_type: str, bits: int) -> np.dtype:
    """
    Convert PDS3 SAMPLE_TYPE to a numpy dtype.

    PDS3 naming convention:
      PC_REAL       = little-endian float  (most CRISM products)
      IEEE_REAL     = big-endian float
      MSB_INTEGER   = big-endian signed int
      LSB_INTEGER   = little-endian signed int
      UNSIGNED_INTEGER = unsigned int
    """
    bytes_per_sample = bits // 8
    if 'REAL' in sample_type:
        base = 'f'
    elif 'INTEGER' in sample_type or 'INT' in sample_type:
        base = 'u' if 'UNSIGNED' in sample_type else 'i'
    else:
        base = 'f'  # fallback

    endian = '>' if sample_type.startswith(('MSB', 'IEEE', 'SUN')) else '<'
    return np.dtype(f'{endian}{base}{bytes_per_sample}')


# ─────────────────────────────────────────────────────────────────────────────
# Main reader
# ─────────────────────────────────────────────────────────────────────────────

def read(img_path: str, lbl_path: str = None) -> tuple:
    """
    Read a PDS3 CRISM .IMG file.

    Parameters
    ----------
    img_path : str
        Path to the .IMG binary file.
    lbl_path : str, optional
        Path to the .LBL label file. If None, looks for <img_path>.lbl or
        <img_path_without_extension>.lbl automatically.

    Returns
    -------
    data : (C, H, W) float32 numpy array
        Normalized hyperspectral cube ready for GHOST.
    labels : (H, W) int64 numpy array
        All zeros — CRISM has no hand-labelled ground truth.
    meta : dict
        Scene metadata: bands, spatial dims, unit, observation info.
    """
    img_path = Path(img_path)

    # ── Find label file ───────────────────────────────────────────────────────
    if lbl_path is None:
        # Try <name>.lbl and <name>.LBL
        for candidate in [img_path.with_suffix('.lbl'),
                          img_path.with_suffix('.LBL'),
                          img_path.parent / (img_path.stem + '.lbl')]:
            if candidate.exists():
                lbl_path = candidate
                break

    if lbl_path is None or not Path(lbl_path).exists():
        raise FileNotFoundError(
            f"Cannot find .lbl label file for {img_path}.\n"
            f"Expected: {img_path.with_suffix('.lbl')}\n"
            f"Pass the path explicitly: pds3_reader.read(img_path, lbl_path)"
        )

    # ── Parse label ───────────────────────────────────────────────────────────
    meta = _parse_lbl(str(lbl_path))

    H = meta['lines']
    W = meta['line_samples']
    C = meta['bands']
    dtype = _sample_type_to_dtype(meta['sample_type'], meta['sample_bits'])
    interleave = meta['interleave']

    print(f"[PDS3] {img_path.name}")
    print(f"  Dims:      {H} lines × {W} samples × {C} bands")
    print(f"  Interleave: {interleave}")
    print(f"  Dtype:     {dtype}  ({meta['sample_type']}, {meta['sample_bits']}-bit)")
    print(f"  Unit:      {meta['unit']}")

    if H == 0 or W == 0 or C == 0:
        raise ValueError(
            f"Failed to parse dimensions from label. Got H={H} W={W} C={C}.\n"
            f"Check that the .lbl file is correct: {lbl_path}"
        )

    # ── Read binary data ───────────────────────────────────────────────────────
    # Memory-map the file — reads only what we need, doesn't load 52MB into RAM all at once
    raw = np.memmap(str(img_path), dtype=dtype, mode='r')

    # Total samples in the image cube
    n_image_samples = H * C * W

    if raw.size < n_image_samples:
        raise ValueError(
            f"File too small! Expected {n_image_samples} samples for "
            f"({H}, {C}, {W}) cube, got {raw.size}."
        )

    image_flat = raw[:n_image_samples]

    # ── Reshape based on interleave format ────────────────────────────────────
    # BIL (Band Interleaved by Line) — most common for CRISM:
    #   Layout in memory: for each row, all C bands, then W samples per band
    #   Shape in memory: (H, C, W)  → we want (C, H, W)
    #
    # BSQ (Band Sequential):
    #   Layout: full H×W image for band 0, then band 1, etc.
    #   Shape in memory: (C, H, W) — already what we want
    #
    # BIP (Band Interleaved by Pixel):
    #   Layout: for each pixel, all C bands consecutively
    #   Shape in memory: (H, W, C) → we want (C, H, W)

    if 'LINE' in interleave or interleave == 'BIL':
        cube = image_flat.reshape(H, C, W).transpose(1, 0, 2)  # → (C, H, W)
    elif 'BAND' in interleave or interleave == 'BSQ':
        cube = image_flat.reshape(C, H, W)                     # already (C, H, W)
    elif 'PIXEL' in interleave or interleave == 'BIP':
        cube = image_flat.reshape(H, W, C).transpose(2, 0, 1)  # → (C, H, W)
    else:
        # Fallback: assume BIL
        cube = image_flat.reshape(H, C, W).transpose(1, 0, 2)

    # Convert to float32 (some products are int16 scaled)
    cube = cube.astype(np.float32)

    # CRISM uses very large values for invalid pixels (e.g. -32768, 65535.0).
    # Normal reflectance (I/F) is between 0 and 1. Anything > 10 is noise/placeholder.
    invalid_mask = (cube < -100) | (cube > 10.0) | np.isinf(cube) | np.isnan(cube)

    # ── Normalize (Per-band Z-score) ──────────────────────────────────────────
    # Normalize each band independently so spectral shapes are preserved 
    # and variations aren't washed out by bright/dark wavelengths.
    for b in range(C):
        band = cube[b]
        band_invalid = invalid_mask[b]
        valid_pixels = band[~band_invalid]
        
        if valid_pixels.size > 0:
            mu  = valid_pixels.mean()
            std = valid_pixels.std()
            # Normalize valid pixels in this band
            cube[b] = (band - mu) / (std + 1e-8)
        
        # Strictly zero out invalid pixels in this band
        cube[b][band_invalid] = 0.0

    if invalid_mask.any():
        print(f"  Invalid pixels: {invalid_mask.sum()} — normalized and zeroed.")


    # ── Labels (all zeros — no ground truth for CRISM) ────────────────────────
    labels = np.zeros((H, W), dtype=np.int64)

    meta.update({'img_path': str(img_path), 'lbl_path': str(lbl_path)})

    print(f"  Loaded:     cube {cube.shape}  labels {labels.shape}")
    return cube, labels, meta
