"""
Bundled sample datasets for GHOST.

Usage:
    from ghost.data import indian_pines_path
    data_path, gt_path = indian_pines_path()
"""
from pathlib import Path

_DATA_DIR = Path(__file__).parent


def indian_pines_path():
    """Return (data_path, gt_path) for the bundled Indian Pines dataset."""
    data = _DATA_DIR / "indian_pines" / "Indian_pines_corrected.mat"
    gt = _DATA_DIR / "indian_pines" / "Indian_pines_gt.mat"
    if not data.exists():
        raise FileNotFoundError(f"Indian Pines data not found at {data}")
    return str(data), str(gt)
