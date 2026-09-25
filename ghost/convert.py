"""
ghost convert_to_mat — Convert ENVI / TIFF / GeoTIFF / HDF5 to .mat format.

Preserves all metadata in a JSON sidecar file for zero data loss.
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import scipy.io as sio

from ghost.datasets.loader import (
    detect_format as _detect_format,
    load_envi, load_hdf5, load_labels as _load_gt, load_mat_cube, load_tiff,
)
from ghost.utils.display import (
    BOLD, RESET, CYAN, GREEN, YELLOW, RED, GRAY, _c,
    print_config_box,
)

_LOADERS = {
    'envi': load_envi,
    'tiff': load_tiff,
    'hdf5': load_hdf5,
    'mat': load_mat_cube,
}


def _exit_missing_dependency(exc: ImportError):
    """CLI-only: print the loader's install hint and stop."""
    first, *rest = str(exc).splitlines() or ['Missing dependency']
    print(f"\n{RED}{BOLD}  {first}{RESET}")
    for line in rest:
        print(f"  {line}")
    print()
    sys.exit(1)


# ── Crop ─────────────────────────────────────────────────────────────────────

def _apply_crop(data: np.ndarray, gt: np.ndarray | None,
                crop: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply spatial crop: (y, x, height, width)."""
    y, x, h, w = crop
    H, W = data.shape[:2]
    if y < 0 or x < 0 or y + h > H or x + w > W:
        raise ValueError(
            f"Crop region ({y}, {x}, {h}, {w}) exceeds image bounds ({H}, {W})"
        )
    data = data[y:y+h, x:x+w]
    if gt is not None:
        gt = gt[y:y+h, x:x+w]
    return data, gt


# ── Summary printer ──────────────────────────────────────────────────────────

def _print_summary(data: np.ndarray, gt: np.ndarray | None,
                   meta: dict, gt_meta: dict | None,
                   out_dir: str, elapsed: float):
    """Print conversion summary with all relevant details."""
    W = 60
    print(f"\n{BOLD}{GREEN}{'═' * W}{RESET}")
    print(f"  {BOLD}{GREEN}Conversion complete!{RESET}")
    print(f"{BOLD}{GREEN}{'═' * W}{RESET}")

    # Image info
    print(f"\n  {BOLD}Image{RESET}")
    print(f"    Format       : {meta.get('_source_format', 'unknown')}")
    print(f"    Shape        : {data.shape}")
    print(f"    Spatial size : {data.shape[0]} x {data.shape[1]}")
    if data.ndim == 3:
        print(f"    Bands        : {data.shape[2]}")
    print(f"    Dtype        : {data.dtype}")
    print(f"    Value range  : [{data.min():.4g}, {data.max():.4g}]")
    size_mb = data.nbytes / (1024 * 1024)
    print(f"    Size in memory: {size_mb:.1f} MB")

    # Wavelengths if available
    wl = meta.get('wavelength')
    if wl:
        wl_floats = [float(w) for w in wl[:5]]
        suffix = f" ... ({len(wl)} total)" if len(wl) > 5 else ""
        print(f"    Wavelengths  : {wl_floats}{suffix}")

    # CRS if available
    crs = meta.get('crs')
    if crs:
        print(f"    CRS          : {crs}")

    # Ground truth info
    if gt is not None:
        print(f"\n  {BOLD}Ground Truth{RESET}")
        print(f"    Format       : {gt_meta.get('_source_format', 'unknown')}")
        print(f"    Shape        : {gt.shape}")
        print(f"    Dtype        : {gt.dtype}")
        classes = np.unique(gt)
        num_classes = len(classes)
        # If 0 is background, report separately
        has_bg = 0 in classes
        if has_bg:
            fg_classes = classes[classes != 0]
            print(f"    Classes      : {len(fg_classes)} (+ background 0)")
        else:
            print(f"    Classes      : {num_classes}")
        print(f"    Class labels : {classes.tolist()}")

        # Per-class pixel counts
        print(f"\n    {GRAY}Per-class pixel counts:{RESET}")
        for c in classes:
            count = int(np.sum(gt == c))
            label = "background" if c == 0 else f"class {c}"
            print(f"      {label:<16}: {count:>8,} px")

    # Output files
    print(f"\n  {BOLD}Output{RESET}")
    print(f"    Directory    : {os.path.abspath(out_dir)}")
    print(f"    data.mat     : image data (key='data')")
    if gt is not None:
        print(f"    gt.mat       : ground truth (key='gt')")
    print(f"    metadata.json: all preserved metadata")
    print(f"\n    {GRAY}Elapsed: {elapsed:.2f}s{RESET}")
    print(f"{BOLD}{GREEN}{'═' * W}{RESET}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog='ghost convert_to_mat',
        description='Convert ENVI / TIFF / GeoTIFF / HDF5 to .mat format',
    )
    parser.add_argument('--img', required=True,
                        help='Path to the image file (ENVI .hdr, .tif, .h5, etc.)')
    parser.add_argument('--gt', default=None,
                        help='Path to ground-truth labels (optional: .mat, .png, .tif, .hdr)')
    parser.add_argument('--out-dir', required=True,
                        help='Output directory for .mat and metadata files')
    parser.add_argument('--crop', nargs=4, type=int, default=None,
                        metavar=('Y', 'X', 'H', 'W'),
                        help='Spatial crop: Y X Height Width (e.g. --crop 448 2560 512 512)')
    parser.add_argument('--data-key', default='data',
                        help='Key name for image data in the output .mat (default: data)')
    parser.add_argument('--gt-key', default='gt',
                        help='Key name for ground truth in the output .mat (default: gt)')

    args = parser.parse_args()

    t0 = time.time()

    # ── Load image ───────────────────────────────────────────────────────
    print(f"\n  {BOLD}Loading image:{RESET} {args.img}")
    fmt = _detect_format(args.img)
    print(f"  {GRAY}Detected format: {fmt.upper()}{RESET}")
    try:
        data, meta = _LOADERS[fmt](args.img)
    except ImportError as exc:
        _exit_missing_dependency(exc)
    print(f"  {GREEN}✓{RESET} Loaded: shape={data.shape} dtype={data.dtype}")

    # ── Load ground truth ────────────────────────────────────────────────
    gt = None
    gt_meta = None
    if args.gt is not None:
        print(f"\n  {BOLD}Loading ground truth:{RESET} {args.gt}")
        try:
            gt, gt_meta = _load_gt(args.gt)
        except ImportError as exc:
            _exit_missing_dependency(exc)
        print(f"  {GREEN}✓{RESET} Loaded: shape={gt.shape} dtype={gt.dtype}")

        # Spatial dimension check
        if gt.shape[:2] != data.shape[:2]:
            print(f"  {YELLOW}⚠ Spatial mismatch: image={data.shape[:2]}, "
                  f"gt={gt.shape[:2]}{RESET}")

    # ── Crop ─────────────────────────────────────────────────────────────
    if args.crop is not None:
        y, x, h, w = args.crop
        print(f"\n  {BOLD}Cropping:{RESET} y={y} x={x} h={h} w={w}")
        data, gt = _apply_crop(data, gt, tuple(args.crop))
        meta['crop'] = {'y': y, 'x': x, 'height': h, 'width': w}
        print(f"  {GREEN}✓{RESET} Cropped to: {data.shape}")

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    # Save data
    data_path = os.path.join(args.out_dir, 'data.mat')
    print(f"\n  {BOLD}Saving:{RESET} {data_path}")
    sio.savemat(data_path, {args.data_key: data}, do_compression=True)
    data_size = os.path.getsize(data_path) / (1024 * 1024)
    print(f"  {GREEN}✓{RESET} data.mat ({data_size:.1f} MB)")

    # Save ground truth
    if gt is not None:
        gt_path = os.path.join(args.out_dir, 'gt.mat')
        sio.savemat(gt_path, {args.gt_key: gt}, do_compression=True)
        gt_size = os.path.getsize(gt_path) / (1024 * 1024)
        print(f"  {GREEN}✓{RESET} gt.mat ({gt_size:.1f} MB)")

    # Save metadata
    combined_meta = {
        'image': meta,
        'conversion': {
            'output_data_key': args.data_key,
            'output_data_shape': list(data.shape),
            'output_data_dtype': str(data.dtype),
        }
    }
    if gt is not None and gt_meta is not None:
        combined_meta['ground_truth'] = gt_meta
        combined_meta['conversion']['output_gt_key'] = args.gt_key
        combined_meta['conversion']['output_gt_shape'] = list(gt.shape)
        combined_meta['conversion']['output_gt_dtype'] = str(gt.dtype)

    meta_path = os.path.join(args.out_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(combined_meta, f, indent=2, default=str)
    print(f"  {GREEN}✓{RESET} metadata.json")

    elapsed = time.time() - t0
    _print_summary(data, gt, meta, gt_meta, args.out_dir, elapsed)
