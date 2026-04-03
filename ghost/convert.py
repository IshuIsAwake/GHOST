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

from ghost.utils.display import (
    BOLD, RESET, CYAN, GREEN, YELLOW, RED, GRAY, _c,
    print_config_box,
)


# ── Format detection ─────────────────────────────────────────────────────────

_EXT_MAP = {
    '.hdr': 'envi',
    '.img': 'envi',
    '.lan': 'envi',
    '.tif': 'tiff',
    '.tiff': 'tiff',
    '.h5': 'hdf5',
    '.hdf5': 'hdf5',
    '.he5': 'hdf5',
    '.hdf': 'hdf5',
}


def _strip_dataset_specifier(path: str) -> str:
    """Remove HDF5 dataset specifier (e.g. 'file.h5:/dataset')."""
    if '://' not in path and ':/' in path:
        return path.rsplit(':/', 1)[0]
    if '::' in path:
        return path.rsplit('::', 1)[0]
    return path


def _detect_format(path: str) -> str:
    clean = _strip_dataset_specifier(path)
    ext = os.path.splitext(clean)[1].lower()
    # .nc files are NetCDF4/HDF5
    if ext == '.nc':
        ext = '.h5'
    fmt = _EXT_MAP.get(ext)
    if fmt is None:
        raise ValueError(
            f"Unrecognised file extension '{ext}'. "
            f"Supported: {', '.join(sorted(set(_EXT_MAP.values())))}"
        )
    return fmt


# ── Dependency checks ────────────────────────────────────────────────────────

def _check_dep(name: str):
    """Raise a helpful error when an optional dep is missing."""
    try:
        __import__(name)
    except ImportError:
        extras = {
            'spectral': 'spectral',
            'rasterio': 'rasterio',
            'h5py': 'h5py',
        }
        pkg = extras.get(name, name)
        print(
            f"\n{RED}{BOLD}  Missing dependency: {name}{RESET}\n"
            f"  Install it with:  pip install {pkg}\n"
            f"  Or install all convert deps:  pip install ghost-hsi[convert]\n"
        )
        sys.exit(1)


# ── Loaders ──────────────────────────────────────────────────────────────────

def _load_envi(path: str) -> tuple[np.ndarray, dict]:
    """Load an ENVI file (.hdr/.img pair). Returns (data, metadata)."""
    _check_dep('spectral')
    import spectral.io.envi as envi

    # Accept either .hdr or .img — find the pair
    base, ext = os.path.splitext(path)
    if ext.lower() == '.hdr':
        hdr_path = path
        # Try common data file extensions
        img_path = None
        for candidate_ext in ['', '.img', '.dat', '.raw', '.bsq', '.bil', '.bip']:
            candidate = base + candidate_ext
            if candidate != hdr_path and os.path.isfile(candidate):
                img_path = candidate
                break
        if img_path is None:
            # spectral can sometimes find it automatically
            img_path = None
    else:
        # User passed the data file; look for .hdr
        img_path = path
        hdr_path = base + '.hdr'
        if not os.path.isfile(hdr_path):
            raise FileNotFoundError(
                f"Cannot find ENVI header: {hdr_path}\n"
                f"ENVI files require a .hdr header alongside the data file."
            )

    if img_path is not None:
        img = envi.open(hdr_path, img_path)
    else:
        img = envi.open(hdr_path)

    data = np.array(img.load())

    # Extract metadata from the ENVI header
    meta = {}
    header = img.metadata if hasattr(img, 'metadata') else {}
    for key in ['description', 'samples', 'lines', 'bands', 'header offset',
                'data type', 'interleave', 'byte order', 'wavelength',
                'wavelength units', 'band names', 'map info',
                'coordinate system string', 'default bands',
                'fwhm', 'reflectance scale factor', 'sensor type']:
        if key in header:
            meta[key] = header[key]

    meta['_source_format'] = 'ENVI'
    meta['_source_file'] = os.path.abspath(hdr_path)

    return data, meta


def _load_tiff(path: str) -> tuple[np.ndarray, dict]:
    """Load a TIFF or GeoTIFF. Returns (data, metadata)."""
    _check_dep('rasterio')
    import rasterio

    meta = {}
    with rasterio.open(path) as src:
        # Read all bands → (bands, H, W), transpose to (H, W, bands)
        data = src.read()  # shape: (C, H, W)
        data = np.transpose(data, (1, 2, 0))  # → (H, W, C)

        meta['driver'] = src.driver
        meta['dtype'] = str(src.dtypes[0])
        meta['nodata'] = src.nodata
        meta['width'] = src.width
        meta['height'] = src.height
        meta['count'] = src.count

        # CRS
        if src.crs is not None:
            meta['crs'] = src.crs.to_string()
            meta['crs_wkt'] = src.crs.to_wkt()

        # Affine transform
        if src.transform is not None:
            t = src.transform
            meta['transform'] = [t.a, t.b, t.c, t.d, t.e, t.f]

        # Bounds
        meta['bounds'] = {
            'left': src.bounds.left,
            'bottom': src.bounds.bottom,
            'right': src.bounds.right,
            'top': src.bounds.top,
        }

        # Band descriptions / names
        if src.descriptions and any(d is not None for d in src.descriptions):
            meta['band_descriptions'] = list(src.descriptions)

        # Tags (TIFF metadata)
        tags = src.tags()
        if tags:
            meta['tags'] = dict(tags)

        # Per-band tags
        band_tags = {}
        for i in range(1, src.count + 1):
            bt = src.tags(i)
            if bt:
                band_tags[str(i)] = dict(bt)
        if band_tags:
            meta['band_tags'] = band_tags

    meta['_source_format'] = 'GeoTIFF' if meta.get('crs') else 'TIFF'
    meta['_source_file'] = os.path.abspath(path)

    return data, meta


def _load_hdf5(path: str) -> tuple[np.ndarray, dict]:
    """Load an HDF5 file. Returns (data, metadata).

    Auto-detects the main dataset (largest array) if no specific dataset
    is specified via 'path.h5:/dataset_name' syntax.
    """
    _check_dep('h5py')
    import h5py

    # Check for dataset specifier: file.h5:/path/to/dataset
    dataset_key = None
    if '://' not in path and ':/' in path:
        path, dataset_key = path.rsplit(':/', 1)
    elif '::' in path:
        path, dataset_key = path.rsplit('::', 1)

    meta = {}
    with h5py.File(path, 'r') as f:
        # Catalogue all datasets
        datasets = {}

        def _visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = obj.shape

        f.visititems(_visitor)
        meta['available_datasets'] = {k: list(v) for k, v in datasets.items()}

        if dataset_key is not None:
            if dataset_key not in f:
                raise KeyError(
                    f"Dataset '{dataset_key}' not found in {path}.\n"
                    f"Available datasets: {list(datasets.keys())}"
                )
            ds = f[dataset_key]
        else:
            # Pick the largest dataset by number of elements
            if not datasets:
                raise ValueError(f"No datasets found in {path}")
            dataset_key = max(datasets, key=lambda k: np.prod(datasets[k]))
            ds = f[dataset_key]

        data = ds[:]
        meta['dataset_used'] = dataset_key

        # Dataset attributes
        ds_attrs = {k: _json_safe(v) for k, v in ds.attrs.items()}
        if ds_attrs:
            meta['dataset_attrs'] = ds_attrs

        # File-level attributes
        file_attrs = {k: _json_safe(v) for k, v in f.attrs.items()}
        if file_attrs:
            meta['file_attrs'] = file_attrs

    # If data is 2D (H, W) or already 3D (H, W, C), keep as-is
    # If 3D as (C, H, W) with C << H and C << W, transpose
    if data.ndim == 3:
        c, h, w = data.shape
        if c < h and c < w:
            data = np.transpose(data, (1, 2, 0))
            meta['_transposed'] = 'CHW → HWC'

    meta['_source_format'] = 'HDF5'
    meta['_source_file'] = os.path.abspath(path)

    return data, meta


def _json_safe(val):
    """Make HDF5 attribute values JSON-serializable."""
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='replace')
    return val


# ── Format dispatch ──────────────────────────────────────────────────────────

_LOADERS = {
    'envi': _load_envi,
    'tiff': _load_tiff,
    'hdf5': _load_hdf5,
}


# ── Ground truth loading ─────────────────────────────────────────────────────

def _load_gt(path: str) -> tuple[np.ndarray, dict]:
    """Load a ground-truth file. Supports .mat, .png/.tif images, and ENVI."""
    ext = os.path.splitext(path)[1].lower()
    meta = {'_source_file': os.path.abspath(path)}

    if ext == '.mat':
        mat = sio.loadmat(path)
        # Find the ground-truth array (skip MATLAB internal keys)
        candidates = {k: v for k, v in mat.items() if not k.startswith('_')}
        if len(candidates) == 1:
            key = list(candidates.keys())[0]
        elif 'gt' in candidates:
            key = 'gt'
        elif 'groundtruth' in candidates:
            key = 'groundtruth'
        else:
            key = max(candidates, key=lambda k: np.prod(np.array(candidates[k]).shape))
        gt = np.array(candidates[key]).squeeze()
        meta['mat_key'] = key
        meta['_source_format'] = 'MAT'
        return gt, meta

    if ext in ('.png', '.jpg', '.jpeg', '.bmp'):
        from PIL import Image
        gt = np.array(Image.open(path))
        meta['_source_format'] = 'image'
        if gt.ndim == 3:
            meta['_note'] = (
                'Ground truth loaded as RGB image. '
                'You may need to map RGB values to class labels manually.'
            )
        return gt, meta

    if ext in ('.tif', '.tiff'):
        _check_dep('rasterio')
        import rasterio
        with rasterio.open(path) as src:
            gt = src.read(1)  # single band
            if src.crs is not None:
                meta['crs'] = src.crs.to_string()
        meta['_source_format'] = 'GeoTIFF'
        return gt, meta

    if ext in ('.hdr', '.img'):
        data, envi_meta = _load_envi(path)
        gt = data.squeeze()
        meta.update(envi_meta)
        return gt, meta

    raise ValueError(f"Unsupported ground truth format: {ext}")


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
    data, meta = _LOADERS[fmt](args.img)
    print(f"  {GREEN}✓{RESET} Loaded: shape={data.shape} dtype={data.dtype}")

    # ── Load ground truth ────────────────────────────────────────────────
    gt = None
    gt_meta = None
    if args.gt is not None:
        print(f"\n  {BOLD}Loading ground truth:{RESET} {args.gt}")
        gt, gt_meta = _load_gt(args.gt)
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
