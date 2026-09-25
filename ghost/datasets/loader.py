"""Universal hyperspectral loader: .mat, ENVI, TIFF/GeoTIFF and HDF5 cubes, always returned as (H, W, Bands)."""
from __future__ import annotations

import importlib
import os

import numpy as np
import scipy.io as sio

EXT_MAP = {
    '.mat': 'mat',
    '.hdr': 'envi', '.img': 'envi', '.lan': 'envi',
    '.tif': 'tiff', '.tiff': 'tiff',
    '.h5': 'hdf5', '.hdf5': 'hdf5', '.he5': 'hdf5', '.hdf': 'hdf5', '.nc': 'hdf5',
}
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')
_PIP_NAMES = {'spectral': 'spectral', 'rasterio': 'rasterio', 'h5py': 'h5py', 'PIL': 'Pillow'}
_ENVI_KEYS = ['description', 'samples', 'lines', 'bands', 'header offset', 'data type', 'interleave',
              'byte order', 'wavelength', 'wavelength units', 'band names', 'map info',
              'coordinate system string', 'default bands', 'fwhm', 'reflectance scale factor',
              'sensor type']


def require(name: str):
    """Import an optional dependency, or raise ImportError naming the pip package."""
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        pkg = _PIP_NAMES.get(name.split('.')[0], name)
        raise ImportError(
            f"Missing dependency: {name}\n"
            f"Install it with:  pip install {pkg}\n"
            f"Or install all format dependencies:  pip install ghost-hsi[convert]"
        ) from exc


def split_dataset_specifier(path: str) -> tuple[str, str | None]:
    """Split 'file.h5:/dataset' or 'file.h5::dataset' into (file, dataset); real files pass through."""
    if os.path.exists(path):
        return path, None
    if '://' not in path and ':/' in path:
        file_path, key = path.rsplit(':/', 1)
        return file_path, key
    if '::' in path:
        file_path, key = path.rsplit('::', 1)
        return file_path, key
    return path, None


def detect_format(path: str) -> str:
    ext = os.path.splitext(split_dataset_specifier(path)[0])[1].lower()
    fmt = EXT_MAP.get(ext)
    if fmt is None:
        raise ValueError(f"Unrecognised file extension '{ext}'. Supported: {', '.join(sorted(EXT_MAP))}")
    return fmt


def json_safe(val):
    """Make an HDF5 attribute value JSON-serialisable."""
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='replace')
    return val


def _wavelengths(values, num_bands: int):
    """Band centres as floats, or None unless there is one finite, strictly increasing value per band."""
    if values is None:
        return None
    try:
        wl = np.asarray(values, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if wl.size != num_bands or not np.isfinite(wl).all() or not (np.diff(wl) > 0).all():
        return None
    return wl.tolist()


def _attach_wavelengths(meta: dict, candidates, data: np.ndarray):
    if data.ndim != 3:
        return
    for values in candidates:
        wl = _wavelengths(values, data.shape[-1])
        if wl is not None:
            meta['wavelengths'] = wl
            return


# ── Per-format loaders (raw dtype, canonical orientation) ───────────────────

def load_envi(path: str) -> tuple[np.ndarray, dict]:
    """ENVI .hdr/.img pair; spectral already returns (rows, cols, bands)."""
    require('spectral')
    envi = require('spectral.io.envi')

    base, ext = os.path.splitext(path)
    if ext.lower() == '.hdr':
        hdr_path, img_path = path, None
        for candidate_ext in ['', '.img', '.dat', '.raw', '.bsq', '.bil', '.bip']:
            candidate = base + candidate_ext
            if candidate != hdr_path and os.path.isfile(candidate):
                img_path = candidate
                break
    else:
        img_path, hdr_path = path, base + '.hdr'
        if not os.path.isfile(hdr_path):
            raise FileNotFoundError(f"Cannot find ENVI header: {hdr_path}\n"
                                    f"ENVI files require a .hdr header alongside the data file.")

    img = envi.open(hdr_path, img_path) if img_path is not None else envi.open(hdr_path)
    data = np.array(img.load())

    header = img.metadata if hasattr(img, 'metadata') else {}
    meta = {key: header[key] for key in _ENVI_KEYS if key in header}
    meta['_source_format'] = 'ENVI'
    meta['_source_file'] = os.path.abspath(hdr_path)
    _attach_wavelengths(meta, [header.get('wavelength')], data)
    return data, meta


def load_tiff(path: str) -> tuple[np.ndarray, dict]:
    """TIFF/GeoTIFF; rasterio reads band-first, so the cube is always transposed to (H, W, bands)."""
    rasterio = require('rasterio')
    meta = {}
    with rasterio.open(path) as src:
        data = np.transpose(src.read(), (1, 2, 0))
        meta['driver'] = src.driver
        meta['dtype'] = str(src.dtypes[0])
        meta['nodata'] = src.nodata
        meta['width'] = src.width
        meta['height'] = src.height
        meta['count'] = src.count
        if src.crs is not None:
            meta['crs'] = src.crs.to_string()
            meta['crs_wkt'] = src.crs.to_wkt()
        if src.transform is not None:
            t = src.transform
            meta['transform'] = [t.a, t.b, t.c, t.d, t.e, t.f]
        meta['bounds'] = {'left': src.bounds.left, 'bottom': src.bounds.bottom,
                          'right': src.bounds.right, 'top': src.bounds.top}
        if src.descriptions and any(d is not None for d in src.descriptions):
            meta['band_descriptions'] = list(src.descriptions)
        tags = src.tags()
        if tags:
            meta['tags'] = dict(tags)
        band_tags = {str(i): dict(src.tags(i)) for i in range(1, src.count + 1) if src.tags(i)}
        if band_tags:
            meta['band_tags'] = band_tags
    meta['_source_format'] = 'GeoTIFF' if meta.get('crs') else 'TIFF'
    meta['_source_file'] = os.path.abspath(path)
    return data, meta


def load_hdf5(path: str) -> tuple[np.ndarray, dict]:
    """HDF5; picks the largest dataset unless 'file.h5:/dataset' names one. A (C, H, W) cube with C
    strictly smallest is moved to (H, W, C)."""
    h5py = require('h5py')
    path, dataset_key = split_dataset_specifier(path)

    meta = {}
    with h5py.File(path, 'r') as f:
        datasets = {}

        def _visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = obj.shape

        f.visititems(_visitor)
        meta['available_datasets'] = {k: list(v) for k, v in datasets.items()}
        if dataset_key is not None:
            if dataset_key not in f:
                raise KeyError(f"Dataset '{dataset_key}' not found in {path}.\n"
                               f"Available datasets: {list(datasets.keys())}")
        else:
            if not datasets:
                raise ValueError(f"No datasets found in {path}")
            dataset_key = max(datasets, key=lambda k: np.prod(datasets[k]))
        ds = f[dataset_key]
        data = ds[:]
        meta['dataset_used'] = dataset_key
        ds_attrs = {k: json_safe(v) for k, v in ds.attrs.items()}
        if ds_attrs:
            meta['dataset_attrs'] = ds_attrs
        file_attrs = {k: json_safe(v) for k, v in f.attrs.items()}
        if file_attrs:
            meta['file_attrs'] = file_attrs

    if data.ndim == 3:
        c, h, w = data.shape
        if c < h and c < w:
            data = np.transpose(data, (1, 2, 0))
            meta['_transposed'] = 'CHW → HWC'

    attrs = {**meta.get('file_attrs', {}), **meta.get('dataset_attrs', {})}
    _attach_wavelengths(meta, [v for k, v in attrs.items() if k.lower() in ('wavelength', 'wavelengths')], data)
    meta['_source_format'] = 'HDF5'
    meta['_source_file'] = os.path.abspath(path)
    return data, meta


def _mat_arrays(path: str) -> tuple[dict, str]:
    """Every array in a .mat file; MATLAB v7.3 files are HDF5 with the axes stored reversed."""
    try:
        mat = sio.loadmat(path)
        return {k: np.asarray(v) for k, v in mat.items() if not k.startswith('_')}, '5'
    except NotImplementedError:
        h5py = require('h5py')
        arrays = {}
        with h5py.File(path, 'r') as f:
            def _visitor(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim >= 2 and not name.startswith('#'):
                    arrays[name] = np.asarray(obj[()]).T

            f.visititems(_visitor)
        return arrays, '7.3'


def load_mat_cube(path: str) -> tuple[np.ndarray, dict]:
    """The largest 3-D array in a .mat file."""
    arrays, version = _mat_arrays(path)
    cubes = {k: v for k, v in arrays.items() if v.ndim == 3}
    if not cubes:
        shapes = {k: v.shape for k, v in arrays.items()}
        raise ValueError(f"No 3-D (H, W, Bands) array in {path}. Arrays found: {shapes}")
    key = max(cubes, key=lambda k: cubes[k].size)
    meta = {'_source_format': 'MAT', '_source_file': os.path.abspath(path), 'mat_key': key,
            'mat_version': version}
    _attach_wavelengths(meta, [v for k, v in arrays.items() if k.lower() in ('wavelength', 'wavelengths')],
                        cubes[key])
    return cubes[key], meta


_CUBE_LOADERS = {'mat': load_mat_cube, 'envi': load_envi, 'tiff': load_tiff, 'hdf5': load_hdf5}


# ── Public entry points ─────────────────────────────────────────────────────

def load_cube(path: str) -> tuple[np.ndarray, dict]:
    """Any supported cube as float32 (H, W, Bands) plus its metadata; 'wavelengths' is set when usable."""
    data, meta = _CUBE_LOADERS[detect_format(path)](path)
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D (H, W, Bands) cube in {path}, got shape {data.shape}")
    return data.astype(np.float32, copy=False), meta


def load_labels(path: str) -> tuple[np.ndarray, dict]:
    """A ground-truth file (.mat, image, TIFF, ENVI or HDF5), returned as stored."""
    file_path = split_dataset_specifier(path)[0]
    ext = os.path.splitext(file_path)[1].lower()
    meta = {'_source_file': os.path.abspath(file_path)}

    if ext == '.mat':
        arrays, version = _mat_arrays(path)
        if not arrays:
            raise ValueError(f"No arrays found in {path}")
        if len(arrays) == 1:
            key = next(iter(arrays))
        elif 'gt' in arrays:
            key = 'gt'
        elif 'groundtruth' in arrays:
            key = 'groundtruth'
        else:
            key = max(arrays, key=lambda k: arrays[k].size)
        meta.update({'mat_key': key, 'mat_version': version, '_source_format': 'MAT'})
        return np.squeeze(arrays[key]), meta

    if ext in IMAGE_EXTS:
        image = require('PIL.Image')
        gt = np.array(image.open(path))
        meta['_source_format'] = 'image'
        if gt.ndim == 3:
            meta['_note'] = ('Ground truth loaded as RGB image. '
                             'You may need to map RGB values to class labels manually.')
        return gt, meta

    if ext in ('.tif', '.tiff'):
        rasterio = require('rasterio')
        with rasterio.open(path) as src:
            gt = src.read(1)
            if src.crs is not None:
                meta['crs'] = src.crs.to_string()
        meta['_source_format'] = 'GeoTIFF'
        return gt, meta

    fmt = EXT_MAP.get(ext)
    if fmt in ('envi', 'hdf5'):
        data, source_meta = (load_envi if fmt == 'envi' else load_hdf5)(path)
        meta.update(source_meta)
        return np.squeeze(data), meta

    raise ValueError(f"Unsupported ground truth format: {ext}")
