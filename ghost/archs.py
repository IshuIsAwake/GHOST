"""Architectures shipped in this package, and which one a checkpoint belongs to."""
from __future__ import annotations

import importlib
import zipfile

DEFAULT_ARCH = '0.2.0'
ARCHS = {
    '0.1.7': {'train': 'ghost.train:main', 'train_spt': 'ghost.train_rssp:main',
              'predict': 'ghost.predict:main', 'visualize': 'ghost.visualize:main'},
    '0.2.0': {'train': 'ghost.v0_2.train:main', 'predict': 'ghost.v0_2.predict:main',
              'visualize': 'ghost.v0_2.visualize:main'},
}


def available() -> str:
    return ', '.join(ARCHS)


def resolve_arch(spec: str | None = None) -> str:
    """Exact versions only for now ('0.2.0' or 'v0.2.0'); None means the default."""
    if spec is None:
        return DEFAULT_ARCH
    key = spec.strip()
    key = key[1:] if key[:1] in ('v', 'V') else key
    if key not in ARCHS:
        raise ValueError(f"Unknown architecture '{spec}'. Available: {available()} (exact versions only)")
    return key


def get_entry(arch: str, command: str):
    target = ARCHS[arch].get(command)
    if target is None:
        supported = [a for a, commands in ARCHS.items() if command in commands]
        raise ValueError(f"'{command}' is only available for architecture {', '.join(supported)}, not {arch}")
    module, func = target.split(':')
    return getattr(importlib.import_module(module), func)


def detect_checkpoint_arch(path) -> str:
    """v0.2 checkpoints are torch zip files carrying a format marker; v0.1 SPT models are raw pickles."""
    path = str(path)
    if zipfile.is_zipfile(path):
        import torch
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=True)
        except Exception as exc:
            raise ValueError(f"{path} is not a GHOST checkpoint ({type(exc).__name__})") from exc
        if isinstance(ckpt, dict) and ckpt.get('format') == 'ghost-checkpoint':
            return ckpt['arch']
        raise ValueError(f"{path} looks like a v0.1 `ghost train` state_dict; v0.1 predict and visualize "
                         f"only read SPT models (spt_models.pkl)")
    with open(path, 'rb') as f:
        if f.read(1) == b'\x80':
            return '0.1.7'
    raise ValueError(f"{path} is not a GHOST checkpoint")
