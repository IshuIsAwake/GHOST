"""v0.2 checkpoints: one plain dict of tensors and Python values, loadable with weights_only=True."""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from ghost.v0_2 import ARCH_VERSION
from ghost.v0_2.model import build_model

FORMAT = 'ghost-checkpoint'
FORMAT_VERSION = 1
SPLIT_KEYS = ('train_idx', 'val_idx', 'test_idx')


def fingerprint(array) -> str:
    """sha256 over shape, dtype and bytes; tells predict whether it is looking at the training scene."""
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(repr((a.shape, str(a.dtype))).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _plain(value):
    """numpy → Python types, recursively, so the weights-only unpickler accepts the payload."""
    if isinstance(value, dict):
        return {(k.item() if isinstance(k, np.generic) else k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_payload(model, class_ids, preprocessing: dict, scene: dict, split: dict, training: dict) -> dict:
    from ghost import __version__
    split = {k: (torch.as_tensor(np.asarray(v, dtype=np.int64)) if k in SPLIT_KEYS else _plain(v))
             for k, v in split.items()}
    return {
        'format': FORMAT,
        'format_version': FORMAT_VERSION,
        'arch': ARCH_VERSION,
        'ghost_version': __version__,
        'model_config': _plain(model.config),
        'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'class_ids': [int(c) for c in class_ids],
        'preprocessing': _plain(preprocessing),
        'scene': _plain(scene),
        'split': split,
        'training': _plain(training),
    }


def save_checkpoint(path, payload: dict):
    torch.save(payload, str(path))


def load_checkpoint(path) -> dict:
    try:
        ckpt = torch.load(str(path), map_location='cpu', weights_only=True)
    except Exception as exc:
        raise ValueError(f"{path} is not a GHOST v0.2 checkpoint ({type(exc).__name__})") from exc
    if not isinstance(ckpt, dict) or ckpt.get('format') != FORMAT:
        raise ValueError(f"{path} is not a GHOST v0.2 checkpoint")
    if ckpt.get('format_version', 0) > FORMAT_VERSION:
        raise ValueError(f"{path} was written by a newer GHOST (format {ckpt['format_version']}); upgrade ghost-hsi")
    return ckpt


def model_from_checkpoint(ckpt: dict):
    model = build_model(ckpt['model_config'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model
