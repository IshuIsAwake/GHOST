"""Checkpoints: plain dicts that load with weights_only=True and say which architecture made them."""
import pickle

import numpy as np
import pytest
import torch

from ghost.archs import detect_checkpoint_arch
from ghost.v0_2.checkpoint import (FORMAT, FORMAT_VERSION, build_payload, fingerprint,
                                   load_checkpoint, model_from_checkpoint, save_checkpoint)
from ghost.v0_2.model import build_model, default_config


def _payload(model):
    return build_payload(
        model=model,
        class_ids=[2, 5, 9],
        preprocessing={"cr_mode": "full", "x_axis": "index", "wavelengths": None},
        scene={"shape": [4, 4, 24], "fingerprint": "abc", "label_fingerprint": "def"},
        split={"mode": "ratio", "seed": 1, "params": {"train_ratio": 0.2},
               "train_idx": [0, 1], "val_idx": [2], "test_idx": [3, 4]},
        training={"best_epoch": 3, "args": {"lr": 0.001}},
    )


def test_round_trip_gives_identical_logits(tmp_path):
    model = build_model(default_config(24, 3))
    model.eval()
    x = torch.randn(10, 24)
    path = tmp_path / "m.pt"
    save_checkpoint(path, _payload(model))
    torch.load(path, weights_only=True)
    restored = model_from_checkpoint(load_checkpoint(path))
    with torch.no_grad():
        assert torch.equal(model(x), restored(x))


def test_payload_carries_the_contract(tmp_path):
    path = tmp_path / "m.pt"
    save_checkpoint(path, _payload(build_model(default_config(24, 3))))
    ckpt = load_checkpoint(path)
    assert ckpt["format"] == FORMAT and ckpt["format_version"] == FORMAT_VERSION
    assert ckpt["arch"] == "0.2.0"
    assert {"ghost_version", "model_config", "state_dict", "class_ids", "preprocessing",
            "scene", "split", "training"} <= set(ckpt)
    assert ckpt["class_ids"] == [2, 5, 9]
    assert torch.equal(ckpt["split"]["test_idx"], torch.tensor([3, 4]))


def test_fingerprint_is_stable_and_sensitive():
    a = np.random.default_rng(0).random((5, 4, 3)).astype(np.float32)
    assert fingerprint(a) == fingerprint(a.copy())
    b = a.copy()
    b[2, 1, 0] += 1e-3
    assert fingerprint(a) != fingerprint(b)
    assert fingerprint(a) != fingerprint(a.reshape(4, 5, 3))


def test_detect_checkpoint_arch(tmp_path):
    v02 = tmp_path / "m.pt"
    save_checkpoint(v02, _payload(build_model(default_config(24, 3))))
    assert detect_checkpoint_arch(v02) == "0.2.0"

    v01 = tmp_path / "spt_models.pkl"
    with open(v01, "wb") as f:
        pickle.dump({"tree": {}, "trained_models": {}}, f)
    assert detect_checkpoint_arch(v01) == "0.1.7"

    flat = tmp_path / "best_model.pth"
    torch.save({"w": torch.zeros(2)}, flat)
    with pytest.raises(ValueError, match="v0.1"):
        detect_checkpoint_arch(flat)


def test_load_rejects_foreign_files(tmp_path):
    path = tmp_path / "x.pt"
    torch.save({"a": 1}, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)
