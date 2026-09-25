"""Model inference, shared by train and predict so both score pixels identically."""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def predict_logits(model, feats: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Logits for features already on the model's device."""
    model.eval()
    if len(feats) == 0:
        return feats.new_zeros((0, model.config['num_classes']))
    return torch.cat([model(feats[i:i + batch_size]) for i in range(0, len(feats), batch_size)])


@torch.no_grad()
def scene_predictions(model, feats: np.ndarray, valid: np.ndarray, class_ids: list, device,
                      batch_size: int = 8192) -> np.ndarray:
    """(H·W,) predicted label ids in original label space; invalid pixels are 0. Chunks move to the device
    one at a time, so the scene never has to fit in GPU memory."""
    model.eval()
    flat_valid = valid.reshape(-1)
    x = torch.from_numpy(feats[flat_valid])
    winners = [model(x[i:i + batch_size].to(device)).argmax(dim=1).cpu() for i in range(0, len(x), batch_size)]
    pred = np.zeros(flat_valid.size, dtype=np.int64)
    if winners:
        pred[flat_valid] = np.asarray(class_ids, dtype=np.int64)[torch.cat(winners).numpy()]
    return pred
