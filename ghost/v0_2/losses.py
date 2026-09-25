"""Losses for (N, K) logits with labels 0..K−1. Nothing is ignored: index 0 is a real class here."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

LOSSES = ('ce', 'squared_ce', 'focal', 'dice')


class DiceLoss(nn.Module):
    """1 − mean soft Dice over all classes, pooled over the batch."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(target, logits.shape[1]).to(probs.dtype)
        inter = (probs * onehot).sum(0)
        card = probs.sum(0) + onehot.sum(0)
        return 1.0 - ((2.0 * inter + self.smooth) / (card + self.smooth)).mean()


class CEDiceLoss(nn.Module):
    """0.5 · cross-entropy + 0.5 · Dice, the pairing v0.1's `--loss dice` uses."""

    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()

    def forward(self, logits, target):
        return 0.5 * F.cross_entropy(logits, target) + 0.5 * self.dice(logits, target)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none')
        return ((1.0 - (-ce).exp()) ** self.gamma * ce).mean()


class SquaredCELoss(nn.Module):
    def forward(self, logits, target):
        return F.cross_entropy(logits, target) ** 2


def build_criterion(name: str, focal_gamma: float = 2.0) -> nn.Module:
    if name == 'ce':
        return nn.CrossEntropyLoss()
    if name == 'squared_ce':
        return SquaredCELoss()
    if name == 'focal':
        return FocalLoss(focal_gamma)
    if name == 'dice':
        return CEDiceLoss()
    raise ValueError(f"Unknown loss '{name}'. Choose from: {', '.join(LOSSES)}")
