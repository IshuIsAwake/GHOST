import torch
import torch.nn as nn
import torch.nn.functional as F


class SquaredCELoss(nn.Module):
    """
    Squared Cross-Entropy loss.
    L = CE(x, y)^2

    Amplifies the penalty for already-high-loss pixels (hard / minority-class
    examples) without requiring explicit class frequency counts.
    Softer than focal loss — scaling comes from loss magnitude, not a
    hand-tuned gamma.

    Averages over non-background pixels only (ignore_index=0).
    """

    def __init__(self, ignore_index: int = 0):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(inputs, targets,
                               ignore_index=self.ignore_index,
                               reduction='none')
        mask = targets != self.ignore_index
        return (ce[mask] ** 2).mean() if mask.any() else ce.sum() * 0.0


class FocalLoss(nn.Module):
    """
    Focal loss for dense prediction.
    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Averages over non-background pixels only (ignore_index=0).
    """

    def __init__(self, gamma: float = 2.0, ignore_index: int = 0):
        super().__init__()
        self.gamma        = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(inputs, targets,
                               ignore_index=self.ignore_index,
                               reduction='none')
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce

        mask = targets != self.ignore_index
        return loss[mask].mean() if mask.any() else loss.sum() * 0.0


def build_criterion(loss_type: str,
                    focal_gamma: float = 2.0,
                    ignore_index: int = 0) -> nn.Module:
    """
    Loss function factory.

    loss_type:
        'ce'         — standard CrossEntropyLoss (default, existing behaviour)
        'squared_ce' — CE squared; amplifies hard-example penalty without
                       explicit class weighting
        'focal'      — focal loss with tunable gamma
        'dice'       — combined CrossEntropy + Dice loss (CE weight=0.5, Dice weight=0.5)
    """
    valid = {'ce', 'squared_ce', 'focal', 'dice'}
    if loss_type not in valid:
        raise ValueError(
            f"Unknown loss type: '{loss_type}'. Choose from: {sorted(valid)}"
        )

    if loss_type == 'ce':
        return nn.CrossEntropyLoss(ignore_index=ignore_index)

    elif loss_type == 'squared_ce':
        return SquaredCELoss(ignore_index=ignore_index)

    elif loss_type == 'focal':
        return FocalLoss(gamma=focal_gamma, ignore_index=ignore_index)

    elif loss_type == 'dice':
        from ghost.losses import CEDiceLoss
        return CEDiceLoss(ignore_index=ignore_index)