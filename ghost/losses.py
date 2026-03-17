import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Per-class Dice loss, ignoring background (class 0).
    """
    def __init__(self, smooth=1.0, ignore_index=0):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: (B, C, H, W)
        # targets: (B, H, W)
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets.clamp(min=0), num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()   # (B, C, H, W)

        # Mask: ignore background pixels
        mask = (targets != self.ignore_index).unsqueeze(1).float()  # (B, 1, H, W)

        probs           = probs * mask
        targets_one_hot = targets_one_hot * mask

        # Per-class Dice
        dims = (0, 2, 3)  # sum over batch, H, W
        intersection = (probs * targets_one_hot).sum(dim=dims)
        cardinality  = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Skip background class
        dice_per_class = dice_per_class[1:]  # exclude class 0

        return 1.0 - dice_per_class.mean()


class CEDiceLoss(nn.Module):
    """
    Combined CrossEntropy + Dice loss.
    """
    def __init__(self, ce_weight=0.5, dice_weight=0.5, ignore_index=0):
        super().__init__()
        self.ce          = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice        = DiceLoss(ignore_index=ignore_index)
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.ce_weight * self.ce(logits, targets) + \
               self.dice_weight * self.dice(logits, targets)
