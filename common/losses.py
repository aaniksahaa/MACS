"""Binary segmentation losses and the training-time metrics that go with them."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE
        return F_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersect = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2. * intersect + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        true_pos = (pred * target).sum(dim=(1, 2, 3))
        false_neg = (target * (1 - pred)).sum(dim=(1, 2, 3))
        false_pos = ((1 - target) * pred).sum(dim=(1, 2, 3))
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        return 1 - tversky.mean()


class CombinedLoss(nn.Module):
    """Dice + Focal + BCE, the loss every MACS source model and student is trained with."""

    def __init__(self, focal_alpha=0.8, focal_gamma=2, smooth=1.):
        super().__init__()
        self.dice = DiceLoss(smooth=smooth)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return self.dice(pred, target) + self.focal(pred, target) + self.bce(pred, target)


LOSSES = ("combined", "bce", "dice", "focal", "tversky")


def build_loss(name, focal_alpha=0.8, focal_gamma=2, tversky_alpha=0.5, tversky_beta=0.5):
    name = name.lower()
    if name == "combined":
        return CombinedLoss(focal_alpha=focal_alpha, focal_gamma=focal_gamma)
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "dice":
        return DiceLoss()
    if name == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    if name == "tversky":
        return TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
    raise ValueError(f"Unknown loss '{name}'. Available: {', '.join(LOSSES)}")


def dice_coef(logits, target, smooth=1.):
    """Hard Dice of thresholded sigmoid predictions over the whole batch."""
    pred = (torch.sigmoid(logits) > 0.5).float()
    intersect = (pred * target).sum()
    return (2. * intersect + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(logits, target, smooth=1.):
    pred = (torch.sigmoid(logits) > 0.5).float()
    intersect = (pred * target).sum()
    union = pred.sum() + target.sum() - intersect
    return (intersect + smooth) / (union + smooth)
