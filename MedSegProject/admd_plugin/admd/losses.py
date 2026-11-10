import torch
import torch.nn.functional as F


def dice_loss(logits, target, valid, eps=1.0):
    p = torch.sigmoid(logits) * valid
    t = target * valid
    num = 2 * (p * t).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


def focal_loss(logits, target, valid, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    at = alpha * target + (1 - alpha) * (1 - target)
    loss = at * (1 - pt) ** gamma * bce * valid
    return loss.sum() / valid.sum().clamp_min(1.0)


def seg_loss(logits, target, valid, w_dice=0.8):
    """0.8·Dice + 0.2·Focal as in MMSeg; `valid` masks out the SAM padding region."""
    return w_dice * dice_loss(logits, target, valid) + (1 - w_dice) * focal_loss(logits, target, valid)


@torch.no_grad()
def dice_coef_lowres(logits, target, valid, eps=1e-6):
    """Per-sample Dice at 256x256 (fast validation proxy; final numbers come from 04_metrics.py)."""
    p = ((logits > 0).float() * valid)
    t = target * valid
    inter = (p * t).sum(dim=(1, 2, 3))
    den = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
    return torch.where(den > 0, 2 * inter / (den + eps), torch.ones_like(den))
