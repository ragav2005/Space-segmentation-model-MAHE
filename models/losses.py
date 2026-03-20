import torch
import torch.nn as nn
import torch.nn.functional as F

class OHEMCrossEntropyLoss(nn.Module):
    """
    Online Hard Example Mining Cross Entropy Loss
    Focuses on the top K% hardest pixels (highest loss)
    """
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000):
        super(OHEMCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, pred, target):
        loss = self.criterion(pred, target)
        
        # Handle ignore index
        mask = target != self.ignore_index
        loss = loss[mask]
        
        if loss.numel() == 0:
            return loss.sum() * 0.0

        # Hard example mining
        loss, _ = torch.sort(loss, descending=True)
        
        if loss.numel() > self.min_kept:
            # Keep at least min_kept, and everything above thresh
            threshold_idx = min(self.min_kept, loss.numel() - 1)
            threshold_val = loss[threshold_idx]
            threshold_val = max(threshold_val, -torch.log(torch.tensor(self.thresh, device=loss.device)))
            loss = loss[loss >= threshold_val]

        return loss.mean()

class DiceLoss(nn.Module):
    """
    Dice loss for improving region boundaries and class imbalance.
    """
    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(torch.clamp(target, 0, pred.shape[1]-1), num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        mask = (target != self.ignore_index).float().unsqueeze(1)
        pred = pred * mask
        target_one_hot = target_one_hot * mask

        intersection = torch.sum(pred * target_one_hot, dim=(0, 2, 3))
        union = torch.sum(pred, dim=(0, 2, 3)) + torch.sum(target_one_hot, dim=(0, 2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class CombinedLoss(nn.Module):
    """
    OHEM + Dice combination for robust segmentation.
    Weights optimized for real-time drivable segmentations.
    """
    def __init__(self, weight_ohem=0.6, weight_dice=0.4):
        super(CombinedLoss, self).__init__()
        self.weight_ohem = weight_ohem
        self.weight_dice = weight_dice
        self.ohem = OHEMCrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        loss_ohem = self.ohem(pred, target)
        loss_dice = self.dice(pred, target)
        return self.weight_ohem * loss_ohem + self.weight_dice * loss_dice
