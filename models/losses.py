"""
Phase 3: Advanced Loss Functions for Robust Binary Segmentation

Implements multiple loss functions optimized for handling pseudo-label noise:
- Focal Loss (class imbalance, hard examples)
- Dice Loss (boundary precision, region coverage)
- Boundary-Aware Loss (edge preservation)
- OHEM (Online Hard Example Mining for noisy labels)
- Weighted Combined Loss (noise-robust training)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance and hard examples.
    Focuses training on hard negatives (non-drivable areas incorrectly predicted as drivable).
    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 2, H, W) model predictions
            target: (B, H, W) binary ground truth
        """
        # Get probabilities for drivable class (class 1)
        probs = F.softmax(logits, dim=1)
        p_t = probs[:, 1]  # P(class=1)

        # Where target=1, use p_t; where target=0, use 1-p_t
        p_t = torch.where(target.bool(), p_t, 1.0 - p_t)

        # Focal weight: (1 - p_t)^gamma - down-weights easy examples
        focal_weight = (1.0 - p_t) ** self.gamma

        # CE loss
        ce = F.cross_entropy(logits, target.long(), reduction="none")

        # Apply alpha and focal weighting
        focal_loss = self.alpha * focal_weight * ce

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss (F1 loss) - better for boundary precision and region coverage.
    Directly optimizes the mIoU metric.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 2, H, W) model predictions
            target: (B, H, W) binary ground truth
        Returns:
            Dice loss (1.0 - Dice coefficient)
        """
        probs = F.softmax(logits, dim=1)[:, 1]  # P(drivable)
        target_f = target.float()

        inter = (probs * target_f).sum()
        union = probs.sum() + target_f.sum()

        dice = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice


class BoundaryAwareLoss(nn.Module):
    """
    Boundary-Aware Loss - gives higher weight to predictions near edges.
    Improves segmentation quality at object boundaries.
    """

    def __init__(self, sigma: float = 2.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.sigma = sigma
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 2, H, W) model predictions
            target: (B, H, W) binary ground truth
        Returns:
            Boundary-weighted loss
        """
        # Compute boundary map using Sobel-like edge detection
        target_f = target.float().unsqueeze(1)  # (B, 1, H, W)

        # Sobel filters for edge detection
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=target.device)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=target.device)
        kernel_x = kernel_x.view(1, 1, 3, 3)
        kernel_y = kernel_y.view(1, 1, 3, 3)

        edges_x = F.conv2d(target_f, kernel_x, padding=1)
        edges_y = F.conv2d(target_f, kernel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2 + self.eps).squeeze(1)

        # Weight mask: higher weight near boundaries
        boundary_weight = 1.0 + self.sigma * edges

        # Compute CE loss with boundary weighting
        ce = F.cross_entropy(logits, target.long(), reduction="none")
        weighted_ce = (ce * boundary_weight).mean()

        return weighted_ce


class OHEMLoss(nn.Module):
    """
    Online Hard Example Mining (OHEM) Loss.
    Selects hardest examples during training to handle noisy pseudo-labels.
    Reference: Li et al., "Training Region-based Object Detectors with Online Hard Example Mining"
    """

    def __init__(self, base_loss: nn.Module, threshold: float = 0.7) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.threshold = threshold

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 2, H, W) model predictions
            target: (B, H, W) binary ground truth
        Returns:
            OHEM-weighted loss
        """
        # Compute per-pixel loss
        loss_map = F.cross_entropy(logits, target.long(), reduction="none")  # (B, H, W)

        # Compute confidence
        probs = F.softmax(logits, dim=1)
        confidence = probs.max(dim=1)[0]  # (B, H, W)

        # Select hard examples: low confidence OR high loss
        hard_mask = (confidence < self.threshold) | (loss_map > loss_map.mean(dim=(1, 2), keepdim=True))

        if hard_mask.sum() == 0:
            # No hard examples, use base loss
            return loss_map.mean()

        # Return loss only for hard examples
        return loss_map[hard_mask].mean()


class RobustCombinedLoss(nn.Module):
    """
    Robust Combined Loss optimized for 75-80% mIoU target.
    Combines multiple loss functions weighted for noise robustness.
    """

    def __init__(
        self,
        ce_weight: float = 0.4,
        focal_weight: float = 0.3,
        dice_weight: float = 0.2,
        boundary_weight: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        boundary_sigma: float = 2.0,
        ohem_enabled: bool = True,
        ohem_threshold: float = 0.7,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss()
        self.boundary = BoundaryAwareLoss(sigma=boundary_sigma)
        self.ohem = OHEMLoss(self.ce, threshold=ohem_threshold) if ohem_enabled else None

        # Normalize weights
        total_weight = ce_weight + focal_weight + dice_weight + boundary_weight
        self.ce_weight = ce_weight / total_weight
        self.focal_weight = focal_weight / total_weight
        self.dice_weight = dice_weight / total_weight
        self.boundary_weight = boundary_weight / total_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 2, H, W) model predictions
            target: (B, H, W) binary ground truth
        Returns:
            Combined loss (normalized)
        """
        ce_loss = self.ce(logits, target.long())
        focal_loss = self.focal(logits, target)
        dice_loss = self.dice(logits, target)
        boundary_loss = self.boundary(logits, target)

        # OHEM on cross-entropy
        if self.ohem is not None:
            ce_loss = self.ohem(logits, target)

        # Weighted combination
        total_loss = (
            self.ce_weight * ce_loss
            + self.focal_weight * focal_loss
            + self.dice_weight * dice_loss
            + self.boundary_weight * boundary_loss
        )

        return total_loss
