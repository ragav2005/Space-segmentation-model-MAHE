from __future__ import annotations

import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int = 2) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.conf = torch.zeros((self.num_classes, self.num_classes), dtype=torch.long)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = torch.argmax(logits, dim=1).view(-1)
        tgt = target.view(-1)
        mask = (tgt >= 0) & (tgt < self.num_classes)
        idx = self.num_classes * tgt[mask] + pred[mask]
        binc = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.conf += binc.reshape(self.num_classes, self.num_classes).cpu()

    def compute(self) -> dict[str, float]:
        conf = self.conf.float()
        tp = torch.diag(conf)
        fp = conf.sum(0) - tp
        fn = conf.sum(1) - tp

        iou = tp / (tp + fp + fn + 1e-6)
        miou = iou.mean().item() * 100.0
        drivable_iou = iou[1].item() * 100.0 if self.num_classes > 1 else iou[0].item() * 100.0
        pix_acc = (tp.sum() / (conf.sum() + 1e-6)).item() * 100.0

        return {
            "mIoU": miou,
            "Drivable_IoU": drivable_iou,
            "Pixel_Acc": pix_acc,
        }
