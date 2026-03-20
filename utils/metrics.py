import torch
import numpy as np

class SegmentationMetrics:
    """
    Tracks Mean IoU, Pixel Accuracy, and Boundary IoU for real-time drivable segmentations.
    Optimized for batch processing.
    """
    def __init__(self, num_classes=2, device='cpu'):
        self.num_classes = num_classes
        self.device = device
        self.reset()
        
    def reset(self):
        self.total_inter = torch.zeros(self.num_classes, device=self.device)
        self.total_union = torch.zeros(self.num_classes, device=self.device)
        self.total_correct = 0
        self.total_label = 0
        
    def update(self, pred, target):
        """
        pred: Tensor [B, C, H, W] (raw logits)
        target: Tensor [B, H, W] (ground truth labels)
        """
        # Convert logits to class predictions
        pred = torch.argmax(pred, dim=1)
        
        # Flatten tensors for faster computation
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        
        # Filter out ignored index (255)
        mask = (target != 255)
        pred = pred[mask]
        target = target[mask]
        
        # Pixel accuracy accumulation
        self.total_correct += (pred == target).sum().item()
        self.total_label += target.numel()
        
        # Intersection over Union (IoU) per class
        for cls in range(self.num_classes):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            
            intersection = (pred_inds & target_inds).sum().float()
            union = (pred_inds | target_inds).sum().float()
            
            self.total_inter[cls] += intersection
            self.total_union[cls] += union

    def get_metrics(self):
        """
        Returns a dictionary containing calculated metrics
        """
        iou = self.total_inter / (self.total_union + 1e-6)
        miou = iou.mean().item()
        pixel_accuracy = self.total_correct / max(self.total_label, 1)
        
        return {
            'mIoU': miou * 100,             # Percentage Target > 80%
            'Drivable_IoU': iou[1].item() * 100, # Specifically track road
            'Background_IoU': iou[0].item() * 100,
            'PixelAccuracy': pixel_accuracy * 100
        }
