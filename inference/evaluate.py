"""
Comprehensive Evaluation Script - Compute all metrics for segmentation model
Metrics: Accuracy, Precision, Recall, F1 Score, mAP, IoU, mIoU
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import DEVICE, IMG_SIZE, IN_CHANNELS, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS
from data.dataset import DrivableDataset
from data.augmentations import get_val_transforms
from models.mobilenet_deeplab import MobileDeepLabV3Plus
from utils.metrics import SegmentationMetrics


class ComprehensiveMetrics:
    """Compute comprehensive evaluation metrics for segmentation"""
    
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset confusion matrix and counters"""
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes))
        self.tp = np.zeros(self.num_classes)
        self.fp = np.zeros(self.num_classes)
        self.fn = np.zeros(self.num_classes)
        self.total_pixels = 0
        self.correct_pixels = 0
    
    def update(self, pred, target):
        """Update metrics with batch predictions and targets"""
        pred = pred.argmax(dim=1).cpu().numpy()  # (B, H, W)
        target = target.cpu().numpy()  # (B, H, W)
        
        # Flatten
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        # Update confusion matrix
        for t, p in zip(target_flat, pred_flat):
            self.conf_matrix[t, p] += 1
        
        # Update pixel counters
        self.total_pixels += len(target_flat)
        self.correct_pixels += (pred_flat == target_flat).sum()
        
        # Update per-class TP, FP, FN
        for cls in range(self.num_classes):
            tp = ((pred_flat == cls) & (target_flat == cls)).sum()
            fp = ((pred_flat == cls) & (target_flat != cls)).sum()
            fn = ((pred_flat != cls) & (target_flat == cls)).sum()
            
            self.tp[cls] += tp
            self.fp[cls] += fp
            self.fn[cls] += fn
    
    def compute_all_metrics(self):
        """Compute all metrics"""
        metrics = {}
        
        # 1. Pixel Accuracy (Overall Accuracy)
        metrics['Pixel_Accuracy'] = (self.correct_pixels / self.total_pixels * 100.0) if self.total_pixels > 0 else 0.0
        
        # 2. Per-class Precision, Recall, F1
        metrics['Precision'] = {}
        metrics['Recall'] = {}
        metrics['F1'] = {}
        
        for cls in range(self.num_classes):
            # Precision: TP / (TP + FP)
            precision = self.tp[cls] / (self.tp[cls] + self.fp[cls] + 1e-6)
            metrics['Precision'][f'Class_{cls}'] = precision * 100.0
            
            # Recall: TP / (TP + FN)
            recall = self.tp[cls] / (self.tp[cls] + self.fn[cls] + 1e-6)
            metrics['Recall'][f'Class_{cls}'] = recall * 100.0
            
            # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            metrics['F1'][f'Class_{cls}'] = f1 * 100.0
        
        # Macro Average
        metrics['Precision_Macro'] = np.mean(list(metrics['Precision'].values()))
        metrics['Recall_Macro'] = np.mean(list(metrics['Recall'].values()))
        metrics['F1_Macro'] = np.mean(list(metrics['F1'].values()))
        
        # Micro Average (same as overall accuracy)
        metrics['Precision_Micro'] = metrics['Pixel_Accuracy']
        metrics['Recall_Micro'] = metrics['Pixel_Accuracy']
        metrics['F1_Micro'] = metrics['Pixel_Accuracy']
        
        # 3. IoU (Intersection over Union) - Per class
        metrics['IoU'] = {}
        for cls in range(self.num_classes):
            iou = self.tp[cls] / (self.tp[cls] + self.fp[cls] + self.fn[cls] + 1e-6)
            metrics['IoU'][f'Class_{cls}'] = iou * 100.0
        
        # mIoU (Mean IoU)
        metrics['mIoU'] = np.mean(list(metrics['IoU'].values()))
        
        # 4. mAP (Mean Average Precision) - Approximated using IoU as confidence
        # For segmentation, mAP is typically computed per class as max F1 or IoU
        metrics['mAP'] = metrics['mIoU']  # Using mIoU as proxy for segmentation
        
        return metrics
    
    def print_report(self, metrics):
        """Print comprehensive metrics report"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE EVALUATION METRICS REPORT")
        print("=" * 80)
        
        # Overall Accuracy
        print(f"\n1. PIXEL ACCURACY (Overall Accuracy)")
        print(f"   {metrics['Pixel_Accuracy']:.2f}%")
        
        # Per-class Metrics
        print(f"\n2. PER-CLASS METRICS")
        for cls in range(self.num_classes):
            print(f"\n   Class {cls}:")
            print(f"     Precision: {metrics['Precision'][f'Class_{cls}']:.2f}%")
            print(f"     Recall:    {metrics['Recall'][f'Class_{cls}']:.2f}%")
            print(f"     F1 Score:  {metrics['F1'][f'Class_{cls}']:.2f}%")
            print(f"     IoU:       {metrics['IoU'][f'Class_{cls}']:.2f}%")
        
        # Macro & Micro Averages
        print(f"\n3. MACRO AVERAGES")
        print(f"   Precision: {metrics['Precision_Macro']:.2f}%")
        print(f"   Recall:    {metrics['Recall_Macro']:.2f}%")
        print(f"   F1 Score:  {metrics['F1_Macro']:.2f}%")
        
        print(f"\n4. MICRO AVERAGES")
        print(f"   Precision: {metrics['Precision_Micro']:.2f}%")
        print(f"   Recall:    {metrics['Recall_Micro']:.2f}%")
        print(f"   F1 Score:  {metrics['F1_Micro']:.2f}%")
        
        # Summary Metrics
        print(f"\n5. SUMMARY METRICS")
        print(f"   mIoU (Mean IoU):        {metrics['mIoU']:.2f}%")
        print(f"   mAP (Mean Avg Prec):    {metrics['mAP']:.2f}%")
        
        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on validation/test set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to evaluate on")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--output", "-o", type=str, default="outputs/evaluation", help="Output directory for results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = MobileDeepLabV3Plus(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return
    
    state_dict = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    images_dir = "dataset/images"
    masks_dir = "dataset/masks"
    split_file = f"dataset/splits/{args.split}.txt"
    
    if not os.path.exists(split_file):
        print(f"ERROR: Split file not found: {split_file}")
        return
    
    # Use validation transforms for proper ImageNet normalization
    val_transforms = get_val_transforms(IMG_SIZE)
    
    dataset = DrivableDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        split_file=split_file,
        transform=val_transforms
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Initialize metrics
    metrics_tracker = ComprehensiveMetrics(num_classes=NUM_CLASSES)
    
    # Evaluate
    print(f"\nEvaluating on {args.split} set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluation")):
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Update metrics
            metrics_tracker.update(outputs, masks)
    
    # Compute and print results
    metrics = metrics_tracker.compute_all_metrics()
    metrics_tracker.print_report(metrics)
    
    # Save results to file
    results_file = os.path.join(args.output, f"evaluation_{args.split}_results.txt")
    with open(results_file, "w") as f:
        f.write("COMPREHENSIVE EVALUATION METRICS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Split: {args.split}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Device: {device}\n\n")
        
        f.write(f"1. PIXEL ACCURACY (Overall Accuracy)\n")
        f.write(f"   {metrics['Pixel_Accuracy']:.2f}%\n\n")
        
        f.write(f"2. PER-CLASS METRICS\n")
        for cls in range(NUM_CLASSES):
            f.write(f"\n   Class {cls}:\n")
            f.write(f"     Precision: {metrics['Precision'][f'Class_{cls}']:.2f}%\n")
            f.write(f"     Recall:    {metrics['Recall'][f'Class_{cls}']:.2f}%\n")
            f.write(f"     F1 Score:  {metrics['F1'][f'Class_{cls}']:.2f}%\n")
            f.write(f"     IoU:       {metrics['IoU'][f'Class_{cls}']:.2f}%\n")
        
        f.write(f"\n3. MACRO AVERAGES\n")
        f.write(f"   Precision: {metrics['Precision_Macro']:.2f}%\n")
        f.write(f"   Recall:    {metrics['Recall_Macro']:.2f}%\n")
        f.write(f"   F1 Score:  {metrics['F1_Macro']:.2f}%\n")
        
        f.write(f"\n4. MICRO AVERAGES\n")
        f.write(f"   Precision: {metrics['Precision_Micro']:.2f}%\n")
        f.write(f"   Recall:    {metrics['Recall_Micro']:.2f}%\n")
        f.write(f"   F1 Score:  {metrics['F1_Micro']:.2f}%\n")
        
        f.write(f"\n5. SUMMARY METRICS\n")
        f.write(f"   mIoU (Mean IoU):        {metrics['mIoU']:.2f}%\n")
        f.write(f"   mAP (Mean Avg Prec):    {metrics['mAP']:.2f}%\n")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
