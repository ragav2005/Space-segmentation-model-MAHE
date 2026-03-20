import sys
import os
import time
from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from config import config as cfg
from models.shufflenet_seg import ShuffleNetSegmentation
from models.losses import CombinedLoss
from data.dataset import MultiModalDrivableDataset
from data.augmentations import get_train_transforms, get_val_transforms
from utils.metrics import SegmentationMetrics
from utils.logger import AdvancedTrainingLogger

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, metrics):
    model.train()
    metrics.reset()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True) # Optimized zeroing
        
        # Forward pass with mixed precision
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with autocast(device_type=device_type, enabled=(device_type == 'cuda')): 
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        running_loss += loss.item()
        # Detach and disable grad for metrics plotting to save memory
        with torch.no_grad():
             metrics.update(outputs.detach(), targets)
             
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
    epoch_loss = running_loss / len(loader)
    epoch_metrics = metrics.get_metrics()
    return epoch_loss, epoch_metrics

@torch.no_grad()
def validate(model, loader, criterion, device, metrics):
    model.eval()
    metrics.reset()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Validation")
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with autocast(device_type=device_type, enabled=(device_type == 'cuda')):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        running_loss += loss.item()
        metrics.update(outputs, targets)
        
    epoch_loss = running_loss / len(loader)
    epoch_metrics = metrics.get_metrics()
    return epoch_loss, epoch_metrics

def main():
    print("Initializing Multi-Modal Training Pipeline ")
    
    # Check Hardware (Handling CPU gracefully for now before GPU migration)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Target Device: {device}")
    if device.type == 'cpu':
        print(" WARNING: Training on CPU. This will be extremely slow. Moving to NVIDIA GPU is highly recommended.")
        
    # Dataset Preparation
    train_split = cfg.DATASET_DIR / 'splits' / 'train.txt'
    val_split = cfg.DATASET_DIR / 'splits' / 'val.txt'
    
    train_dataset = MultiModalDrivableDataset(
        cfg.DATASET_DIR, train_split, cfg.CAMERAS, transform=get_train_transforms(cfg.IMG_SIZE)
    )
    val_dataset = MultiModalDrivableDataset(
        cfg.DATASET_DIR, val_split, cfg.CAMERAS, transform=get_val_transforms(cfg.IMG_SIZE)
    )
    
    # DataLoaders designed for RTX 3050 Memory Constraints
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS if device.type == 'cuda' else 0, # Disable workers on CPU to avoid crashes
        pin_memory=cfg.PIN_MEMORY if device.type == 'cuda' else False,
        prefetch_factor=cfg.PREFETCH_FACTOR if device.type == 'cuda' else None,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS if device.type == 'cuda' else 0,
        pin_memory=cfg.PIN_MEMORY if device.type == 'cuda' else False
    )
    
    # Model Initialization
    model = ShuffleNetSegmentation(in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES)
    model = model.to(device)
    
    # Advanced Loss & Optimization
    criterion = CombinedLoss(weight_ohem=cfg.LOSS_WEIGHTS['ohem'], weight_dice=cfg.LOSS_WEIGHTS['dice']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    # Mixed Precision Scaler for RTX 3050 (Falls back cleanly if on CPU)
    scaler = GradScaler(device.type, enabled=(device.type == 'cuda'))
    
    # Metrics Tracker & Logger
    metrics = SegmentationMetrics(num_classes=cfg.NUM_CLASSES, device=device)
    logger = AdvancedTrainingLogger(log_dir=cfg.LOG_DIR)
    
    best_miou = 0.0
    print(f"Starting Training for {cfg.EPOCHS} Epochs...\n")
    
    try:
        for epoch in range(1, cfg.EPOCHS + 1):
            print(f"Epoch {epoch}/{cfg.EPOCHS}")
            print("-" * 20)
            
            # Train Loop
            train_loss, train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, device, metrics
            )
            
            # Validation Loop
            val_loss, val_metrics = validate(
                model, val_loader, criterion, device, metrics
            )
            
            # Step LR Scheduler
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            # Log Metrics & Check Stability
            stability_warn = logger.log_epoch(epoch, train_loss, val_loss, train_metrics, val_metrics, current_lr)
            
            print(f"Train Loss: {train_loss:.4f} | Train mIoU: {train_metrics['mIoU']:.2f}% | Drivable IoU: {train_metrics['Drivable_IoU']:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val mIoU: {val_metrics['mIoU']:.2f}%   | Drivable IoU: {val_metrics['Drivable_IoU']:.2f}%")
            print(f"Current LR: {current_lr:.6f}{stability_warn}\n")
            
            # Save Best Model Checkpoint
            if val_metrics['mIoU'] > best_miou:
                best_miou = val_metrics['mIoU']
                save_path = cfg.CHECKPOINT_DIR / 'best_shufflenet_multimodal.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_miou': best_miou,
                }, save_path)
                print(f" New Best mIoU! Weights saved to {save_path} \n")
    finally:
        logger.close()

if __name__ == '__main__':
      main()
