import csv
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class AdvancedTrainingLogger:
    """
    Advanced Training Monitoring Logger matching Phase 4.5 requirements.
    Tracks: Loss component analysis, multi-modal metrics, and training stability.
    """
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard Writer
        self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        
        # Initialize CSV logger for metrics backup
        self.csv_path = self.log_dir / 'training_metrics.csv'
        self.csv_file = open(self.csv_path, mode='w', newline='')
        self.writer = csv.writer(self.csv_file)
        
        # Write CSV header and initialize stability tracker
        self.writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Train_mIoU', 'Val_mIoU', 'Val_Drivable_IoU', 'LR'])
        self.best_val_loss = float('inf')

    def log_epoch(self, epoch, train_loss, val_loss, train_metrics, val_metrics, lr):
        """Logs scalar metrics per epoch to TensorBoard and CSV"""
        
        # 1. TensorBoard Logging
        self.tb_writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch)
        self.tb_writer.add_scalars('mIoU', {'Train': train_metrics['mIoU'], 'Validation': val_metrics['mIoU']}, epoch)
        self.tb_writer.add_scalars('Drivable_IoU', {'Train': train_metrics['Drivable_IoU'], 'Validation': val_metrics['Drivable_IoU']}, epoch)
        self.tb_writer.add_scalar('Learning_Rate', lr, epoch)
        
        # 2. CSV Logging
        self.writer.writerow([
            epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", 
            f"{train_metrics['mIoU']:.2f}", f"{val_metrics['mIoU']:.2f}", 
            f"{val_metrics['Drivable_IoU']:.2f}", f"{lr:.6f}"
        ])
        self.csv_file.flush()
        
        # 3. Training Stability Check
        stability_warning = ""
        if val_loss > self.best_val_loss * 1.5:  # Loss spike detection
            stability_warning = " ⚠️ WARNING: Validation Loss Spike Detected! Check gradient stability."
        self.best_val_loss = min(self.best_val_loss, val_loss)
        
        return stability_warning

    def close(self):
        self.tb_writer.close()
        self.csv_file.close()
