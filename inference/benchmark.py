import sys
import os
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import config as cfg
from models.mobilenet_deeplab import DeepLabV3Plus
from data.dataset import MultiModalDrivableDataset
from data.augmentations import get_val_transforms
from utils.metrics import SegmentationMetrics

def sync_time(device):
    """Synchronize CUDA to ensure accurate timing, or just return time.time()."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return time.time()

def run_benchmark(checkpoint_path):
    print("="*60)
    print("🚀 Drivable Space Segmentation Benchmark")
    print("="*60)

    # 1. Device Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    print(f"Device: {device.type.upper()}")

    # 2. Data Loading
    print("Loading validation dataset...")
    val_split = cfg.DATASET_DIR / 'splits' / 'val.txt'
    
    val_dataset = MultiModalDrivableDataset(
        cfg.DATASET_DIR, val_split, cfg.CAMERAS, transform=get_val_transforms(cfg.IMG_SIZE)
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS if device.type == 'cuda' else 0,
        pin_memory=cfg.PIN_MEMORY if device.type == 'cuda' else False
    )
    print(f"Validation Samples: {len(val_dataset)}")
    
    # 3. Model Initialization
    print("\nInitializing model...")
    model = DeepLabV3Plus(in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES)
    
    print(f"Loading weights from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return
        
    state_dict = torch.load(checkpoint_path, map_location=device)
    # Handle if saving format was dict containing 'model_state_dict' (like our AdvancedLogger)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
        
    model = model.to(device)
    model.eval()

    metrics = SegmentationMetrics(num_classes=cfg.NUM_CLASSES, device=device)

    # GPU Warmup (Crucial for accurate FPS calculation)
    print("\nWarming up GPU...")
    warmup_batches = 5
    with torch.no_grad():
        with torch.amp.autocast(device_type=device_type):
            for i, (images, targets) in enumerate(val_loader):
                if i >= warmup_batches:
                    break
                images = images.to(device)
                _ = model(images)
    
    print("GPU Warmup complete.")

    # 4. Benchmarking Loop
    print("\nStarting evaluation...")
    total_images = 0
    total_network_time = 0.0 # Time spent ONLY in the model forward pass
    total_pipeline_time = 0.0 # Time spent loading data, moving to GPU, logic, etc
    
    metrics.reset()

    # Sync before starting the pipeline timer
    pipeline_start = sync_time(device)

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Benchmarking"):
            images = images.to(device)
            targets = targets.to(device)
            B = images.size(0)
            
            # Start network timer
            network_start = sync_time(device)
            
            # FP16 Inference
            with torch.amp.autocast(device_type=device_type):
                outputs = model(images)
            
            # End network timer
            network_time = sync_time(device) - network_start
            total_network_time += network_time
            total_images += B

            # Update Metrics
            metrics.update(outputs, targets)

    # End pipeline timer
    total_pipeline_time = sync_time(device) - pipeline_start

    # 5. Calculate Final Statistics
    results = metrics.get_metrics()
    
    # FPS Calculations
    network_fps = total_images / total_network_time
    pipeline_fps = total_images / total_pipeline_time
    
    print("\n" + "="*60)
    print(" BENCHMARK RESULTS")
    print("="*60)
    
    print("\n SPEED METRICS:")
    print(f"Total Images Processed: {total_images}")
    print(f"Pure Network FPS:       {network_fps:.2f} frames/sec")
    print(f"End-to-End FPS:         {pipeline_fps:.2f} frames/sec")
    print(f"Average Latency:        {(1000.0 / network_fps):.2f} ms")

    print("\n ACCURACY METRICS:")
    for metric_name, value in results.items():
        print(f"{metric_name}:{' '*(20-len(metric_name))}{value:.2f}%")

    print("="*60)
    
    # Save the output to a text log for documentation
    summary_txt = f"Network FPS: {network_fps:.2f} | Drivable IoU: {results.get('Drivable_IoU', 0.0):.2f}%"
    with open(PROJECT_ROOT / 'outputs' / 'logs' / 'benchmark_summary.txt', 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {summary_txt}\n")

if __name__ == '__main__':
    best_checkpoint = PROJECT_ROOT / 'outputs' / 'checkpoints' / 'best_deeplabv3plus_multimodal.pth'
    run_benchmark(best_checkpoint)
