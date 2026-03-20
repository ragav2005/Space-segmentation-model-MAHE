import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / 'v1.0-mini'

# Check if dataset exists
assert DATA_ROOT.exists(), f"Dataset not found at {DATA_ROOT}"

# Cameras and Sensors
CAMERAS = [
    'CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 
    'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]
LIDAR_SENSOR = 'LIDAR_TOP'

# Image Parameters
IMG_SIZE = (256, 512) # (height, width) for fast inference
IN_CHANNELS = 5  # RGB(3) + Depth(1) + Height(1)
NUM_CLASSES = 2  # Drivable vs Non-drivable

# Lidar Preprocessing Parameters
MAX_DEPTH = 80.0       # Max depth in meters (truncate beyond)
MIN_DEPTH = 1.0        # Min depth in meters
MAX_HEIGHT = 5.0       # Max height relative to sensor 
MIN_HEIGHT = -3.0      # Min height relative to sensor
LIDAR_DENSIFICATION = True # Apply optimal depth densification
DILATION_KERNEL_SIZE = 5   # Kernel size for depth filling


BATCH_SIZE = 12  # Reduced from 16 due to 5-channel input block in VRAM
NUM_WORKERS = 6  # Best IO handling for 6-core/12-thread 
PREFETCH_FACTOR = 2
PIN_MEMORY = True
USE_GRADIENT_CHECKPOINTING = True

# Training Parameters
EPOCHS = 120
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4

# Inference Parameters
TARGET_FPS = 170

# Advanced Training Configuration
LOSS_WEIGHTS = {
    'ohem': 0.6,
    'dice': 0.3,
    'lightweight_boundary': 0.1
}

# Output Directories
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
LOG_DIR = OUTPUT_DIR / 'logs'
DATASET_DIR = PROJECT_ROOT / 'dataset'

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Ensure camera output directories exist
for cam in CAMERAS:
    (DATASET_DIR / cam / 'images').mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / cam / 'masks').mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / cam / 'depth').mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / cam / 'height').mkdir(parents=True, exist_ok=True)
