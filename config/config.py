from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Check for Ablation variables
ABLATION_RUN_NAME = os.environ.get("ABLATION_RUN_NAME", "")
if ABLATION_RUN_NAME:
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints" / ABLATION_RUN_NAME
    LOG_DIR = OUTPUT_DIR / "logs" / ABLATION_RUN_NAME
else:
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    LOG_DIR = OUTPUT_DIR / "logs"

SPLITS_DIR = DATASET_DIR / "splits"

IMAGES_DIR = DATASET_DIR / "images"
MASKS_DIR = DATASET_DIR / "masks"
ANALYSIS_DIR = DATASET_DIR / "analysis"

# Input and task settings
INPUT_MODE = "rgb"

IMG_HEIGHT = int(os.environ.get("ABLATION_IMG_HEIGHT", 384))
IMG_WIDTH = int(os.environ.get("ABLATION_IMG_WIDTH", 640))
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)  # (H, W)

IN_CHANNELS = 3
NUM_CLASSES = 2

# Runtime and reproducibility
SEED = 42
DETERMINISTIC = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Architecture and Loss
WIDTH_MULT = float(os.environ.get("ABLATION_WIDTH_MULT", 1.0))
USE_BOUNDARY_LOSS = os.environ.get("ABLATION_USE_BOUNDARY", "True").lower() == "true"

# DataLoader
BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True

# Optimization defaults
EPOCHS = 80
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4

# Checkpointing
SAVE_EVERY_EPOCHS = 5
BEST_CHECKPOINT_NAME = "best_model.pth"


def ensure_paths() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED, deterministic: bool = DETERMINISTIC) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
