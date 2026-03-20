import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

from nuscenes.nuscenes import NuScenes

sys.path.append(str(Path(__file__).parent.parent))
from config import config

def visualize_multimodal_sample(sample_token=None, base_dir=config.DATASET_DIR):
    """Visualize RGB, Depth, Height, and Mask for a sample."""
    cam_name = config.CAMERAS[0]
    cam_path = Path(base_dir) / cam_name
    if sample_token is None:
        import random
        images_dir = cam_path / 'images'
        sample_files = list(images_dir.glob('*.png'))
        if not sample_files:
            print("No samples found. Run preprocessing first.")
            return
        sample_token = random.choice(sample_files).stem

    img_path = cam_path / 'images' / f"{sample_token}.png"
    mask_path = cam_path / 'masks' / f"{sample_token}.png"
    depth_path = cam_path / 'depth' / f"{sample_token}.png"
    height_path = cam_path / 'height' / f"{sample_token}.png"
    if not img_path.exists():
        print(f"Sample {sample_token} does not exist.")
        return
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    height = cv2.imread(str(height_path), cv2.IMREAD_ANYDEPTH)
    
    depth_vis = (depth / 65535.0 * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    height_vis = (height / 65535.0 * 255).astype(np.uint8)
    height_colored = cv2.applyColorMap(height_vis, cv2.COLORMAP_PLASMA)
    height_colored = cv2.cvtColor(height_colored, cv2.COLOR_BGR2RGB)
    fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle(f"Multi-Modal Validation: {sample_token}", fontsize=16)
    
    axs[0, 0].imshow(img)
    axs[0, 0].set_title('RGB Camera')
    axs[0, 0].axis('off')
    axs[0, 1].imshow(img)
    transparent_mask = np.ma.masked_where(mask == 0, mask)
    axs[0, 1].imshow(transparent_mask, alpha=0.5, cmap='autumn')
    axs[0, 1].set_title('Drivable Area Mask')
    axs[0, 1].axis('off')
    axs[1, 0].imshow(img)
    axs[1, 0].imshow(depth_colored, alpha=0.6)
    axs[1, 0].set_title('LIDAR Depth')
    axs[1, 0].axis('off')
    axs[1, 1].imshow(img)
    axs[1, 1].imshow(height_colored, alpha=0.6)
    axs[1, 1].set_title('LIDAR Height')
    axs[1, 1].axis('off')

    out_file = Path('outputs/visualizations/sample_multimodal.png')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_file))
    print(f"Saved visualization to: {out_file}")

def create_dataset_splits(dataroot=config.DATA_ROOT, out_dir=config.DATASET_DIR):
    """Create train/val/test splits (80/10/10) stratified by scene."""
    print(f"Creating dataset splits...")
    nusc = NuScenes(version='v1.0-mini', dataroot=str(dataroot), verbose=False)
    np.random.seed(42)
    scenes = nusc.scene
    scene_tokens = [s['token'] for s in scenes]
    np.random.shuffle(scene_tokens)
    n_total = len(scene_tokens)
    n_train = max(1, int(n_total * 0.8))
    n_val = max(1, int(n_total * 0.1))
    train_scenes = scene_tokens[:n_train]
    val_scenes = scene_tokens[n_train:n_train+n_val]
    test_scenes = scene_tokens[n_train+n_val:]
    
    splits = {'train': [], 'val': [], 'test': []}
    for s in nusc.sample:
        if s['scene_token'] in train_scenes:
            splits['train'].append(s['token'])
        elif s['scene_token'] in val_scenes:
            splits['val'].append(s['token'])
        else:
            splits['test'].append(s['token'])
    out_path = Path(out_dir) / 'splits'
    out_path.mkdir(exist_ok=True)
    for k, v in splits.items():
        with open(out_path / f"{k}.txt", "w") as f:
            f.write("\n".join(v))
    print(f"Splits: Train={len(splits['train'])} Val={len(splits['val'])} Test={len(splits['test'])}")

if __name__ == '__main__':
    create_dataset_splits()
    visualize_multimodal_sample()
