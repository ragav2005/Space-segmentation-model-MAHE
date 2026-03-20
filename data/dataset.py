import os
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class MultiModalDrivableDataset(Dataset):
    """
    Multi-modal PyTorch Dataset for Real-Time Drivable Space Segmentation.
    Loads RGB Images (Camera) and matching Depth/Height maps (LIDAR).
    """
    def __init__(self, data_root, split_file, cameras, transform=None):
        """
        Args:
            data_root (str/Path): Base dataset directory (e.g., 'dataset/')
            split_file (str/Path): Path to split text file (e.g., 'dataset/splits/train.txt')
            cameras (list): List of camera names to load
            transform (callable, optional): Albumentations augmentations
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.cameras = cameras
        
        # Read sample tokens from the split file (each token represents a keyframe snapshot)
        with open(split_file, 'r') as f:
            tokens = [line.strip() for line in f if line.strip()]
        
        # Flatten all (camera, token) pairs to create independent training samples
        self.samples = []
        for token in tokens:
            for cam in self.cameras:
                # Basic validation to ensure the image exists (avoids broken paths mid-training)
                img_path = self.data_root / cam / 'images' / f"{token}.png"
                if img_path.exists():
                    self.samples.append((cam, token))

        print(f"Loaded {len(self.samples)} {str(split_file).split('/')[-1]} multi-modal samples from {len(tokens)} keyframes.")

    def __len__(self):
        return len(self.samples)

    def read_image_rgb(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Missing RGB image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    def read_grayscale_float(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback to zero-matrix if depth/height packet dropped in specific view
            return np.zeros((256, 512), dtype=np.float32)
        # Assuming preprocessing saved as uint8 grayscale scaled 0-255. Convert back to normalized float 0-1
        return img.astype(np.float32) / 255.0

    def read_mask(self, path):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
             raise FileNotFoundError(f"Missing Mask: {path}")
        # Assuming mask is 0/1 or 0/255. Force to binary 0/1.
        return (mask > 0).astype(np.uint8)

    def __getitem__(self, idx):
        cam, token = self.samples[idx]
        
        # Construct Paths
        img_path = self.data_root / cam / 'images' / f"{token}.png"
        mask_path = self.data_root / cam / 'masks' / f"{token}.png"
        depth_path = self.data_root / cam / 'depth' / f"{token}.png"
        height_path = self.data_root / cam / 'height' / f"{token}.png"

        # Load Modalities
        image = self.read_image_rgb(img_path)          # (H, W, 3)
        mask = self.read_mask(mask_path)               # (H, W)
        depth = self.read_grayscale_float(depth_path)  # (H, W)
        height = self.read_grayscale_float(height_path)# (H, W)

        if self.transform:
            augmented = self.transform(
                image=image,
                mask=mask,
                depth=depth,
                height=height
            )
            
            # Albumentations outputs
            image_aug = augmented['image'] # [3, H, W] tensor (because of ToTensorV2)
            mask_aug = augmented['mask']   # [H, W] tensor (or numpy, ToTensorV2 handles masks)
            depth_aug = augmented['depth']
            height_aug = augmented['height']
            
            # Ensure tensors
            if not isinstance(depth_aug, torch.Tensor):
                depth_aug = torch.from_numpy(depth_aug).float()
                height_aug = torch.from_numpy(height_aug).float()
                mask_aug = torch.from_numpy(mask_aug).long()
            
            # Expand dims for concatenation (H,W) -> (1, H, W)
            depth_aug = depth_aug.unsqueeze(0).float()
            height_aug = height_aug.unsqueeze(0).float()
            mask_aug = mask_aug.long()

            # Fused Tensor (5, H, W) -> RGB(3) + Depth(1) + Height(1)
            fused_input = torch.cat([image_aug, depth_aug, height_aug], dim=0)

            return fused_input, mask_aug

        else:
            # Fallback without albumentations (mostly structural fallback, shouldn't hit ideally)
            image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            depth_t = torch.from_numpy(depth).unsqueeze(0).float()
            height_t = torch.from_numpy(height).unsqueeze(0).float()
            mask_t = torch.from_numpy(mask).long()
            
            fused_input = torch.cat([image_t, depth_t, height_t], dim=0)
            return fused_input, mask_t
