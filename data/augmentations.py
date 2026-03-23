import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size=(512, 512)):
    """
    Advanced Multi-Modal Augmentation Pipeline for Real-time Segmentation.
    Using Albumentations to ensure geometric consistency across RGB, Mask, Depth, and Height.
    
    additional_targets are declared as 'mask' so that Photometric changes 
    (ColorJitter, Brightness) only affect the RGB 'image', and NOT the spatial maps.
    """
    return A.Compose([
        # 1. Geometric Trandformations (Applied perfectly to ALL channels)
        A.Resize(height=img_size[0], width=img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.0625, 0.0625),
            rotate=(-10, 10),
            p=0.5
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2), # Perspective changes

        # 2. Photometric Transformations (Applied ONLY to RGB 'image')
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.3),
        
        # 3. LIDAR-specific Noise / Dropped packets Simulation
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 20), hole_width_range=(8, 20), p=0.2),

        # 4. Normalization
        # Normalize RGB. Depth and Height are likely already bounded or scaled in preprocessing
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        ToTensorV2()
    ], additional_targets={
        'depth': 'mask',
        'height': 'mask'
    })

def get_val_transforms(img_size=(512, 512)):
    """
    Validation/Test pipeline: Just resize and normalize.
    """
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={
        'depth': 'mask',
        'height': 'mask'
    })
