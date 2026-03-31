from __future__ import annotations

from typing import Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _normalize() -> A.Normalize:
    return A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)


def get_train_transforms(img_size: Tuple[int, int]) -> A.Compose:
    h, w = img_size
    return A.Compose(
        [
            A.Resize(height=h, width=w),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                scale=(1.0 - 0.12, 1.0 + 0.12),
                rotate=(-7, 7),
                interpolation=1, # cv2.INTER_LINEAR
                p=0.6,
            ),
            A.RandomBrightnessContrast(p=0.35),
            A.RandomGamma(p=0.2),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=14, val_shift_limit=10, p=0.25),
            A.GaussianBlur(blur_limit=(3, 5), p=0.12),
            A.ImageCompression(quality_range=(60, 100), p=0.1),
            _normalize(),
            ToTensorV2(transpose_mask=True),
        ]
    )


def get_val_transforms(img_size: Tuple[int, int]) -> A.Compose:
    h, w = img_size
    return A.Compose(
        [
            A.Resize(height=h, width=w),
            _normalize(),
            ToTensorV2(transpose_mask=True),
        ]
    )
