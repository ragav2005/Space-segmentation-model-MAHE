from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DrivableDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        split_file: Path,
        transform: Callable | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        with open(split_file, "r", encoding="utf-8") as f:
            stems = [line.strip() for line in f if line.strip()]

        samples: List[Tuple[Path, Path]] = []
        for stem in stems:
            img = self.images_dir / f"{stem}.jpg"
            msk = self.masks_dir / f"{stem}.png"
            if img.exists() and msk.exists():
                samples.append((img, msk))

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Missing image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _read_mask(path: Path) -> np.ndarray:
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Missing mask: {path}")
        return (m > 0).astype(np.uint8)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]
        image = self._read_image(img_path)
        mask = self._read_mask(mask_path)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image_t = aug["image"].float()
            mask_t = aug["mask"].long()
            if mask_t.ndim == 3:
                mask_t = mask_t.squeeze(0)
            return image_t, mask_t

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy(mask).long()
        return image_t, mask_t
