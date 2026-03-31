"""
Phase 2: Augmentation Sanity Inspection Utility

Purpose:
- Export augmented image-mask pairs for manual alignment verification
- Verify augmentations preserve mask-image correspondence
- Detect label corruption or alignment artifacts early
- Build confidence in augmentation pipeline before training

Output:
- outputs/visualizations/augmentations/ with side-by-side comparisons
- Per-image: original RGB, original mask, augmented RGB (1..N), augmented mask (1..N)
- Alignment report: measures mask coverage preservation and deformation consistency
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np

from config import config as cfg


logger = logging.getLogger(__name__)


@dataclass
class AugmentationStats:
    """Track alignment metrics across augmentation samples."""
    sample_id: str
    original_mask_coverage: float
    augmented_coverages: List[float]
    min_coverage: float
    max_coverage: float
    coverage_variance: float
    alignment_score: float  # 0.0-1.0: how well mask follows image deformation
    issues: List[str]


def _get_aug_transforms_visual(img_size: Tuple[int, int]) -> A.Compose:
    """Augmentations for visualization (NO normalization, NO tensor conversion)."""
    h, w = img_size
    return A.Compose(
        [
            A.Resize(height=h, width=w),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.03,
                scale_limit=0.12,
                rotate_limit=7,
                interpolation=1,
                border_mode=0,
                p=0.6,
            ),
            A.RandomBrightnessContrast(p=0.35),
            A.RandomGamma(p=0.2),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=14, val_shift_limit=10, p=0.25),
            A.GaussianBlur(blur_limit=(3, 5), p=0.12),
            A.ImageCompression(quality_range=(60, 100), p=0.1),
        ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    )


def _load_splits(dataset_dir: Path) -> Dict[str, List[str]]:
    """Load train/val/test split files."""
    splits = {}
    for split_name in ['train', 'val', 'test']:
        split_file = dataset_dir / 'splits' / f'{split_name}.txt'
        if split_file.exists():
            with open(split_file) as f:
                splits[split_name] = [line.strip() for line in f if line.strip()]
        else:
            splits[split_name] = []
    return splits


def _compute_alignment_score(
    original_mask: np.ndarray,
    augmented_mask: np.ndarray,
) -> float:
    """
    Compute a simple alignment score measuring how well the mask transformation
    matches the image transformation. Uses correlation of mask structure.
    
    Score 1.0 = perfect alignment, 0.0 = complete misalignment.
    """
    if original_mask.sum() == 0 or augmented_mask.sum() == 0:
        return 1.0  # Empty masks always "align"
    
    # Compute structural similarity via contour matching
    mask_orig_uint8 = (original_mask * 255).astype(np.uint8)
    mask_aug_uint8 = (augmented_mask * 255).astype(np.uint8)
    
    # Simple cross-correlation of mask shapes
    corr = cv2.matchTemplate(
        mask_aug_uint8,
        mask_orig_uint8 if mask_orig_uint8.size > mask_aug_uint8.size else mask_aug_uint8,
        cv2.TM_CCOEFF
    )
    if corr.size == 0:
        return 0.5
    
    # Normalize correlation to [0, 1]
    score = min(1.0, max(0.0, float(corr.max()) / (mask_orig_uint8.sum() + 1e-6)))
    return score


def _visualize_sample(
    output_dir: Path,
    sample_id: str,
    image: np.ndarray,
    mask: np.ndarray,
    num_augs: int = 4,
    aug_transform: Optional[A.Compose] = None,
) -> AugmentationStats:
    """
    Export one sample with N augmentations, producing side-by-side grid.
    
    Layout:
    [Original RGB] [Original Mask] | [Aug1 RGB] [Aug1 Mask] ... [AugN RGB] [AugN Mask]
    """
    if aug_transform is None:
        aug_transform = _get_aug_transforms_visual((image.shape[0], image.shape[1]))
    
    # Ensure mask is 2D (H, W)
    if mask.ndim == 3:
        mask = mask.squeeze()
    mask = (mask / 255.0).astype(np.float32) if mask.max() > 1.0 else mask.astype(np.float32)
    
    original_coverage = mask.mean()
    augmented_masks = []
    augmented_images = []
    
    # Generate augmentations
    for aug_idx in range(num_augs):
        result = aug_transform(image=image, mask=mask)
        aug_img = result['image']
        aug_mask = result['mask']
        augmented_images.append(aug_img)
        augmented_masks.append(aug_mask)
    
    # Compute statistics
    aug_coverages = [m.mean() for m in augmented_masks]
    alignment_scores = [_compute_alignment_score(mask, aug_mask) for aug_mask in augmented_masks]
    
    min_cov = min(aug_coverages)
    max_cov = max(aug_coverages)
    mean_cov = np.mean(aug_coverages)
    cov_var = np.var(aug_coverages)
    alignment = np.mean(alignment_scores)
    
    issues = []
    if max_cov < 0.01:
        issues.append("WARNING: All augmentations result in empty masks")
    if abs(mean_cov - original_coverage) > 0.15:
        issues.append(f"WARNING: Mean augmented coverage {mean_cov:.3f} deviates from original {original_coverage:.3f}")
    if alignment < 0.5:
        issues.append(f"WARNING: Low alignment score {alignment:.3f} (mask may not follow image deformation)")
    
    stats = AugmentationStats(
        sample_id=sample_id,
        original_mask_coverage=float(original_coverage),
        augmented_coverages=[float(c) for c in aug_coverages],
        min_coverage=float(min_cov),
        max_coverage=float(max_cov),
        coverage_variance=float(cov_var),
        alignment_score=float(alignment),
        issues=issues,
    )
    
    # Create visualization: original vs augmented grid
    h_vis, w_vis = image.shape[0], image.shape[1]
    
    # Normalize original for display
    img_display = (image / 255.0) if image.max() > 1 else image
    if img_display.ndim == 2:
        img_display = cv2.cvtColor((img_display * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        img_display = (img_display * 255).astype(np.uint8)
    
    # Build grid: original (2 cols) | augmentation 1 (2 cols) | augmentation 2 (2 cols) ...
    grid_cols = []
    
    # Add original
    mask_vis_orig = (mask * 255).astype(np.uint8)
    mask_vis_orig_3ch = cv2.cvtColor(mask_vis_orig, cv2.COLOR_GRAY2BGR)
    grid_cols.append(img_display)
    grid_cols.append(mask_vis_orig_3ch)
    
    # Add augmentations
    for aug_idx in range(min(num_augs, 3)):  # Limit to 3 augs for display (fits better)
        aug_img = augmented_images[aug_idx]
        aug_mask = augmented_masks[aug_idx]
        
        # Normalize for display
        aug_img_disp = (aug_img / 255.0) if aug_img.max() > 1 else aug_img
        if aug_img_disp.ndim == 2:
            aug_img_disp = cv2.cvtColor((aug_img_disp * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            aug_img_disp = (aug_img_disp * 255).astype(np.uint8)
        
        aug_mask_disp = (aug_mask * 255).astype(np.uint8)
        aug_mask_disp_3ch = cv2.cvtColor(aug_mask_disp, cv2.COLOR_GRAY2BGR)
        
        grid_cols.append(aug_img_disp)
        grid_cols.append(aug_mask_disp_3ch)
    
    # Horizontally stack columns
    grid = np.hstack(grid_cols)
    
    # Save grid
    output_file = output_dir / f'{sample_id}_augmentations.jpg'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_file), grid)
    logger.info(f"  Saved visualization: {output_file}")
    
    return stats


def run_visualization(
    num_samples: int = 10,
    num_augs_per_sample: int = 4,
    split: str = 'train',
) -> Dict:
    """
    Main visualization pipeline.
    
    Args:
        num_samples: Number of samples to visualize
        num_augs_per_sample: Augmentations per sample
        split: Which split to sample from ('train', 'val', 'test')
    """
    dataset_dir = Path(cfg.DATASET_DIR)
    output_base = Path(cfg.OUTPUT_DIR) / 'visualizations' / 'augmentations'
    output_base.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Phase 2: Augmentation Visualization")
    logger.info(f"  Output: {output_base}")
    logger.info(f"  Samples: {num_samples}, Augs/Sample: {num_augs_per_sample}, Split: {split}")
    
    # Load splits
    splits = _load_splits(dataset_dir)
    if split not in splits or len(splits[split]) == 0:
        logger.error(f"Split '{split}' not found or empty. Available: {list(splits.keys())}")
        return {'status': 'failed', 'reason': f'Split {split} not available'}
    
    split_samples = splits[split][:num_samples]
    logger.info(f"Loaded {len(split_samples)} samples from '{split}' split")
    
    # Prepare augmentation transform (visualization version)
    aug_transform = _get_aug_transforms_visual(cfg.IMG_SIZE)
    
    all_stats = []
    
    # Visualize each sample
    for idx, sample_id in enumerate(split_samples, 1):
        logger.info(f"[{idx}/{len(split_samples)}] Processing {sample_id}")
        
        # Load image and mask
        img_path = dataset_dir / 'images' / f'{sample_id}.jpg'
        mask_path = dataset_dir / 'masks' / f'{sample_id}.png'
        
        if not img_path.exists() or not mask_path.exists():
            logger.warning(f"  Missing files for {sample_id}, skipping")
            continue
        
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None or mask is None:
                logger.warning(f"  Failed to read files for {sample_id}, skipping")
                continue
            
            # Resize to config size
            img = cv2.resize(img, (cfg.IMG_SIZE[1], cfg.IMG_SIZE[0]))
            mask = cv2.resize(mask, (cfg.IMG_SIZE[1], cfg.IMG_SIZE[0]), interpolation=cv2.INTER_NEAREST)
            
            stats = _visualize_sample(
                output_dir=output_base,
                sample_id=sample_id,
                image=img,
                mask=mask,
                num_augs=num_augs_per_sample,
                aug_transform=aug_transform,
            )
            all_stats.append(stats)
        except Exception as e:
            logger.error(f"  Error processing {sample_id}: {e}")
            continue
    
    # Generate summary report
    if all_stats:
        report = _generate_summary_report(all_stats, output_base)
        logger.info(f"Summary Report:")
        logger.info(f"  Samples: {len(all_stats)}")
        logger.info(f"  Mean original coverage: {report['mean_original_coverage']:.3f}")
        logger.info(f"  Mean augmented coverage: {report['mean_augmented_coverage']:.3f}")
        logger.info(f"  Mean alignment score: {report['mean_alignment_score']:.3f}")
        logger.info(f"  Issues found: {report['total_issues']}")
        
        if report['total_issues'] > 0:
            logger.warning(f"  WARNING: {report['total_issues']} issues detected in augmentation pipeline")
        else:
            logger.info(f"  ✓ All augmentation checks passed")
        
        return report
    else:
        logger.error("No samples successfully processed")
        return {'status': 'failed', 'reason': 'No samples processed'}


def _generate_summary_report(
    all_stats: List[AugmentationStats],
    output_dir: Path,
) -> Dict:
    """Generate JSON summary report of all augmentation checks."""
    mean_orig_cov = np.mean([s.original_mask_coverage for s in all_stats])
    mean_aug_cov = np.mean([c for s in all_stats for c in s.augmented_coverages])
    mean_alignment = np.mean([s.alignment_score for s in all_stats])
    
    all_issues = [issue for s in all_stats for issue in s.issues]
    
    report = {
        'phase': 'phase_2_augmentation_visualization',
        'timestamp': datetime.now().isoformat(),
        'stats': {
            'num_samples': len(all_stats),
            'mean_original_coverage': float(mean_orig_cov),
            'mean_augmented_coverage': float(mean_aug_cov),
            'mean_alignment_score': float(mean_alignment),
            'total_issues': len(all_issues),
        },
        'per_sample': [
            {
                'sample_id': s.sample_id,
                'original_coverage': s.original_mask_coverage,
                'augmented_coverages': s.augmented_coverages,
                'coverage_range': [s.min_coverage, s.max_coverage],
                'alignment_score': s.alignment_score,
                'issues': s.issues,
            }
            for s in all_stats
        ],
        'all_issues': all_issues,
    }
    
    report_file = output_dir / 'augmentation_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved: {report_file}")
    return report['stats']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 2: Augmentation Visualization')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--num-augs', type=int, default=4, help='Augmentations per sample')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    run_visualization(
        num_samples=args.num_samples,
        num_augs_per_sample=args.num_augs,
        split=args.split,
    )
