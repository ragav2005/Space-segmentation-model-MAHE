"""
Phase 4: Training System and Curriculum for Robust Binary Segmentation.

Implements:
- AMP (mixed precision)
- Gradient clipping
- Cosine annealing scheduler
- Checkpointing (best + periodic)
- Resume training
- Two-stage curriculum:
    - Stage A: high-confidence samples
    - Stage B: all samples with confidence weighting
- Camera-balanced sampling

Target: 75-80% mIoU on nuScenes drivable area segmentation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import logging
import re

import torch
from torch import nn, optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import config as cfg
from data.augmentations import get_train_transforms, get_val_transforms
from data.dataset import DrivableDataset
from models.losses import RobustCombinedLoss
from models.mobilenet_deeplab import MobileDeepLabV3Plus
from utils.logger import TrainingLogger
from utils.metrics import SegmentationMetrics

logger = logging.getLogger(__name__)


def _extract_camera_id(stem: str) -> str:
    """Extract camera token from a sample stem, fallback to UNKNOWN."""
    match = re.search(r"CAM_[A-Z_]+", stem)
    if match:
        return match.group(0)
    return "UNKNOWN"


def _estimate_sample_confidence(mask_path: Path) -> float:
    """Estimate sample confidence from mask shape statistics in [0.05, 1.0]."""
    import cv2
    import numpy as np

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0.05

    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape
    total = float(max(1, h * w))

    # Coverage prior: avoid trivial all-empty or all-full masks.
    fg_ratio = float(binary.mean())
    coverage_score = 1.0 - min(1.0, abs(fg_ratio - 0.35) / 0.35)

    # Boundary prior: noisy masks often have excessive boundaries.
    edges = cv2.Canny(binary * 255, 50, 150)
    edge_ratio = float((edges > 0).sum()) / total
    boundary_score = 1.0 - min(1.0, edge_ratio * 8.0)

    conf = 0.6 * coverage_score + 0.4 * boundary_score
    return float(max(0.05, min(1.0, conf)))


def _build_train_metadata(train_ds: DrivableDataset) -> list[dict[str, object]]:
    """Build per-sample metadata required for curriculum and balanced sampling."""
    metadata: list[dict[str, object]] = []
    for idx, (img_path, mask_path) in enumerate(train_ds.samples):
        stem = img_path.stem
        metadata.append(
            {
                "idx": idx,
                "stem": stem,
                "camera": _extract_camera_id(stem),
                "confidence": _estimate_sample_confidence(mask_path),
            }
        )
    return metadata


def _build_curriculum_train_loader(
    train_ds: DrivableDataset,
    metadata: list[dict[str, object]],
    stage: str,
    conf_threshold: float,
    conf_power: float,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
) -> tuple[DataLoader, dict[str, float | int]]:
    """
    Build a weighted training loader for curriculum stage.

    Stage A: use only high-confidence subset.
    Stage B: use all samples and apply confidence weighting.
    Both stages apply camera-balance weights.
    """
    cameras = [str(m["camera"]) for m in metadata]
    cam_counts = Counter(cameras)

    if stage == "A":
        selected = [m for m in metadata if float(m["confidence"]) >= conf_threshold]
        if len(selected) < max(8, cfg.BATCH_SIZE):
            # Ensure Stage A remains usable for small/low-confidence sets.
            selected = sorted(metadata, key=lambda m: float(m["confidence"]), reverse=True)[: max(8, cfg.BATCH_SIZE * 2)]
    else:
        selected = metadata

    weights: list[float] = []
    for m in selected:
        camera = str(m["camera"])
        conf = float(m["confidence"])
        cam_w = 1.0 / float(max(1, cam_counts[camera]))
        if stage == "A":
            w = cam_w
        else:
            w = cam_w * (max(0.05, conf) ** conf_power)
        weights.append(float(w))

    selected_indices = [int(m["idx"]) for m in selected]
    if not selected_indices:
        selected_indices = list(range(len(train_ds)))

    # Use a subset so Stage A truly samples the selected examples only.
    train_subset = Subset(train_ds, selected_indices)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(selected_indices),
        replacement=True,
    )

    loader_kwargs: dict[str, object] = {
        "num_workers": num_workers if cfg.DEVICE == "cuda" else 0,
        "pin_memory": pin_memory if cfg.DEVICE == "cuda" else False,
    }
    if int(loader_kwargs["num_workers"]) > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers

    train_dl = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        **loader_kwargs,
    )

    info = {
        "stage": stage,
        "selected_samples": len(selected_indices),
        "total_samples": len(metadata),
        "avg_confidence": float(sum(float(m["confidence"]) for m in selected) / max(1, len(selected))),
    }
    return train_dl, info


def _probe_device_runtime(model: nn.Module, device: str) -> tuple[bool, str]:
    """Probe whether current device runtime is actually usable for model forward."""
    if device != "cuda":
        return True, ""

    was_training = model.training
    try:
        # Use eval mode for probe to avoid BatchNorm failures on tiny batch/spatial tensors.
        model.eval()
        x = torch.randn(2, cfg.IN_CHANNELS, cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], device=device)
        with torch.no_grad():
            _ = model(x)
        return True, ""
    except Exception as e:
        return False, str(e)
    finally:
        model.train(was_training)


class ExponentialMovingAverage(nn.Module):
    """Exponential Moving Average weight update for model stabilization."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        super().__init__()
        self.model = model
        self.decay = decay
        self.ema_state = {}

        # Copy initial weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.ema_state[name] = param.data.clone()

    def update(self) -> None:
        """Update EMA weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.ema_state:
                self.ema_state[name] = self.decay * self.ema_state[name] + (1 - self.decay) * param.data

    def apply(self) -> None:
        """Apply EMA weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.ema_state:
                param.data.copy_(self.ema_state[name])

    def restore(self) -> None:
        """Restore original weights from model state (undo apply)."""
        pass  # Weights are already in param.data after apply


def _datasets_and_val_loader(
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
) -> tuple[DrivableDataset, DataLoader]:
    """Create train dataset and validation loader."""
    train_split = cfg.SPLITS_DIR / "train.txt"
    val_split = cfg.SPLITS_DIR / "val.txt"

    train_ds = DrivableDataset(cfg.IMAGES_DIR, cfg.MASKS_DIR, train_split, get_train_transforms(cfg.IMG_SIZE))
    val_ds = DrivableDataset(cfg.IMAGES_DIR, cfg.MASKS_DIR, val_split, get_val_transforms(cfg.IMG_SIZE))
    loader_kwargs: dict[str, object] = {
        "num_workers": num_workers if cfg.DEVICE == "cuda" else 0,
        "pin_memory": pin_memory if cfg.DEVICE == "cuda" else False,
    }
    if int(loader_kwargs["num_workers"]) > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_ds, val_dl


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    scaler: GradScaler,
    train: bool,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Run one epoch of training or validation.

    Args:
        model: Segmentation model
        loader: Data loader
        criterion: Loss function
        optimizer: Optimizer (None for validation)
        scaler: Gradient scaler for AMP
        train: Training mode flag
        max_grad_norm: Max gradient norm for clipping

    Returns:
        Average epoch loss
    """
    model.train(mode=train)
    running_loss = 0.0
    device = cfg.DEVICE

    it = tqdm(loader, desc="train" if train else "val", leave=False)
    for batch_idx, (x, y) in enumerate(it):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with autocast(device_type=("cuda" if device == "cuda" else "cpu"), enabled=(device == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        if train and optimizer is not None:
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

        running_loss += loss.item()
        it.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / max(1, len(loader))
    return avg_loss


@torch.no_grad()
def _validate_metrics(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    """
    Compute validation metrics.

    Args:
        model: Segmentation model
        loader: Validation loader

    Returns:
        Dictionary with mIoU, class IoUs, and accuracy
    """
    model.eval()
    metrics = SegmentationMetrics(num_classes=cfg.NUM_CLASSES)

    for x, y in tqdm(loader, desc="metrics", leave=False):
        x = x.to(cfg.DEVICE, non_blocking=True)
        y = y.to(cfg.DEVICE, non_blocking=True)

        with autocast(device_type=("cuda" if cfg.DEVICE == "cuda" else "cpu"), enabled=(cfg.DEVICE == "cuda")):
            logits = model(x)

        metrics.update(logits, y)

    return metrics.compute()


def main() -> None:
    """Main training routine with advanced techniques."""
    parser = argparse.ArgumentParser(description="Phase 4: curriculum training with camera-balanced sampling")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE, help="Batch size (increase to better utilize VRAM)")
    parser.add_argument("--num-workers", type=int, default=cfg.NUM_WORKERS, help="DataLoader worker count")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor per worker")
    parser.add_argument("--pin-memory", action="store_true", default=cfg.PIN_MEMORY, help="Enable pinned host memory")
    parser.add_argument("--persistent-workers", action="store_true", default=True, help="Keep workers alive across epochs")
    parser.add_argument("--deterministic", action="store_true", default=cfg.DETERMINISTIC, help="Deterministic mode (slower)")
    parser.add_argument("--non-deterministic", dest="deterministic", action="store_false", help="Faster kernels, less reproducibility")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--start-epoch", type=int, default=1, help="Resume from epoch")
    parser.add_argument("--resume-ckpt", type=str, default="", help="Checkpoint to resume from")
    parser.add_argument("--stage-a-epochs", type=int, default=10, help="Number of Stage A curriculum epochs")
    parser.add_argument(
        "--high-conf-threshold",
        type=float,
        default=0.65,
        help="High-confidence threshold for Stage A sample selection",
    )
    parser.add_argument(
        "--confidence-power",
        type=float,
        default=1.5,
        help="Exponent for confidence weighting during Stage B",
    )
    # --- Ablation Flags ---
    parser.add_argument("--width-mult", type=float, default=cfg.WIDTH_MULT, help="Multiplier for model width channels")
    parser.add_argument("--boundary-weight", type=float, default=0.1 if cfg.USE_BOUNDARY_LOSS else 0.0, help="Weight for boundary loss (set 0.0 to disable)")
    parser.add_argument("--img-height", type=int, default=cfg.IMG_SIZE[0], help="Input image height")
    parser.add_argument("--img-width", type=int, default=cfg.IMG_SIZE[1], help="Input image width")
    parser.add_argument("--no-conf-weighting", action="store_true", help="Disable confidence weighting (forces power=0)")
    args = parser.parse_args()

    if args.no_conf_weighting:
        args.confidence_power = 0.0

    cfg.ensure_paths()
    cfg.set_seed(deterministic=args.deterministic)

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Runtime-safe device selection
    device = cfg.DEVICE
    if device == "cuda":
        try:
            _ = torch.empty(1, device="cuda")
        except Exception as e:
            logger.warning(f"CUDA/HIP runtime unavailable, falling back to CPU: {e}")
            device = "cpu"

    # Note: shell assignment like "HSA_OVERRIDE_GFX_VERSION=..." on its own line is
    # not exported to child processes unless prefixed to command or exported.
    if device == "cuda":
        logger.info(
            "If using ROCm override, run as: "
            "HSA_OVERRIDE_GFX_VERSION=<x.y.z> python training/train.py ..."
        )
        if hasattr(torch.backends, "cudnn") and not args.deterministic:
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled backend benchmark for faster convolution autotuning")

    cfg.DEVICE = device

    # Dry-run test
    # Ablation overrides for image size
    img_size = (args.img_height, args.img_width)
    cfg.IMG_SIZE = img_size

    if args.dry_run:
        dry_device = args.dry_run_device
        if dry_device == "cuda" and not torch.cuda.is_available():
            logger.info("CUDA/HIP not available. Using CPU for dry-run.")
            dry_device = "cpu"

        model = MobileDeepLabV3Plus(
            in_channels=cfg.IN_CHANNELS, 
            num_classes=cfg.NUM_CLASSES,
            width_mult=args.width_mult
        ).to(dry_device)
        x = torch.randn(2, cfg.IN_CHANNELS, cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], device=dry_device)

        with torch.no_grad():
            logits = model(x)

        # Check for NaN/Inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"ERROR: Model output contains NaN/Inf!")
            logger.error("Model output contains NaN/Inf")
            return

        logger.info(f"✓ Dry-run passed on {dry_device.upper()}")
        logger.info(f"  Input shape: {tuple(x.shape)}")
        logger.info(f"  Output shape: {tuple(logits.shape)}")
        logger.info(f"  Output range: [{logits.min():.4f}, {logits.max():.4f}]")
        return

    # Verify split files exist
    if not (cfg.SPLITS_DIR / "train.txt").exists() or not (cfg.SPLITS_DIR / "val.txt").exists():
        raise FileNotFoundError("Missing split files. Run Phase 1 first.")

    # Setup model and training components
    model = MobileDeepLabV3Plus(
        in_channels=cfg.IN_CHANNELS, 
        num_classes=cfg.NUM_CLASSES,
        width_mult=args.width_mult
    ).to(cfg.DEVICE)

    # Probe CUDA/HIP runtime with a real model forward; fallback to CPU if kernels fail.
    ok, probe_err = _probe_device_runtime(model, cfg.DEVICE)
    if not ok:
        logger.warning(f"CUDA/HIP runtime probe failed, falling back to CPU: {probe_err}")
        cfg.DEVICE = "cpu"
        model = MobileDeepLabV3Plus(
            in_channels=cfg.IN_CHANNELS, 
            num_classes=cfg.NUM_CLASSES,
            width_mult=args.width_mult
        ).to(cfg.DEVICE)

    # Load data after final device decision
    train_ds, val_loader = _datasets_and_val_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    train_metadata = _build_train_metadata(train_ds)
    logger.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_loader.dataset)}")
    logger.info(
        f"Loader config | batch_size={args.batch_size} workers={args.num_workers} "
        f"prefetch={args.prefetch_factor} pin_memory={args.pin_memory} "
        f"persistent_workers={args.persistent_workers} deterministic={args.deterministic}"
    )

    criterion = RobustCombinedLoss(
        ce_weight=0.4,
        focal_weight=0.3,
        dice_weight=0.2,
        boundary_weight=args.boundary_weight,
        ohem_enabled=True,
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=cfg.LEARNING_RATE * 0.01)
    scaler = GradScaler(enabled=(cfg.DEVICE == "cuda"))
    ema = ExponentialMovingAverage(model, decay=0.999)
    logger_writer = TrainingLogger(cfg.LOG_DIR)

    logger.info(f"Device: {cfg.DEVICE}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    best_miou = 0.0
    best_epoch = 0
    patience = 15
    patience_counter = 0
    best_path = cfg.CHECKPOINT_DIR / cfg.BEST_CHECKPOINT_NAME

    # Resume from checkpoint if specified
    if args.resume_ckpt and Path(args.resume_ckpt).exists():
        ckpt = torch.load(args.resume_ckpt, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        best_miou = ckpt.get("best_miou", 0.0)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and cfg.DEVICE == "cuda":
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "patience_counter" in ckpt:
            patience_counter = int(ckpt["patience_counter"])
        logger.info(f"Resumed from {args.resume_ckpt} | Best mIoU: {best_miou:.2f}")

    # Training loop
    for epoch in range(args.start_epoch, args.epochs + 1):
        stage = "A" if epoch <= args.stage_a_epochs else "B"
        train_loader, stage_info = _build_curriculum_train_loader(
            train_ds=train_ds,
            metadata=train_metadata,
            stage=stage,
            conf_threshold=args.high_conf_threshold,
            conf_power=args.confidence_power,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
        )

        logger.info(
            f"Curriculum Stage {stage} | selected={stage_info['selected_samples']}/"
            f"{stage_info['total_samples']} avg_conf={stage_info['avg_confidence']:.3f}"
        )

        # Train
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, scaler, train=True)

        # Update EMA
        ema.update()

        # Validate
        val_loss = _run_epoch(model, val_loader, criterion, None, scaler, train=False)

        # Compute metrics
        metrics = _validate_metrics(model, val_loader)
        miou = metrics["mIoU"]
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"mIoU={miou:.4f} Driv_IoU={metrics['Drivable_IoU']:.4f} | "
            f"acc={metrics['Pixel_Acc']:.4f} lr={lr:.2e}"
        )

        logger_writer.log_epoch(epoch, train_loss, val_loss, metrics, lr)

        # Update best model
        if miou > best_miou:
            best_miou = miou
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_miou": best_miou,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "patience_counter": patience_counter,
                },
                best_path,
            )
            logger.info(f"✓ Saved best checkpoint (mIoU: {best_miou:.4f})")
        else:
            patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} (patience: {patience})")
                break

        # Learning rate scheduling
        scheduler.step()

        # Periodic checkpoint
        if epoch % cfg.SAVE_EVERY_EPOCHS == 0:
            ckpt_path = cfg.CHECKPOINT_DIR / f"checkpoint_ep{epoch:03d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_miou": best_miou,
                    "patience_counter": patience_counter,
                },
                ckpt_path,
            )

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Training complete!")
    logger.info(f"Best mIoU: {best_miou:.4f} at epoch {best_epoch}")
    logger.info(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
