from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import config as cfg
from models.mobilenet_deeplab import MobileDeepLabV3Plus

def benchmark(checkpoint_path: str = "", runs: int = 200, warmup: int = 40, batch_size: int = 1, fp16: bool = True) -> None:
    device = cfg.DEVICE
    model = MobileDeepLabV3Plus(in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES).to(device)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading weights from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    
    model.eval()

    x = torch.randn(batch_size, cfg.IN_CHANNELS, cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], device=device)
    
    # Runtime optimization: Channels Last
    if device == "cuda":
        x = x.to(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)

    if device == "cuda" and fp16:
        print("Using FP16 (Half Precision) Inference")
    
    # Optional setup for torch.compile or torch.jit if available, but staying simple as per requirements
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = True

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        with torch.autocast(device_type=device, enabled=(device == "cuda" and fp16)):
            for _ in range(warmup):
                _ = model(x)
            
        if device == "cuda":
            torch.cuda.synchronize()

        print(f"Running benchmark ({runs} iterations)...")
        timings = []
        for _ in range(runs):
            t0 = time.perf_counter()
            with torch.autocast(device_type=device, enabled=(device == "cuda" and fp16)):
                _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    timings.sort()
    p50 = timings[int(0.50 * len(timings))]
    p95 = timings[int(0.95 * len(timings))]
    fps = 1000.0 / p50 if p50 > 0 else 0.0

    print(f"--- Benchmark Results ---")
    print(f"Device: {device}")
    print(f"Resolution: {cfg.IMG_SIZE}")
    print(f"Batch Size: {batch_size}")
    print(f"Channels Last: {device == 'cuda'}")
    print(f"p50 latency: {p50:.2f} ms")
    print(f"p95 latency: {p95:.2f} ms")
    print(f"FPS (p50): {fps:.2f}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Model benchmark")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 inference")
    args = parser.parse_args()
    
    benchmark(
        checkpoint_path=args.checkpoint,
        runs=args.runs, 
        warmup=args.warmup, 
        batch_size=args.batch_size,
        fp16=not args.no_fp16
    )

if __name__ == "__main__":
    main()
