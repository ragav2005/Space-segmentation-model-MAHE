from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import config as cfg
from data.augmentations import get_val_transforms
from models.mobilenet_deeplab import MobileDeepLabV3Plus


def predict(
    ckpt_path: str,
    input_path: str,
    output_dir: str,
    fp16: bool = True
) -> None:
    device = cfg.DEVICE
    model = MobileDeepLabV3Plus(in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES).to(device)
    
    if not Path(ckpt_path).exists():
        print(f"Error: Checkpoint {ckpt_path} not found.")
        sys.exit(1)
        
    print(f"Loading weights from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()

    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)

    input_p = Path(input_path)
    if input_p.is_dir():
        # Predict on all jpg/png files
        image_files = list(input_p.glob("*.jpg")) + list(input_p.glob("*.png"))
    else:
        image_files = [input_p]

    if not image_files:
        print(f"No valid image files found in {input_path}")
        sys.exit(1)

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    
    transform = get_val_transforms(cfg.IMG_SIZE)

    print(f"Running inference on {len(image_files)} image(s)...")

    # Colors for overlay (BGR for OpenCV)
    # Class 1 (Drivable): Green mask overlay
    color_drivable = np.array([0, 255, 0], dtype=np.uint8)

    with torch.no_grad():
        for img_file in image_files:
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"Failed to read {img_file}")
                continue
                
            orig_h, out_w = img.shape[:2]
            
            # Albumentations expects RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            aug = transform(image=img_rgb)
            img_tensor = aug["image"].unsqueeze(0).to(device)
            
            if device == "cuda":
                img_tensor = img_tensor.to(memory_format=torch.channels_last)

            # Inference
            with torch.autocast(device_type=device, enabled=(device == "cuda" and fp16)):
                logits = model(img_tensor)
            
            # Predict
            preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            
            # Resize pred back to original size for beautiful overlay
            # Convert to float for linear interpolation to get a smooth transition
            preds_float = preds.astype(np.float32)
            preds_resized = cv2.resize(preds_float, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # Apply Gaussian Blur to feather the edges
            preds_resized = cv2.GaussianBlur(preds_resized, (15, 15), 0)
            
            # Make sure it is bound between 0 and 1
            preds_resized = np.clip(preds_resized, 0, 1)

            # Make overlay
            overlay = img.copy()
            
            # Create a full color mask
            color_mask = np.zeros_like(img)
            color_mask[:] = color_drivable
            
            # Alpha blending the whole image where mask exists
            blended = cv2.addWeighted(overlay, 0.5, color_mask, 0.5, 0)
            
            # Blend smoothly into overlay based on the float mask
            alpha = preds_resized[..., np.newaxis]
            overlay = (alpha * blended + (1 - alpha) * overlay).astype(np.uint8)
            
            out_file = out_p / f"{img_file.stem}_pred.jpg"
            cv2.imwrite(str(out_file), overlay)
            print(f"Saved: {out_file}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on an image or directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output", "-o", type=str, default="outputs/predictions", help="Directory to save outputs")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 inference")
    args = parser.parse_args()
    
    predict(
        ckpt_path=args.checkpoint,
        input_path=args.input,
        output_dir=args.output,
        fp16=not args.no_fp16
    )

if __name__ == "__main__":
    main()
