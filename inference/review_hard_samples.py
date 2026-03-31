import argparse
import os
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import Rellis3DDataset
from models.mobilenet_deeplab import MobileNetV3DeepLabV3
from config.config import CFG

def calculate_iou(pred, target, num_classes=5):
    ious = []
    for cls in range(num_classes):
        if cls == 0:  # ignore void
            continue
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    if len(valid_ious) == 0:
        return float('nan')
    return np.mean(valid_ious)

def main():
    parser = argparse.ArgumentParser("Review Hard Samples")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate")
    parser.add_argument("--frac", type=float, default=0.1, help="Fraction of worst samples to flag")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = Rellis3DDataset(split=args.split, transforms=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Load model
    model = MobileNetV3DeepLabV3(num_classes=CFG.num_classes)
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return
        
    state_dict = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()

    sample_scores = []
    
    print("Evaluating samples to find hard cases...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # calculate mIoU for the sample
            pred_np = preds[0].cpu().numpy()
            mask_np = masks[0].cpu().numpy()
            
            miou = calculate_iou(pred_np, mask_np, CFG.num_classes)
            if np.isnan(miou):
                miou = 1.0 # Ignore empty targets
                
            sample_scores.append({
                "index": i,
                "image_path": dataset.image_paths[i],
                "mask_path": dataset.mask_paths[i],
                "miou": miou
            })

    # Sort by mIoU ascending (worst first)
    sample_scores.sort(key=lambda x: x["miou"])
    
    num_hard = int(len(sample_scores) * args.frac)
    hard_samples = sample_scores[:num_hard]
    
    out_dir = "outputs/hard_samples"
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, f"hard_samples_{args.split}.txt"), "w") as f:
        for s in hard_samples:
            f.write(f"{s['image_path']}, mIoU: {s['miou']:.4f}\n")
            
    print(f"\nIdentified top {num_hard} hard samples (worst mIoU).")
    print(f"List saved to {out_dir}/hard_samples_{args.split}.txt")
    print("Sample of worst 5:")
    for i in range(min(5, len(hard_samples))):
        print(f"  {hard_samples[i]['image_path']} - mIoU: {hard_samples[i]['miou']:.4f}")

if __name__ == "__main__":
    main()
