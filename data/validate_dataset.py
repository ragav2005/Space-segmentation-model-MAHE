import sys
import time
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from config.config import DATASET_DIR, CAMERAS, BATCH_SIZE, NUM_WORKERS, IMG_SIZE
from data.augmentations import get_train_transforms
from data.dataset import MultiModalDrivableDataset

def validate_dataloader():
    print("--- Phase 3: Validating Data Loader & Augmentation ---")
    
    # 1. Initialize Transform
    train_transform = get_train_transforms(IMG_SIZE)
    
    train_split = DATASET_DIR / 'splits' / 'train.txt'
    if not train_split.exists():
        print(f" Cannot find split file at {train_split}")
        return

    # 2. Init Dataset
    dataset = MultiModalDrivableDataset(
        data_root=DATASET_DIR,
        split_file=str(train_split),
        cameras=CAMERAS,
        transform=train_transform
    )
    
    print(f"Total Samples format checked: {len(dataset)}")
    
    if len(dataset) == 0:
        print(" Dataset length is 0. Check your paths.")
        return

    # 3. Test single getitem speed and shape
    start_time = time.time()
    fused_input, mask = dataset[0]
    get_time = time.time() - start_time
    
    print(f"Single Sample Retrieval: {get_time*1000:.2f} ms")
    print(f"Input Shape: {fused_input.shape} (Expected: [5, {IMG_SIZE[0]}, {IMG_SIZE[1]}])")
    print(f"Mask Shape: {mask.shape} (Expected: [{IMG_SIZE[0]}, {IMG_SIZE[1]}])")
    print(f"Input Range: min={fused_input.min().item():.4f}, max={fused_input.max().item():.4f}")
    print(f"Mask Unique: {torch.unique(mask).tolist()}")
    
    assert fused_input.shape[0] == 5, " Fused input doesn't have 5 channels!"
    print(" Shape validation passed!")

    # 4. DataLoader testing (RTX 3050 Optimized)
    print("\n--- Validating DataLoader Speed ---")
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True,     # crucial for GPU transfer
        prefetch_factor=2    # preload batches
    )
    
    # Measure time to fetch a batch
    start_time = time.time()
    for i, (batch_inputs, batch_masks) in enumerate(loader):
        batch_time = time.time() - start_time
        print(f"Batch 1 Loaded in {batch_time*1000:.2f} ms")
        print(f"Batch Input Shape: {batch_inputs.shape}")
        print(f"Batch Mask Shape: {batch_masks.shape}")
        
        # Test dtype consistency
        assert batch_inputs.dtype == torch.float32, "Inputs should be float32"
        assert batch_masks.dtype == torch.int64, "Masks should be int64 (long) for cross entropy"
        assert batch_inputs.shape[1] == 5, "Batch input must contain 5 channels"
        print("Dtype validation passed!")
        print("DataLoader behaves correctly!")
        break # Only test one batch

if __name__ == '__main__':
    validate_dataloader()
