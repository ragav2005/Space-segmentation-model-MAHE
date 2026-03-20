import torch
import time
from models.shufflenet_seg import ShuffleNetSegmentation
from models.losses import CombinedLoss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model():
    print("--- Testing Phase 2 Model Architecture ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on Device: {device}")
    
    # 1. Create Model
    model = ShuffleNetSegmentation(in_channels=5, num_classes=2).to(device)
    params = count_parameters(model)
    print(f"Total Model Parameters: {params:,} (Target: ~3-5M for Real-Time)")
    
    # 2. Dummy Multi-Modal Input (B, 5, H, W) -> e.g., 256x512
    batch_size = 2 # RTX 3050 friendly test batch
    H, W = 256, 512
    dummy_input = torch.randn(batch_size, 5, H, W).to(device)
    
    print(f"Input Shape: {dummy_input.shape} (RGB + D + H)")
    
    # 3. Test Forward Pass
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        output = model(dummy_input)
        end_time = time.time()
        
    print(f"Output Shape: {output.shape}")
    print(f"Forward Pass Time ({batch_size} samples): {(end_time - start_time)*1000:.2f} ms")
    
    if output.shape == (batch_size, 2, H, W):
        print(" Output shape validation passed!")
    else:
        print(" Model output shape mismatch!")

    # 4. Test Loss Function Pipeline
    print("\n--- Testing Combined Loss ---")
    dummy_target = torch.randint(0, 2, (batch_size, H, W)).to(device)
    criterion = CombinedLoss()
    
    loss = criterion(output, dummy_target)
    print(f"Computed Loss: {loss.item():.4f}")
    if torch.isnan(loss) or torch.isinf(loss):
        print(" Loss computation failed (NaN / Inf)!")
    else:
        print(" Loss computation passed!")

if __name__ == "__main__":
    test_model()
