import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class MobileNetV3Encoder(nn.Module):
    """
    MobileNetV3-Large Encoder for 5-Channel input.
    Extracts low-level features and high-level features for DeepLabV3+.
    Initialized with random weights.
    """
    def __init__(self, in_channels=5):
        super().__init__()
        
        # Load vision model with random initialization
        mnet = mobilenet_v3_large(weights=None)
        
        # Modify the first Convolution to accept `in_channels`
        # Original: Conv2dNormActivation(3, 16, kernel_size=(3, 3), stride=(2, 2))
        original_conv = mnet.features[0][0]
        new_conv = nn.Conv2d(in_channels, original_conv.out_channels, 
                             kernel_size=original_conv.kernel_size, 
                             stride=original_conv.stride, 
                             padding=original_conv.padding, 
                             bias=False)
        
        # Initialize with Kaiming (He) initialization for proper data flow
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        mnet.features[0][0] = new_conv
        
        # For DeepLabV3+, we need low level features and high level features.
        # MobileNet_v3_large typical expansion blocks:
        # features[3] outputs 24 channels (stride 4) -> good for low_level
        # features[16] outputs 960 channels (stride 32) -> good for ASPP high_level
        
        self.low_level_features = mnet.features[:4] # Output: [B, 24, H/4, W/4]
        self.high_level_features = mnet.features[4:17] # Output: [B, 960, H/32, W/32]
        
        # Low level and High level output channels
        self.low_level_out_channels = 24
        self.high_level_out_channels = 960

    def forward(self, x):
        low_level = self.low_level_features(x)
        high_level = self.high_level_features(low_level)
        return low_level, high_level

if __name__ == '__main__':
    model = MobileNetV3Encoder(in_channels=5)
    x = torch.randn(2, 5, 512, 512)
    low, high = model(x)
    print("Low level shape:", low.shape)   # Expected: [2, 24, 128, 128]
    print("High level shape:", high.shape) # Expected: [2, 960, 16, 16]
    
    # Count params
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,}")
