import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.mobilenetv3 import MobileNetV3Encoder
from models.modules.aspp import ASPP, DepthwiseSeparableConv

class DeepLabV3Plus(nn.Module):
    """
    MobileNetV3-Large + DeepLabV3+ Decoder
    Highly optimized for latency and mIoU logic.
    """
    def __init__(self, in_channels=5, num_classes=2):
        super().__init__()
        
        # 1. Encoder (MobileNetV3-Large config for 5-channel)
        self.encoder = MobileNetV3Encoder(in_channels=in_channels)
        
        # 2. ASPP Module (takes 960 channels in, outputs 256)
        self.aspp = ASPP(in_channels=self.encoder.high_level_out_channels, out_channels=256)
        
        # 3. Decoder
        # Reduce low level features channel count
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(self.encoder.low_level_out_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Refinement Convolutions (After concatenating low level and ASPP features)
        # Using Depthwise Separable to keep FPS high
        self.cat_conv = nn.Sequential(
            DepthwiseSeparableConv(256 + 48, 256, kernel_size=3, padding=1, dilation=1),
            DepthwiseSeparableConv(256, 256, kernel_size=3, padding=1, dilation=1),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1) # Logits spatial output
        )

    def forward(self, x):
        input_shape = x.shape[2:] # Usually (512, 512)
        
        # Encoder High/Low Pass
        low_level, high_level = self.encoder(x)
        
        # ASPP context extrapolation
        aspp_out = self.aspp(high_level)
        
        # Decoder Fusion
        low_level_features = self.shortcut_conv(low_level)
        
        # Upsample ASPP logic to match Low-level structure
        aspp_upsampled = F.interpolate(aspp_out, size=low_level_features.shape[2:], mode='bilinear', align_corners=False)
        
        # Concat
        cat_features = torch.cat([low_level_features, aspp_upsampled], dim=1)
        
        # Refine and scale down to Classes
        out = self.cat_conv(cat_features)
        
        # Ensure final mask is perfectly fitted to Image Size Input
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        
        return out

if __name__ == '__main__':
    # Test Full pipeline
    model = DeepLabV3Plus(in_channels=5, num_classes=2)
    x = torch.randn(2, 5, 512, 512)
    out = model(x)
    print("Final Model output shape:", out.shape) # Expected: [2, 2, 512, 512]
