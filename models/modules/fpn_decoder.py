import torch
import torch.nn as nn
import torch.nn.functional as F

class UpSamplingBlock(nn.Module):
    """
    Lightweight UpSampling with Depthwise Separable convolutions for smoothing.
    """
    def __init__(self, in_channels, out_channels):
        super(UpSamplingBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(x)

class EnhancedFPNDecoder(nn.Module):
    """
    Lightweight FPN Decoder.
    Takes multi-scale features from ShuffleNet (1/4, 1/8, 1/16) and produces a full resolution mask.
    """
    def __init__(self, in_channels_list=[116, 232, 464], out_channels=64, num_classes=2):
        super(EnhancedFPNDecoder, self).__init__()
        
        # Lateral connections
        self.lat2 = nn.Conv2d(in_channels_list[0], out_channels, 1, bias=False)
        self.lat3 = nn.Conv2d(in_channels_list[1], out_channels, 1, bias=False)
        self.lat4 = nn.Conv2d(in_channels_list[2], out_channels, 1, bias=False)
        
        # Top-down pathways
        self.up4_to_3 = UpSamplingBlock(out_channels, out_channels)
        self.up3_to_2 = UpSamplingBlock(out_channels, out_channels)
        
        self.smooth2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True)
        )
        
        # Final upsampling (1/4 to full resolution -> 4x upsampling)
        self.final_up = nn.Sequential(
            UpSamplingBlock(out_channels, out_channels // 2),
            UpSamplingBlock(out_channels // 2, out_channels // 2),
            nn.Conv2d(out_channels // 2, num_classes, kernel_size=1)
        )

    def forward(self, features):
        c2, c3, c4 = features  # 1/4, 1/8, 1/16

        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + self.up4_to_3(p4)
        p2 = self.lat2(c2) + self.up3_to_2(p3)
        
        p2 = self.smooth2(p2)
        
        # Upsample 4x to original resolution
        out = self.final_up(p2)
        return out
