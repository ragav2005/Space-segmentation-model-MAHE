import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """
    Lightweight cross-modal attention to interact RGB and LIDAR features.
    """
    def __init__(self, in_channels):
        super(CrossModalAttention, self).__init__()
        # Squeeze and Excitation like approach to re-weight channels based on combined context
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.Hardswish(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = avg_out + max_out
        return x * attention


class EarlyFusionStem(nn.Module):
    """
    Early fusion strategy combining 3-channel RGB and 2-channel LIDAR (depth+height).
    Outputs standard 24 channels for ShuffleNet v2 integration.
    """
    def __init__(self, rgb_channels=3, lidar_channels=2, out_channels=24):
        super(EarlyFusionStem, self).__init__()
        rgb_out = int(out_channels * (rgb_channels / (rgb_channels + lidar_channels)))
        lidar_out = out_channels - rgb_out
        
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(rgb_out),
            nn.Hardswish(inplace=True)
        )
        
        self.lidar_conv = nn.Sequential(
            nn.Conv2d(lidar_channels, lidar_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(lidar_out),
            nn.Hardswish(inplace=True)
        )
        
        self.attention = CrossModalAttention(out_channels)

    def forward(self, x):
        # Assumes input x is (B, 5, H, W) where first 3 are RGB, last 2 are Depth, Height
        rgb = x[:, :3, :, :]
        lidar = x[:, 3:, :, :]
        
        rgb_feat = self.rgb_conv(rgb)
        lidar_feat = self.lidar_conv(lidar)
        
        # Concatenate along channel dimension
        fused = torch.cat([rgb_feat, lidar_feat], dim=1)
        
        # Cross Modal Interaction
        fused = self.attention(fused)
        
        return fused
