import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=1, padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling with Depthwise Separable Convolutions.
    Highly optimized for Real-time constraint on Edge devices.
    """
    def __init__(self, in_channels, out_channels=256, extract_rates=(1, 6, 12, 18)):
        super().__init__()
        
        # 1x1 Conv
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Depthwise Separable Convs with dilations
        self.aspp2 = DepthwiseSeparableConv(in_channels, out_channels, 3, padding=extract_rates[1], dilation=extract_rates[1])
        self.aspp3 = DepthwiseSeparableConv(in_channels, out_channels, 3, padding=extract_rates[2], dilation=extract_rates[2])
        self.aspp4 = DepthwiseSeparableConv(in_channels, out_channels, 3, padding=extract_rates[3], dilation=extract_rates[3])
        
        # Global Image Pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final Projection
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1) # low dropout since parameters are few
        )

    def forward(self, x):
        H, W = x.shape[2:]
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=(H, W), mode='bilinear', align_corners=False)
        
        out = torch.cat((x1, x2, x3, x4, x5), dim=1)
        return self.project(out)

if __name__ == '__main__':
    # Test ASPP block
    x = torch.randn(2, 960, 16, 16)
    aspp = ASPP(in_channels=960, out_channels=256)
    out = aspp(x)
    print("ASPP output shape:", out.shape) # Expected: [2, 256, 16, 16]
