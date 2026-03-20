import torch
import torch.nn as nn

from models.modules.multimodal_fusion import EarlyFusionStem
from models.modules.shufflenet import ShuffleNetV2Encoder
from models.modules.fpn_decoder import EnhancedFPNDecoder

class ShuffleNetSegmentation(nn.Module):
    """
    Multi-Modal Real-Time Drivable Space Segmentation Network
    Encoder: Early Fusion + ShuffleNet v2
    Decoder: Lightweight FPN
    """
    def __init__(self, in_channels=5, num_classes=2):
        super(ShuffleNetSegmentation, self).__init__()
        
        # 1. Early Fusion Stem (Transforms 5 channels to 24 channels block)
        self.stem = EarlyFusionStem(rgb_channels=3, lidar_channels=2, out_channels=24)
        
        # 2. Backbone: ShuffleNet v2 x1.0
        self.encoder = ShuffleNetV2Encoder(stages_repeats=[4, 8, 4], stages_out_channels=[24, 116, 232, 464])
        
        # 3. Decoder: Enhanced FPN
        self.decoder = EnhancedFPNDecoder(in_channels_list=[116, 232, 464], out_channels=64, num_classes=num_classes)
        
    def forward(self, x):
        # x is expected to be [B, 5, H, W]
        # H, W must be divisible by 32
        
        x = self.stem(x)
        features = self.encoder(x)
        out = self.decoder(features)
        
        return out

if __name__ == '__main__':
    # Test instantiating the model
    model = ShuffleNetSegmentation(in_channels=5, num_classes=2)
    print("Model successfully created.")
