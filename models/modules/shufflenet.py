import torch
import torch.nn as nn

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class ShuffleNetV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ShuffleNetV2Block, self).__init__()
        self.stride = stride
        branch_features = out_channels // 2

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(in_channels, in_channels, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.Hardswish(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Hardswish(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Hardswish(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride, padding):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=False, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out

class ShuffleNetV2Encoder(nn.Module):
    def __init__(self, stages_repeats=[4, 8, 4], stages_out_channels=[24, 116, 232, 464]):
        super(ShuffleNetV2Encoder, self).__init__()
        self.stages_repeats = stages_repeats
        self.stages_out_channels = stages_out_channels

        # Stem is handled by MultimodalFusion layer, expected input channels is stages_out_channels[0] which is 24

        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0])
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1])
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2])
        
        self._initialize_weights()

    def _make_stage(self, input_channels, output_channels, repeats):
        modules = [ShuffleNetV2Block(input_channels, output_channels, 2)]
        for _ in range(repeats - 1):
            modules.append(ShuffleNetV2Block(output_channels, output_channels, 1))
        return nn.Sequential(*modules)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x is from stem, size: 1/2 of original
        endpoints = []
        x = self.stage2(x) # 1/4
        endpoints.append(x)
        x = self.stage3(x) # 1/8
        endpoints.append(x)
        x = self.stage4(x) # 1/16
        endpoints.append(x)
        return endpoints
