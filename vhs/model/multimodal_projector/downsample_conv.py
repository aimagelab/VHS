import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableCompressor(nn.Module):
    def __init__(self, out_scale=0.75, in_channels=32, out_channels=32):
        super().__init__()
        self.out_scale = out_scale
        # 1x1 conv keeps channel count fixed
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        if x.ndim == 3:
            B, HW, C = x.shape
            side = int(torch.sqrt(torch.tensor(HW)))
            x = x.reshape(B, C, side, side)
        B, C, H, W = x.shape
        # 1x1 conv can learn channel mixing but not downsample spatially,
        # so combine with interpolate:
        x = self.proj(x)
        x = F.interpolate(x, size=(int(H * self.out_scale), int(W * self.out_scale)),
                          mode='bilinear', align_corners=False)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0,2,1)
        return x
