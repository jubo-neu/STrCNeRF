"""Channel-only Feature Aggregation Module, Eqs. (5)-(9), Fig. 2."""
import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.shared_mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, channels, 1))

    def forward(self, x):
        mean, maximum = x.mean((2, 3), keepdim=True), x.amax((2, 3), keepdim=True)
        weights = (self.shared_mlp(mean) + self.shared_mlp(maximum)).sigmoid()
        return x * weights


class FAM(nn.Module):
    def __init__(self, global_channels, local_channels, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.global_attention = ChannelAttention(global_channels)
        self.local_attention = ChannelAttention(local_channels)
        self.out_channels = global_channels + local_channels

    def forward(self, global_feature, local_feature):
        if global_feature.shape[-2:] != local_feature.shape[-2:]:
            raise ValueError("FAM requires aligned global and local spatial dimensions")
        if self.enabled:
            global_feature = self.global_attention(global_feature)
            local_feature = self.local_attention(local_feature)
        return torch.cat([global_feature, local_feature], dim=1)
