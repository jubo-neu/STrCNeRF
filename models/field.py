"""Eight-layer, 256-wide coordinate MLP in the paper-scale presets."""
import torch
from torch import nn
from torch.nn import functional as F
from .encoding import FourierEncoding


class NeRFMLP(nn.Module):
    def __init__(self, feature_channels, hidden=256, layers=8, skip_layer=4,
                 encoding=None, direction_frequencies=4):
        super().__init__()
        self.position = FourierEncoding(**(encoding or {}))
        self.direction = FourierEncoding(mode="basic", frequencies=direction_frequencies)
        self.skip_layer = skip_layer
        self.input_dim = feature_channels + self.position.out_dim
        self.trunk = nn.ModuleList()
        for i in range(layers):
            cin = self.input_dim if i == 0 else hidden + (self.input_dim if i == skip_layer else 0)
            self.trunk.append(nn.Linear(cin, hidden))
        self.sigma = nn.Linear(hidden, 1)
        self.rgb = nn.Sequential(nn.Linear(hidden + self.direction.out_dim, hidden // 2),
                                  nn.ReLU(), nn.Linear(hidden // 2, 3), nn.Sigmoid())

    def forward(self, feature, points, directions):
        inputs = torch.cat([feature, self.position(points)], -1)
        x = inputs
        for i, layer in enumerate(self.trunk):
            if i and i == self.skip_layer: x = torch.cat([x, inputs], -1)
            x = F.relu(layer(x))
        sigma = F.softplus(self.sigma(x).squeeze(-1) - 1.0)
        rgb = self.rgb(torch.cat([x, self.direction(directions)], -1))
        return rgb, sigma
