"""Explicit Fourier options; see docs/REPRODUCTION.md for the Eq. (13) ambiguity."""
import math
import torch
from torch import nn


class FourierEncoding(nn.Module):
    def __init__(self, input_dim=3, mode="complex", frequencies=10, gaussian_features=64,
                 gaussian_sigma=10.0, seed=42, include_input=True):
        super().__init__()
        if mode not in {"none", "basic", "complex", "gaussian"}: raise ValueError(f"Unknown encoding: {mode}")
        if frequencies < 1 or gaussian_features < 1 or gaussian_sigma <= 0: raise ValueError("Invalid Fourier parameters")
        self.mode, self.include_input = mode, include_input
        self.register_buffer("bands", math.pi * 2.0 ** torch.arange(frequencies))
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("gaussian_matrix", torch.randn(gaussian_features, input_dim, generator=generator) * gaussian_sigma)
        self.out_dim = input_dim if mode == "none" else (input_dim if include_input else 0) + (
            2 * gaussian_features if mode == "gaussian" else 2 * input_dim * frequencies)

    def forward(self, x):
        if self.mode == "none": return x
        # Stay real-valued: real/imaginary parts of exp(i theta) are cos/sin.
        # This is the corrected Euler interpretation, not the printed minus/minus expression.
        if self.mode == "gaussian":
            theta = 2 * math.pi * (x @ self.gaussian_matrix.T)
            encoded = torch.cat([theta.sin(), theta.cos()], -1)
        else:
            theta = x[..., None] * self.bands
            encoded = torch.cat([theta.sin(), theta.cos()], -1).flatten(-2)
        return torch.cat([x, encoded], -1) if self.include_input else encoded
