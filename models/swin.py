"""Hierarchical Swin encoder with relative bias, shifted windows and padding masks."""
import torch
from torch import nn
from torch.nn import functional as F


def partition(x, size):
    b, h, w, c = x.shape
    return x.reshape(b, h // size, size, w // size, size, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, size * size, c)


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, window):
        super().__init__()
        if dim % heads: raise ValueError("Attention dimension must be divisible by heads")
        self.heads, self.scale = heads, (dim // heads) ** -0.5
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.relative_bias = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        xy = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")).flatten(1)
        relative = (xy[:, :, None] - xy[:, None, :]).permute(1, 2, 0) + window - 1
        self.register_buffer("index", relative[..., 0] * (2 * window - 1) + relative[..., 1])
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(self, x, mask):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        logits = (q * self.scale) @ k.transpose(-1, -2)
        logits = logits + self.relative_bias[self.index].permute(2, 0, 1)[None]
        windows = mask.shape[0]
        logits = (logits.reshape(-1, windows, self.heads, n, n) + mask[None, :, None]).flatten(0, 1)
        output = logits.softmax(-1) @ v
        return self.proj(output.transpose(1, 2).reshape(b, n, c))


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, window, shifted):
        super().__init__()
        self.window, self.shifted = window, shifted
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = WindowAttention(dim, heads, window)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        b, h, w, c = x.shape
        size = self.window
        shift = size // 2 if self.shifted and min(h, w) > size else 0
        y = F.pad(self.norm1(x), (0, 0, 0, (-w) % size, 0, (-h) % size))
        hp, wp = y.shape[1:3]
        labels = y.new_zeros(1, hp, wp, 1)
        if shift:
            counter = 0
            for hs in (slice(0, -size), slice(-size, -shift), slice(-shift, None)):
                for ws in (slice(0, -size), slice(-size, -shift), slice(-shift, None)):
                    labels[:, hs, ws] = counter; counter += 1
        labels = partition(labels, size).squeeze(-1)
        mask = (labels[:, :, None] != labels[:, None, :]).to(y.dtype) * -100.0
        valid = y.new_zeros(1, hp, wp, 1); valid[:, :h, :w] = 1
        if shift:
            y = torch.roll(y, (-shift, -shift), (1, 2))
            valid = torch.roll(valid, (-shift, -shift), (1, 2))
        valid = partition(valid, size).squeeze(-1)
        mask = mask + (1 - valid[:, None, :]) * -100.0
        y = self.attention(partition(y, size), mask)
        y = y.reshape(b, hp // size, wp // size, size, size, c).permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, c)
        if shift: y = torch.roll(y, (shift, shift), (1, 2))
        x = x + y[:, :h, :w]
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm, self.proj = nn.LayerNorm(4 * dim), nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        h, w = x.shape[1:3]
        x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        x = torch.cat([x[:, ::2, ::2], x[:, 1::2, ::2], x[:, ::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.proj(self.norm(x))


class NativeSwin(nn.Module):
    def __init__(self, dim=96, depths=(2, 2, 6, 2), heads=(3, 6, 12, 24), window=7):
        super().__init__()
        if len(depths) != 4 or len(heads) != 4: raise ValueError("Expected four Swin stages")
        self.out_channels = dim * 8
        self.embed = nn.Conv2d(3, dim, 4, 4)
        self.norm = nn.LayerNorm(dim)
        self.stages = nn.ModuleList([nn.Sequential(*[
            SwinBlock(dim * 2 ** i, heads[i], window, bool(j % 2)) for j in range(depths[i])
        ]) for i in range(4)])
        self.merges = nn.ModuleList([PatchMerging(dim * 2 ** i) for i in range(3)])

    def forward(self, image):
        x = self.norm(self.embed(image).permute(0, 2, 3, 1))
        for i, stage in enumerate(self.stages):
            if i: x = self.merges[i - 1](x)
            x = stage(x)
        return x.permute(0, 3, 1, 2).contiguous()


class TimmSwin(nn.Module):
    def __init__(self, image_size=64, pretrained=True):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError('Install pip install -e ".[pretrained]" to use timm Swin') from exc
        self.encoder = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained,
                                         features_only=True, out_indices=(3,), img_size=image_size)
        self.out_channels = self.encoder.feature_info.channels()[0]

    def forward(self, image):
        return self.encoder(image)[0].permute(0, 3, 1, 2).contiguous()
