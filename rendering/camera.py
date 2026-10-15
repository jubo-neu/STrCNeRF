"""Column-vector OpenCV cameras; +x right, +y down, +z forward."""
import torch
from torch.nn import functional as F


def generate_rays(k, c2w, height, width, indices=None):
    if indices is None:
        indices = torch.arange(height * width, device=k.device)[None].expand(k.shape[0], -1)
    pixels = torch.stack([indices % width, indices // width, torch.ones_like(indices)], -1).to(k.dtype)
    local = torch.einsum("bij,brj->bri", torch.linalg.inv(k), pixels)
    direction = F.normalize(torch.einsum("bij,brj->bri", c2w[:, :3, :3], local), dim=-1)
    return c2w[:, None, :3, 3].expand_as(direction), direction


def world_to_camera(points, c2w):
    return torch.einsum("bij,brsj->brsi", c2w[:, :3, :3].transpose(1, 2), points - c2w[:, None, None, :3, 3])


def project(points, k, size):
    h, w = size
    pixels = torch.einsum("bij,brsj->brsi", k, points)
    uv = pixels[..., :2] / pixels[..., 2:].clamp_min(1e-8)
    grid = torch.stack([2 * (uv[..., 0] + 0.5) / w - 1, 2 * (uv[..., 1] + 0.5) / h - 1], -1)
    valid = (points[..., 2] > 0) & (uv[..., 0] >= 0) & (uv[..., 0] <= w - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= h - 1)
    # Invalid coordinates are zeroed before grid_sample to avoid giant AMP inputs.
    return torch.where(valid[..., None], grid, torch.zeros_like(grid)), valid.to(points.dtype)
