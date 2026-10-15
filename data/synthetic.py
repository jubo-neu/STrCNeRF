"""Offline colored-ellipsoid views with analytically known camera matrices."""
from pathlib import Path
import numpy as np


def look_at(eye):
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, [0., 1., 0.]); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.stack([right, down, forward], -1); pose[:3, 3] = eye
    return pose


def make_demo(root, size=32, views=4, scenes=2, seed=42):
    if size < 32 or size % 32 or views < 2 or scenes < 1: raise ValueError("Invalid demo dimensions/counts")
    rng = np.random.default_rng(seed)
    k = np.array([[size * 1.2, 0, (size - 1) / 2], [0, size * 1.2, (size - 1) / 2], [0, 0, 1]], dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    local = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(k).T
    for split in ("train", "val", "test"):
        directory = Path(root) / split; directory.mkdir(parents=True, exist_ok=True)
        for scene in range(scenes):
            radius, base = rng.uniform(0.35, 0.6, 3), rng.uniform(0.15, 0.85, 3)
            images, poses = [], []
            for angle in np.linspace(0, 2 * np.pi, views, endpoint=False):
                eye = np.array([2 * np.sin(angle), 0.35, 2 * np.cos(angle)])
                pose = look_at(eye)
                direction = local @ pose[:3, :3].T; direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
                o, d = eye / radius, direction / radius
                a, b, c = (d * d).sum(-1), 2 * (o * d).sum(-1), (o * o).sum() - 1
                discriminant = b * b - 4 * a * c
                t = (-b - np.sqrt(np.maximum(discriminant, 0))) / (2 * a)
                hit = (discriminant >= 0) & (t > 0)
                p = eye + t[..., None] * direction
                normal = p / radius ** 2; normal /= np.linalg.norm(normal, axis=-1, keepdims=True).clip(1e-8)
                light = np.array([0.4, 0.8, 0.4]); light /= np.linalg.norm(light)
                shade = 0.4 + 0.6 * np.maximum(normal @ light, 0)
                pattern = 0.85 + 0.15 * np.sin(p[..., 0] * 20) * np.cos(p[..., 1] * 18)
                rgb = np.ones((size, size, 3)); rgb[hit] = (base * shade[..., None] * pattern[..., None])[hit]
                images.append((rgb.clip(0, 1) * 255).round().astype(np.uint8)); poses.append(pose)
            np.savez_compressed(directory / f"{split}_{scene:03d}.npz", images=np.stack(images), K=k, c2w=np.stack(poses))
