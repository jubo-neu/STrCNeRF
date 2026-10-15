"""Scene archives with separate source/target policies for cross-scene and LLFF runs."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def read_scene(path):
    with np.load(path, allow_pickle=False) as archive:
        rgb = archive["images"]
        k, pose = archive["K"].astype(np.float32), archive["c2w"].astype(np.float32)
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or len(rgb) < 2: raise ValueError(f"{path}: expected RGB [V,H,W,3], V>=2")
    rgb = rgb.astype(np.float32) / (255 if rgb.dtype == np.uint8 else 1)
    n = len(rgb)
    if k.shape == (3, 3): k = np.repeat(k[None], n, axis=0)
    if k.shape != (n, 3, 3) or pose.shape != (n, 4, 4): raise ValueError(f"{path}: invalid camera shapes")
    if not all(np.isfinite(x).all() for x in (rgb, k, pose)): raise ValueError(f"{path}: nonfinite values")
    if rgb.min() < 0 or rgb.max() > 1: raise ValueError("RGB must be uint8 or float in [0,1]")
    r = pose[:, :3, :3]
    if not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=2e-3) or not np.allclose(np.linalg.det(r), 1, atol=2e-3):
        raise ValueError("Camera rotations must be proper orthonormal matrices")
    if not np.allclose(pose[:, 3], [0, 0, 0, 1], atol=1e-5): raise ValueError("Invalid c2w homogeneous row")
    if np.any(k[:, [0, 1], [0, 1]] <= 0) or not np.allclose(k[:, 2], [0, 0, 1], atol=1e-5):
        raise ValueError("Intrinsics must have positive focal lengths and last row [0,0,1]")
    return torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(k), torch.from_numpy(pose)


class SceneDataset(Dataset):
    def __init__(self, root, split="train", image_size=64, evaluation=False, source_view=0):
        self.paths = sorted((Path(root) / split).glob("*.npz"))
        if not self.paths: raise FileNotFoundError(f"No archives in {Path(root) / split}")
        self.image_size, self.evaluation = image_size, evaluation
        self.policies, self.pairs = [], []
        for scene, path in enumerate(self.paths):
            with np.load(path, allow_pickle=False) as archive:
                n = len(archive["images"])
                sources = archive["source_indices"].astype(int).tolist() if "source_indices" in archive else list(range(n))
                targets = archive["target_indices"].astype(int).tolist() if "target_indices" in archive else list(range(n))
            if not sources or not targets or any(i < 0 or i >= n for i in sources + targets):
                raise ValueError(f"{path}: invalid source/target index lists")
            if any(not any(t != s for t in targets) for s in sources): raise ValueError("Each source needs a different target")
            self.policies.append((sources, targets))
            if evaluation:
                if source_view not in sources: raise ValueError(f"Source view {source_view} is not allowed in {path.name}")
                self.pairs.extend((scene, source_view, t) for t in targets if t != source_view)

    def __len__(self):
        return len(self.pairs) if self.evaluation else len(self.paths)

    def __getitem__(self, index):
        if self.evaluation:
            scene, source, target = self.pairs[index]
        else:
            scene = index
            sources, targets = self.policies[scene]
            source = sources[torch.randint(len(sources), ()).item()]
            candidates = [t for t in targets if t != source]
            target = candidates[torch.randint(len(candidates), ()).item()]
        images, k, poses = read_scene(self.paths[scene])
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"{self.paths[scene]}: image resolution does not match config")
        return {"source": images[source], "target": images[target], "source_K": k[source], "target_K": k[target],
                "source_c2w": poses[source], "target_c2w": poses[target],
                "scene": self.paths[scene].stem, "source_index": source, "target_index": target}
