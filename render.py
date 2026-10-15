import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from strcnerf.checkpoint import model_from_checkpoint
from strcnerf.data import read_scene
from strcnerf.rendering import Renderer
from strcnerf.utils import device_for, save_rgb


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Render scene cameras or a custom OpenCV trajectory")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source-view", type=int, default=0)
    parser.add_argument("--trajectory", help="NPZ: c2w [T,4,4], optional K [T,3,3] or [3,3]")
    parser.add_argument("--output", default="runs/render")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args(); torch.set_num_threads(4)
    device = device_for(args.device); model, state = model_from_checkpoint(args.checkpoint, device)
    images, k, poses = read_scene(args.scene); source = args.source_view
    if not 0 <= source < len(images): raise ValueError("Source view out of range")
    if images.shape[-2:] != (state["config"]["model"]["image_size"],) * 2: raise ValueError("Scene/checkpoint resolution mismatch")
    target_k, target_poses = k.numpy(), poses.numpy()
    if args.trajectory:
        with np.load(args.trajectory, allow_pickle=False) as archive:
            target_poses = archive["c2w"].astype(np.float32)
            target_k = archive["K"].astype(np.float32) if "K" in archive else k[source].numpy()
        if target_k.shape == (3, 3): target_k = np.repeat(target_k[None], len(target_poses), axis=0)
    if target_poses.ndim != 3 or target_poses.shape[1:] != (4, 4) or target_k.shape != (len(target_poses), 3, 3):
        raise ValueError("Trajectory has invalid shapes")
    if not len(target_poses) or not np.isfinite(target_k).all() or not np.isfinite(target_poses).all():
        raise ValueError("Trajectory must contain finite camera matrices")
    rotations = target_poses[:, :3, :3]
    if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=2e-3) or not np.allclose(np.linalg.det(rotations), 1, atol=2e-3):
        raise ValueError("Trajectory rotations must be proper rigid transforms")
    batch = {"source": images[source:source+1].to(device), "source_K": k[source:source+1].to(device), "source_c2w": poses[source:source+1].to(device)}
    renderer = Renderer(**state["config"]["render"]); features = model.encode(batch["source"])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); frames = []
    save_rgb(images[source], output / "source.png")
    for i, (ki, pi) in enumerate(zip(target_k, target_poses)):
        batch["target_K"] = torch.from_numpy(ki)[None].to(device)
        batch["target_c2w"] = torch.from_numpy(pi)[None].to(device)
        result = renderer.image(model, batch, features)
        path = output / f"{i:04d}.png"; save_rgb(result["rgb"][0], path)
        np.savez_compressed(output / f"{i:04d}_geometry.npz", depth=result["depth"][0,0].cpu().numpy(), opacity=result["opacity"][0,0].cpu().numpy())
        if args.gif:
            with Image.open(path) as frame: frames.append(frame.copy())
    if args.gif: frames[0].save(output / "preview.gif", save_all=True, append_images=frames[1:], duration=120, loop=0)
    print(f"Rendered {len(target_poses)} views to {output}")


if __name__ == "__main__": main()
