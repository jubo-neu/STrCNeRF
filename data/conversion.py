"""Explicit camera conversions: manifest, NMR and LLFF poses_bounds exports."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .dataset import read_scene


def load_rgb(path, size, white=True):
    with Image.open(path) as image:
        original = image.size
        array = np.array(image.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)).astype(np.float32) / 255
    rgb = array[..., :3] * array[..., 3:] + float(white) * (1 - array[..., 3:])
    return (rgb * 255).round().astype(np.uint8), original


def resize_k(k, original, size):
    width, height = original
    k = np.array(k, dtype=np.float32, copy=True)[:3, :3]
    sx, sy = size / width, size / height
    k[0] *= sx; k[1] *= sy
    k[0, 2] += (sx - 1) / 2; k[1, 2] += (sy - 1) / 2
    return k


def save_archive(path, images, intrinsic, poses, **extra):
    path = Path(path)
    if path.exists(): raise FileExistsError(f"Refusing to overwrite {path}")
    if path.suffix != ".npz": raise ValueError("Output must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".pending.npz")
    np.savez_compressed(temporary, images=np.stack(images), K=np.stack(intrinsic).astype(np.float32),
                        c2w=np.stack(poses).astype(np.float32), **extra)
    read_scene(temporary); temporary.replace(path)


def convert_manifest(path, output, size, white=True):
    path = Path(path); document = json.loads(path.read_text(encoding="utf-8"))
    convention = document.get("camera_convention", "opencv")
    if convention not in {"opencv", "opengl"}: raise ValueError("Unknown manifest camera convention")
    images, intrinsics, poses = [], [], []
    for frame in document["frames"]:
        rgb, original = load_rgb(path.parent / frame["image"], size, white)
        pose = np.array(frame["c2w"], dtype=np.float32)
        if convention == "opengl": pose = pose @ np.diag([1., -1., -1., 1.])
        images.append(rgb); intrinsics.append(resize_k(frame["K"], original, size)); poses.append(pose)
    save_archive(output, images, intrinsics, poses)


def convert_nmr(path, output, size, convention, intrinsics_space, image_dir="image", white=True):
    if convention not in {"opencv", "opengl"} or intrinsics_space not in {"normalized", "pixel"}:
        raise ValueError("NMR camera convention and intrinsics space must be explicit")
    path = Path(path); images, intrinsics, poses = [], [], []
    files = sorted(p for p in (path / image_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    with np.load(path / "cameras.npz", allow_pickle=False) as archive:
        for i, image_path in enumerate(files):
            rgb, original = load_rgb(image_path, size, white)
            width, height = original
            extrinsic = np.eye(4); extrinsic[:3, :4] = archive[f"world_mat_{i}"][:3, :4]
            pose = np.linalg.inv(extrinsic)
            k = archive[f"camera_mat_{i}"][:3, :3].copy()
            flip = np.diag([1., -1., -1.]) if convention == "opengl" else np.eye(3)
            if convention == "opengl": pose = pose @ np.diag([1., -1., -1., 1.])
            if intrinsics_space == "normalized":
                pixel_map = np.array([[(width - 1) / 2, 0, (width - 1) / 2], [0, (height - 1) / 2, (height - 1) / 2], [0, 0, 1]])
                k = pixel_map @ k
            k = k @ flip; k /= k[2, 2]
            images.append(rgb); intrinsics.append(resize_k(k, original, size)); poses.append(pose)
    save_archive(output, images, intrinsics, poses)


def convert_llff(path, output_root, size=128, holdout=8, image_dir="images", scale=1.0):
    """Convert LLFF [down,right,back] c2w columns to OpenCV [right,down,forward].

    Explicit finite world-space rays, no LLFF NDC warp. Returned bounds are diagnostics.
    """
    if holdout < 2 or scale <= 0: raise ValueError("holdout >= 2 and positive scale required")
    path = Path(path)
    archive = np.load(path / "poses_bounds.npy", allow_pickle=False)
    if archive.ndim != 2 or archive.shape[1] != 17: raise ValueError("LLFF expects poses_bounds.npy [V,17]")
    files = sorted(p for p in (path / image_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if len(files) != len(archive): raise ValueError("Image count does not match LLFF poses")
    images, intrinsics, poses = [], [], []
    for row, image_path in zip(archive, files):
        llff = row[:15].reshape(3, 5)
        original_h, original_w, focal = llff[:, 4]
        k = np.array([[focal, 0, (original_w - 1) / 2], [0, focal, (original_h - 1) / 2], [0, 0, 1]])
        rgb, _ = load_rgb(image_path, size)
        # Calibration uses poses_bounds stored H/W, also for pre-downsampled images.
        intrinsics.append(resize_k(k, (original_w, original_h), size))
        pose = np.eye(4)
        pose[:3, :3] = np.stack([llff[:, 1], llff[:, 0], -llff[:, 2]], -1)
        pose[:3, 3] = llff[:, 3] * scale
        images.append(rgb); poses.append(pose)
    held = np.arange(0, len(files), holdout)
    train = np.array([i for i in range(len(files)) if i not in held], dtype=np.int64)
    if len(train) < 2 or not len(held): raise ValueError("Need >=2 training views and >=1 held-out view")
    name = path.name
    for split, targets in [("train", train), ("val", held), ("test", held)]:
        save_archive(Path(output_root) / split / f"{name}.npz", images, intrinsics, poses,
                     source_indices=train, target_indices=targets)
    return {"scene": name, "training_views": train.tolist(), "heldout_views": held.tolist(),
            "suggested_source_view": int(train[0]), "scaled_depth_bounds": [float(archive[:, 15:].min() * scale), float(archive[:, 15:].max() * scale)],
            "note": "Bounds are camera-depth estimates; rendering uses normalized-ray distance. Inspect coverage before choosing near/far. Val/test share heldout views by default; use a separate validation set for model selection."}
