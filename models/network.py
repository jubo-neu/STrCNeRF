import torch
from torch import nn
from torch.nn import functional as F
from .swin import NativeSwin, TimmSwin
from .cnn import CNNEncoder
from .fam import FAM
from .field import NeRFMLP
from strcnerf.rendering.camera import world_to_camera, project


class STrCNeRF(nn.Module):
    def __init__(self, config):
        super().__init__()
        backbone = config.get("backbone", "native")
        if backbone == "timm":
            self.global_encoder = TimmSwin(config["image_size"], config.get("pretrained", True))
        elif backbone == "native":
            self.global_encoder = NativeSwin(**config.get("swin", {}))
        else: raise ValueError(f"Unknown backbone: {backbone}")
        self.local_encoder = CNNEncoder(config.get("cnn_dim", 32))
        self.fam = FAM(self.global_encoder.out_channels, self.local_encoder.out_channels, config.get("attention", True))
        self.field = NeRFMLP(self.fam.out_channels, **config.get("field", {}))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def encode(self, image):
        if min(image.shape[-2:]) < 32 or any(s % 32 for s in image.shape[-2:]):
            raise ValueError("Image dimensions must be multiples of 32, at least 32")
        # BN in the last CNN block operates before its final pool; 32x32 is valid.
        image = (image - self.mean) / self.std
        return self.fam(self.global_encoder(image), self.local_encoder(image))

    def query(self, features, points, directions, batch):
        b, r, s, _ = points.shape
        with torch.autocast(device_type=points.device.type, enabled=False):
            camera_points = world_to_camera(points.float(), batch["source_c2w"].float())
            grid, valid = project(camera_points, batch["source_K"].float(), batch["source"].shape[-2:])
            sampled = F.grid_sample(features.float(), grid.reshape(b, r * s, 1, 2),
                                    align_corners=False, padding_mode="zeros")
            sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, r, s, -1) * valid[..., None]
            rotation = batch["source_c2w"][:, :3, :3].float().transpose(1, 2)
            camera_dirs = F.normalize(torch.einsum("bij,brsj->brsi", rotation, directions.float()), dim=-1)
        return self.field(sampled, camera_points, camera_dirs)
