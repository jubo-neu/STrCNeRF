import torch
from .sampling import uniform_samples, sample_pdf
from .volume import composite
from .camera import generate_rays


class Renderer:
    def __init__(self, near=0.8, far=3.5, coarse_samples=64, fine_samples=128,
                 ray_chunk=256, query_chunk=8192, white_background=True):
        if not 0 < near < far or coarse_samples < 3 or fine_samples < 0 or min(ray_chunk, query_chunk) < 1:
            raise ValueError("Invalid rendering bounds, sample counts or chunk sizes")
        self.near, self.far = near, far
        self.coarse_samples, self.fine_samples = coarse_samples, fine_samples
        self.ray_chunk, self.query_chunk = ray_chunk, query_chunk
        self.white = white_background

    def _pass(self, model, features, origin, direction, depths, batch):
        b, r, s = depths.shape
        points = origin[..., None, :] + direction[..., None, :] * depths[..., None]
        dirs = direction[..., None, :].expand_as(points)
        points, dirs = points.reshape(b, -1, 1, 3), dirs.reshape(b, -1, 1, 3)
        colors, densities = [], []
        for start in range(0, r * s, self.query_chunk):
            rgb, sigma = model.query(features, points[:, start:start + self.query_chunk], dirs[:, start:start + self.query_chunk], batch)
            colors.append(rgb); densities.append(sigma)
        return composite(torch.cat(colors, 1).reshape(b, r, s, 3), torch.cat(densities, 1).reshape(b, r, s), depths, self.far, self.white)

    def rays(self, model, features, origin, direction, batch, perturb=False):
        t = uniform_samples(origin, self.near, self.far, self.coarse_samples, perturb)
        coarse = self._pass(model, features, origin, direction, t, batch)
        if not self.fine_samples: return {**coarse, "coarse_rgb": coarse["rgb"]}
        mid = (t[..., 1:] + t[..., :-1]) * 0.5
        fine = sample_pdf(mid, coarse["weights"][..., 1:-1], self.fine_samples, not perturb)
        t = torch.cat([t, fine.detach()], -1).sort(-1).values
        return {**self._pass(model, features, origin, direction, t, batch), "coarse_rgb": coarse["rgb"]}

    @torch.no_grad()
    def image(self, model, batch, features=None):
        h, w = batch["source"].shape[-2:]
        if features is None: features = model.encode(batch["source"])
        origin, direction = generate_rays(batch["target_K"], batch["target_c2w"], h, w)
        result = {key: [] for key in ("rgb", "depth", "opacity")}
        for start in range(0, h * w, self.ray_chunk):
            out = self.rays(model, features, origin[:, start:start + self.ray_chunk], direction[:, start:start + self.ray_chunk], batch)
            for key in result: result[key].append(out[key])
        return {key: torch.cat(values, 1).reshape(-1, h, w, 3 if key == "rgb" else 1).permute(0, 3, 1, 2)
                for key, values in result.items()}
