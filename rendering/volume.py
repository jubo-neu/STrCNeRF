import torch


def composite(rgb, sigma, t, far, white=True):
    rgb, sigma, t = rgb.float(), sigma.float(), t.float()
    intervals = torch.cat([t[..., 1:] - t[..., :-1], (far - t[..., -1:]).clamp_min(0)], -1)
    alpha = -torch.expm1(-sigma * intervals)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1 - alpha + 1e-10], -1), -1)[..., :-1]
    weights = alpha * trans
    opacity = weights.sum(-1)
    color = (weights[..., None] * rgb).sum(-2)
    if white: color = color + 1 - opacity[..., None]
    return {"rgb": color, "depth": (weights * t).sum(-1), "opacity": opacity, "weights": weights}
