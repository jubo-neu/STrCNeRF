import torch
from torch.nn import functional as F


def psnr(prediction, target):
    return -10 * torch.log10((prediction - target).square().mean((-3, -2, -1)).clamp_min(1e-12))


def ssim(prediction, target):
    x = torch.arange(11, device=prediction.device, dtype=prediction.dtype) - 5
    kernel = torch.exp(-x.square() / 4.5); kernel /= kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :])[None, None].expand(3, 1, 11, 11)
    def smooth(v): return F.conv2d(v, kernel, groups=3)
    mx, my = smooth(prediction), smooth(target)
    vx, vy = smooth(prediction.square()) - mx.square(), smooth(target.square()) - my.square()
    cov = smooth(prediction * target) - mx * my
    score = ((2 * mx * my + 0.01 ** 2) * (2 * cov + 0.03 ** 2)) / (
        (mx.square() + my.square() + 0.01 ** 2) * (vx + vy + 0.03 ** 2))
    return score.mean((-3, -2, -1))


class Metrics:
    def __init__(self, device, use_lpips=False):
        self.lpips = None
        if use_lpips:
            try: import lpips
            except ImportError as exc: raise ImportError('Install pip install -e ".[metrics]"') from exc
            self.lpips = lpips.LPIPS(net="alex").to(device).eval()

    @torch.no_grad()
    def __call__(self, prediction, target):
        result = {"psnr": psnr(prediction, target).mean().item(), "ssim": ssim(prediction, target).mean().item()}
        if self.lpips is not None: result["lpips"] = self.lpips(2 * prediction - 1, 2 * target - 1).mean().item()
        return result
