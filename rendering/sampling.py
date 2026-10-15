import torch


def uniform_samples(origins, near, far, count, perturb):
    edges = torch.linspace(near, far, count + 1, device=origins.device)
    shape = (*origins.shape[:-1], count)
    offset = torch.rand(shape, device=origins.device) if perturb else origins.new_full(shape, 0.5)
    return edges[:-1] + (edges[1:] - edges[:-1]) * offset


def sample_pdf(bins, weights, count, deterministic):
    weights = weights.detach() + 1e-5
    cdf = (weights / weights.sum(-1, keepdim=True)).cumsum(-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    shape = (*weights.shape[:-1], count)
    u = ((torch.arange(count, device=bins.device) + 0.5) / count).expand(shape) if deterministic else torch.rand(shape, device=bins.device)
    index = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)
    low, high = (index - 1).clamp_min(0), index.clamp_max(cdf.shape[-1] - 1)
    c0, c1 = cdf.gather(-1, low), cdf.gather(-1, high)
    b0, b1 = bins.gather(-1, low), bins.gather(-1, high)
    return b0 + (u - c0) / (c1 - c0).clamp_min(1e-8) * (b1 - b0)
