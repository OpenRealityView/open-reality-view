"""Render Gaussians into (virtual) cameras: color, latent features, depth and alpha."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import torch

from orv.geometry import Cameras
from orv.models.gaussians import Gaussians


@dataclass
class Render:
    color: torch.Tensor  # [B, V, 3, H, W], premultiplied by alpha
    features: torch.Tensor  # [B, V, C, H, W], premultiplied by alpha
    depth: torch.Tensor  # [B, V, 1, H, W], expected z-depth
    alpha: torch.Tensor  # [B, V, 1, H, W]


def render(gaussians: Gaussians, cameras: Cameras, backend: str = "auto") -> Render:
    """Render into cameras with K [B, V, 3, 3] and cam_to_world [B, V, 4, 4].

    ``backend`` is "gsplat" (CUDA), "naive" (pure PyTorch, for debugging) or "auto".
    """
    if backend == "auto":
        use_gsplat = gaussians.means.is_cuda and importlib.util.find_spec("gsplat") is not None
        backend = "gsplat" if use_gsplat else "naive"
    signal = torch.cat([gaussians.colors, gaussians.features], dim=-1)
    if backend == "gsplat":
        out, alpha = _rasterize_gsplat(gaussians, signal, cameras)
    elif backend == "naive":
        out, alpha = _splat_naive(gaussians, signal, cameras)
    else:
        raise ValueError(f"unknown render backend {backend!r}")
    out = out.permute(0, 1, 4, 2, 3)  # [B, V, 3 + C + 1, H, W]
    color, features, depth = out.split([3, signal.shape[-1] - 3, 1], dim=2)
    return Render(color, features, depth, alpha.permute(0, 1, 4, 2, 3))


def _rasterize_gsplat(
    g: Gaussians, signal: torch.Tensor, cams: Cameras
) -> tuple[torch.Tensor, torch.Tensor]:
    from gsplat import rasterization

    viewmats = cams.world_to_cam()
    outs, alphas = [], []
    for b in range(signal.shape[0]):
        out, alpha, _ = rasterization(
            g.means[b],
            g.quats[b],
            g.scales[b],
            g.opacities[b],
            signal[b],
            viewmats[b],
            cams.K[b],
            cams.width,
            cams.height,
            render_mode="RGB+ED",
        )
        outs.append(out)
        alphas.append(alpha)
    return torch.stack(outs), torch.stack(alphas)


def _splat_naive(
    g: Gaussians, signal: torch.Tensor, cams: Cameras
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-point z-buffer splatting in pure PyTorch, for CPU debugging and tests.

    Ignores the Gaussian extent; gradients reach colors, features, opacities and depth but not
    image-plane positions.
    """
    B, V = cams.K.shape[:2]
    H, W, D = cams.height, cams.width, signal.shape[-1]
    uv, z = cams.project(g.means[:, None])  # [B, V, G, 2], [B, V, G]
    u, v = uv.unbind(-1)
    inside = (z > 0.01) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pix = v.clamp(0, H - 1).long() * W + u.clamp(0, W - 1).long()
    key = torch.where(inside, z, torch.full_like(z, float("inf")))
    zbuf = torch.full((B, V, H * W), float("inf"), dtype=z.dtype, device=z.device)
    zbuf = zbuf.scatter_reduce(2, pix, key, reduce="amin")
    front = inside & (key <= zbuf.gather(2, pix))
    weight = front.to(signal.dtype) * g.opacities[:, None]  # [B, V, G]
    values = torch.cat([signal[:, None].expand(-1, V, -1, -1), z[..., None]], dim=-1)
    index = pix[..., None].expand(-1, -1, -1, D + 1)
    image = signal.new_zeros(B, V, H * W, D + 1).scatter_add(2, index, values * weight[..., None])
    alpha = signal.new_zeros(B, V, H * W).scatter_add(2, pix, weight)
    depth = image[..., D:] / alpha[..., None].clamp(min=1e-6)
    out = torch.cat([image[..., :D], depth], dim=-1)
    return out.view(B, V, H, W, D + 1), alpha.view(B, V, H, W, 1)
