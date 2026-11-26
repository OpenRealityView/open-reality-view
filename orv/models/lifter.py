"""Feed-forward lifting of surround images to 3D Gaussians."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from orv.data.sample import SurroundSample
from orv.models.encoder import build_encoder
from orv.models.gaussians import Gaussians


class GaussianLifter(nn.Module):
    """Every encoder feature cell of every camera and frame becomes one pixel-aligned Gaussian.

    Per cell the head predicts depth along the pixel ray, scale, rotation, opacity, color, a latent
    feature vector and a velocity that carries the Gaussian to the reference time. Cross-view and
    temporal attention between encoder and head (STORM-style) is ROADMAP M2.
    """

    def __init__(
        self, encoder: str = "conv", latent_dim: int = 16, near: float = 0.5, far: float = 200.0
    ):
        super().__init__()
        self.encoder = build_encoder(encoder)
        self.latent_dim = latent_dim
        self.near, self.far = near, far
        dim = self.encoder.dim
        # depth 1 + scale 3 + quat 4 + opacity 1 + color 3 + velocity 3 + latent
        self.head = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(), nn.Conv2d(dim, 15 + latent_dim, 1)
        )

    def forward(self, sample: SurroundSample) -> Gaussians:
        B, T, N, H, W = sample.shape
        out = self.head(self.encoder(sample.images.flatten(0, 2)))
        h, w = out.shape[-2:]
        out = out.view(B, T, N, -1, h * w).transpose(-1, -2)  # [B, T, N, h*w, D]
        depth_logit, scale, quat, opacity, color, velocity, latent = out.split(
            [1, 3, 4, 1, 3, 3, self.latent_dim], dim=-1
        )

        # Depth is predicted in disparity space between near and far.
        disparity = 1 / self.far + (1 / self.near - 1 / self.far) * torch.sigmoid(depth_logit)
        depth = 1 / disparity[..., 0]  # [B, T, N, h*w]

        # Centers of the feature cells in input-image pixels.
        ys = (torch.arange(h, device=out.device, dtype=out.dtype) + 0.5) * (H / h)
        xs = (torch.arange(w, device=out.device, dtype=out.dtype) + 0.5) * (W / w)
        v, u = torch.meshgrid(ys, xs, indexing="ij")
        uv = torch.stack([u, v], dim=-1).reshape(h * w, 2).expand(B, T, N, -1, -1)

        means = sample.cameras().unproject(uv, depth)
        # Timestamps are <= 0, so this moves every Gaussian forward to the reference time.
        means = means - velocity * sample.timestamps[:, :, None, None, None]

        # Scales are relative to the cell footprint so that a fresh model roughly tiles the image.
        fx = sample.intrinsics[:, None, :, None, 0, 0]  # [B, 1, N, 1]
        scales = F.softplus(scale) * (depth * (W / w) / fx)[..., None]

        def flat(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(B, T * N * h * w, x.shape[-1])

        return Gaussians(
            means=flat(means),
            quats=flat(F.normalize(quat, dim=-1)),
            scales=flat(scales),
            opacities=flat(torch.sigmoid(opacity))[..., 0],
            colors=flat(torch.sigmoid(color)),
            features=flat(latent),
            velocities=flat(velocity),
        )
