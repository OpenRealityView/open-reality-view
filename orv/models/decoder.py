"""One-step decoder: low-resolution render of a virtual view -> full-resolution frame."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1), nn.GroupNorm(8, cout), nn.SiLU()
    )


class RefineDecoder(nn.Module):
    """Small U-Net that turns the render into an RGB frame ``upscale`` times larger.

    Inputs are the rendered color and latent features, inverse depth, alpha, and the previous
    output warped into the current view with its validity mask (temporal recurrence). Alpha marks
    where nothing was observed and content has to be generated; adversarial and distillation
    training (ROADMAP M3) is what makes the decoder generative.
    """

    def __init__(self, latent_dim: int = 16, width: int = 64, upscale: int = 4):
        super().__init__()
        self.enc = _block(3 + latent_dim + 1 + 1 + 3 + 1, width)
        self.down = nn.Sequential(_block(width, 2 * width, stride=2), _block(2 * width, 2 * width))
        self.fuse = _block(3 * width, width)
        self.out = nn.Sequential(
            nn.Conv2d(width, 3 * upscale**2, 3, padding=1), nn.PixelShuffle(upscale)
        )

    def forward(
        self,
        color: torch.Tensor,
        features: torch.Tensor,
        depth: torch.Tensor,
        alpha: torch.Tensor,
        prev_rgb: torch.Tensor,
        prev_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Inputs are [B, C, h, w] at render resolution; returns RGB [B, 3, h*s, w*s]."""
        inv_depth = alpha / depth.clamp(min=0.5)
        x = self.enc(torch.cat([color, features, inv_depth, alpha, prev_rgb, prev_valid], dim=1))
        y = F.interpolate(self.down(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(self.out(self.fuse(torch.cat([x, y], dim=1))))
