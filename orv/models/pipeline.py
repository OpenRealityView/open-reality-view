"""The full model: surround cameras -> Gaussians -> low-res render in virtual cameras -> decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from orv.data.sample import SurroundSample
from orv.geometry import Cameras, warp_to_view
from orv.models.decoder import RefineDecoder
from orv.models.gaussians import Gaussians
from orv.models.lifter import GaussianLifter
from orv.models.renderer import Render, render


@dataclass
class ViewOutput:
    rgb: torch.Tensor  # [B, V, 3, H, W], full resolution
    render: Render  # intermediate render at 1 / render_scale resolution
    cameras: Cameras  # full-resolution virtual cameras, kept to warp into the next step


class OpenRealityView(nn.Module):
    def __init__(
        self,
        encoder: str = "conv",
        latent_dim: int = 16,
        near: float = 0.5,
        far: float = 200.0,
        render_scale: int = 4,
        decoder_width: int = 64,
        renderer: str = "auto",
    ):
        super().__init__()
        self.lifter = GaussianLifter(encoder, latent_dim, near, far)
        self.decoder = RefineDecoder(latent_dim, decoder_width, upscale=render_scale)
        self.render_scale = render_scale
        self.renderer = renderer

    def forward(
        self, sample: SurroundSample, views: Cameras, prev: ViewOutput | None = None
    ) -> ViewOutput:
        return self.render_views(self.lifter(sample), views, prev)

    def render_views(
        self, gaussians: Gaussians, views: Cameras, prev: ViewOutput | None = None
    ) -> ViewOutput:
        """Decode virtual views [B, V] from lifted Gaussians; one lift can serve many views.

        ``prev`` is the previous time step's output for the same views when streaming.
        """
        s = self.render_scale
        if views.width % s or views.height % s:
            raise ValueError(f"view size {views.width}x{views.height} is not divisible by {s}")
        low = views.scaled(1 / s)
        r = render(gaussians, low, backend=self.renderer)
        B, V = views.K.shape[:2]
        color, features, depth, alpha = (
            x.flatten(0, 1) for x in (r.color, r.features, r.depth, r.alpha)
        )
        if prev is None:
            prev_rgb, prev_valid = torch.zeros_like(color), torch.zeros_like(alpha)
        else:
            prev_cams = prev.cameras.scaled(1 / s).flatten()
            prev_low = F.avg_pool2d(prev.rgb.flatten(0, 1), s)
            prev_rgb, prev_valid = warp_to_view(prev_low, prev_cams, low.flatten(), depth)
            prev_valid = prev_valid * (alpha > 0.5)
        rgb = self.decoder(color, features, depth, alpha, prev_rgb, prev_valid)
        return ViewOutput(rgb=rgb.unflatten(0, (B, V)), render=r, cameras=views)
