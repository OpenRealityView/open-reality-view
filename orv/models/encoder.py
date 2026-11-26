"""Image encoders: images [B, 3, H, W] in [0, 1] -> feature maps [B, dim, h, w]."""

from __future__ import annotations

from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn


class ConvEncoder(nn.Module):
    """Tiny stride-8 CNN for tests and CPU debugging."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        layers: list[nn.Module] = []
        for cin, cout in pairwise([3, dim // 4, dim // 2, dim]):
            layers += [nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.GELU()]
        self.net = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class DinoV2Encoder(nn.Module):
    """DINOv2 patch features via torch.hub (Apache-2.0 code and weights); patch size 14."""

    patch = 14

    def __init__(self, name: str = "dinov2_vits14", frozen: bool = True):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", name)
        self.model.requires_grad_(not frozen)
        self.dim = self.model.embed_dim
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, _, H, W = images.shape
        h, w = max(H // self.patch, 1), max(W // self.patch, 1)
        size = (h * self.patch, w * self.patch)
        x = F.interpolate(images, size=size, mode="bilinear", align_corners=False)
        tokens = self.model.forward_features((x - self.mean) / self.std)["x_norm_patchtokens"]
        return tokens.transpose(1, 2).reshape(B, self.dim, h, w)


def build_encoder(name: str) -> nn.Module:
    if name == "conv":
        return ConvEncoder()
    if name.startswith("dinov2"):
        return DinoV2Encoder(name)
    raise ValueError(f"unknown encoder {name!r}")
