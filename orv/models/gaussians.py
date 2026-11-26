"""Batched 3D Gaussians: the scene representation shared by lifting and rendering."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Gaussians:
    """3D Gaussians in the world frame at the reference time."""

    means: torch.Tensor  # [B, G, 3]
    quats: torch.Tensor  # [B, G, 4], wxyz, normalized
    scales: torch.Tensor  # [B, G, 3], meters
    opacities: torch.Tensor  # [B, G]
    colors: torch.Tensor  # [B, G, 3], RGB in [0, 1]
    features: torch.Tensor  # [B, G, C], latent appearance consumed by the decoder
    velocities: torch.Tensor  # [B, G, 3], m/s in the world frame
