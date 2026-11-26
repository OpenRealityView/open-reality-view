"""The data contract every dataset adapter produces."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from orv.geometry import Cameras


@dataclass
class SurroundSample:
    """T synchronized frames from N surround cameras; the last frame is the reference time.

    Adapters choose the world frame. Consecutive samples of a stream must share it (for example
    the first ego pose of the sequence) so that previous outputs can be warped into new views.
    """

    images: torch.Tensor  # [B, T, N, 3, H, W], RGB in [0, 1]
    intrinsics: torch.Tensor  # [B, N, 3, 3], pixels at H x W
    cam_to_ego: torch.Tensor  # [B, N, 4, 4]
    ego_to_world: torch.Tensor  # [B, T, 4, 4]
    timestamps: torch.Tensor  # [B, T], seconds relative to the reference frame (last = 0)
    lidar_depth: torch.Tensor | None = None  # [B, T, N, 1, H, W], 0 = no return; training only

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        """(B, T, N, H, W)"""
        B, T, N, _, H, W = self.images.shape
        return B, T, N, H, W

    def to(self, device: torch.device | str) -> SurroundSample:
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        return SurroundSample(**{k: v if v is None else v.to(device) for k, v in values.items()})

    def cameras(self) -> Cameras:
        """Real cameras of every frame, with K and cam_to_world shaped [B, T, N, ...]."""
        B, T, N, H, W = self.shape
        cam_to_world = self.ego_to_world[:, :, None] @ self.cam_to_ego[:, None]
        K = self.intrinsics[:, None].expand(B, T, N, 3, 3)
        return Cameras(K, cam_to_world, W, H)
