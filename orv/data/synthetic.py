"""Random but geometrically consistent inputs for smoke tests and CPU debugging."""

from __future__ import annotations

import torch

from orv.data.sample import SurroundSample
from orv.geometry import chase_camera_to_ego, intrinsics_from_fov


def synthetic_sample(
    batch: int = 1,
    frames: int = 2,
    cameras: int = 6,
    height: int = 64,
    width: int = 96,
    fps: float = 10.0,
    speed: float = 10.0,
    generator: torch.Generator | None = None,
) -> SurroundSample:
    """Ego drives straight along +x at ``speed`` m/s with a ring of cameras 1.6 m above ground."""
    K = intrinsics_from_fov(min(360.0 / cameras + 20.0, 120.0), width, height)
    rig = torch.stack(
        [
            chase_camera_to_ego(up=1.6, back=0.0, pitch_deg=0.0, yaw_deg=360.0 * i / cameras)
            for i in range(cameras)
        ]
    )
    t = (torch.arange(frames, dtype=torch.float32) - (frames - 1)) / fps
    ego_to_world = torch.eye(4).repeat(batch, frames, 1, 1)
    ego_to_world[..., 0, 3] = speed * t
    return SurroundSample(
        images=torch.rand(batch, frames, cameras, 3, height, width, generator=generator),
        intrinsics=K.expand(batch, cameras, 3, 3).clone(),
        cam_to_ego=rig.expand(batch, -1, -1, -1).clone(),
        ego_to_world=ego_to_world,
        timestamps=t.expand(batch, -1).clone(),
    )
