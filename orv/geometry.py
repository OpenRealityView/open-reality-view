"""Cameras, the virtual chase-camera family and depth-based warping.

Conventions: the ego frame is x forward, y left, z up with its origin on the ground below the
rear-axle center. Cameras use the OpenCV convention (x right, y down, z forward), which is also
what gsplat expects. Poses are 4x4 rigid transforms named ``<from>_to_<to>``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F


@dataclass
class Cameras:
    """Pinhole cameras with arbitrary leading batch dimensions and one shared image size."""

    K: torch.Tensor  # [..., 3, 3], pixels
    cam_to_world: torch.Tensor  # [..., 4, 4]
    width: int
    height: int

    def world_to_cam(self) -> torch.Tensor:
        return torch.linalg.inv(self.cam_to_world)

    def scaled(self, factor: float) -> Cameras:
        """The same cameras with the image resized by ``factor``."""
        K = self.K.clone()
        K[..., :2, :] *= factor
        return replace(
            self, K=K, width=round(self.width * factor), height=round(self.height * factor)
        )

    def flatten(self) -> Cameras:
        """Merge all leading dimensions into one."""
        return replace(self, K=self.K.flatten(0, -3), cam_to_world=self.cam_to_world.flatten(0, -3))

    def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """World points [..., P, 3] -> pixel coordinates [..., P, 2] and z-depth [..., P]."""
        w2c = self.world_to_cam()
        p = points @ w2c[..., :3, :3].transpose(-1, -2) + w2c[..., None, :3, 3]
        z = p[..., 2]
        uv = (p @ self.K.transpose(-1, -2))[..., :2] / z.clamp(min=1e-6)[..., None]
        return uv, z

    def unproject(self, uv: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Pixel coordinates [..., P, 2] and z-depth [..., P] -> world points [..., P, 3]."""
        rays = F.pad(uv, (0, 1), value=1.0) @ torch.linalg.inv(self.K).transpose(-1, -2)
        p = rays * depth[..., None]
        c2w = self.cam_to_world
        return p @ c2w[..., :3, :3].transpose(-1, -2) + c2w[..., None, :3, 3]


def intrinsics_from_fov(fov_x_deg: float, width: int, height: int) -> torch.Tensor:
    """Pinhole intrinsics with square pixels and a centered principal point."""
    f = 0.5 * width / math.tan(math.radians(fov_x_deg) / 2)
    return torch.tensor([[f, 0.0, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]])


def chase_camera_to_ego(
    up: float, back: float, pitch_deg: float, yaw_deg: float = 0.0
) -> torch.Tensor:
    """Pose of a virtual camera ``back`` m behind and ``up`` m above the ego origin.

    The camera is pitched down by ``pitch_deg`` and orbits the vertical axis by ``yaw_deg``; it
    never rolls. ``pitch_deg=90`` gives a top-down view with the driving direction at the top.
    """
    pitch, yaw = math.radians(pitch_deg), math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    rot_z = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    forward = rot_z @ torch.tensor([math.cos(pitch), 0.0, -math.sin(pitch)])
    right = rot_z @ torch.tensor([0.0, -1.0, 0.0])
    down = torch.linalg.cross(forward, right)
    pose = torch.eye(4)
    pose[:3, :3] = torch.stack([right, down, forward], dim=1)
    pose[:3, 3] = rot_z @ torch.tensor([-back, 0.0, up])
    return pose


@dataclass
class ChaseCameraFamily:
    """Distribution of virtual cameras around the ego vehicle for training and evaluation."""

    up: tuple[float, float] = (3.0, 15.0)
    back: tuple[float, float] = (5.0, 20.0)
    pitch_deg: tuple[float, float] = (15.0, 60.0)
    yaw_deg: tuple[float, float] = (-30.0, 30.0)
    fov_deg: tuple[float, float] = (50.0, 90.0)
    top_down_prob: float = 0.1

    def sample(
        self,
        ego_to_world: torch.Tensor,
        num_views: int,
        width: int,
        height: int,
        rng: random.Random | None = None,
    ) -> Cameras:
        """Random cameras, ``num_views`` per ego pose [B, 4, 4] -> Cameras [B, V]."""
        rng = rng or random.Random()
        poses, Ks = [], []
        for _ in range(ego_to_world.shape[0] * num_views):
            if rng.random() < self.top_down_prob:
                pose = chase_camera_to_ego(rng.uniform(*self.up), 0.0, 90.0)
            else:
                pose = chase_camera_to_ego(
                    rng.uniform(*self.up),
                    rng.uniform(*self.back),
                    rng.uniform(*self.pitch_deg),
                    rng.uniform(*self.yaw_deg),
                )
            poses.append(pose)
            Ks.append(intrinsics_from_fov(rng.uniform(*self.fov_deg), width, height))
        shape = (ego_to_world.shape[0], num_views)
        cam_to_ego = torch.stack(poses).view(*shape, 4, 4).to(ego_to_world)
        K = torch.stack(Ks).view(*shape, 3, 3).to(ego_to_world)
        return Cameras(K, ego_to_world[:, None] @ cam_to_ego, width, height)


def warp_to_view(
    image: torch.Tensor, src: Cameras, dst: Cameras, dst_depth: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-warp ``image`` [B, C, H', W'] seen by ``src`` into ``dst``.

    Uses the z-depth rendered in ``dst`` ([B, 1, H, W]) and assumes a static scene. Both cameras
    have a single batch dimension and live in the same world frame. Returns the warped image and
    a validity mask [B, 1, H, W].
    """
    B, _, H, W = dst_depth.shape
    v, u = torch.meshgrid(
        torch.arange(H, device=dst_depth.device, dtype=dst_depth.dtype) + 0.5,
        torch.arange(W, device=dst_depth.device, dtype=dst_depth.dtype) + 0.5,
        indexing="ij",
    )
    uv = torch.stack([u, v], dim=-1).reshape(1, H * W, 2).expand(B, -1, -1)
    src_uv, src_z = src.project(dst.unproject(uv, dst_depth.reshape(B, H * W)))
    grid = src_uv / src_uv.new_tensor([src.width, src.height]) * 2 - 1
    warped = F.grid_sample(image, grid.view(B, H, W, 2), align_corners=False)
    valid = (src_z > 1e-3) & (grid.abs() <= 1).all(dim=-1)
    return warped, valid.view(B, 1, H, W).to(image.dtype)
