import math
import random

import pytest
import torch

from orv.geometry import (
    Cameras,
    ChaseCameraFamily,
    chase_camera_to_ego,
    intrinsics_from_fov,
    warp_to_view,
)

W, H = 64, 48


def camera(pose: torch.Tensor) -> Cameras:
    return Cameras(intrinsics_from_fov(60.0, W, H)[None], pose[None], W, H)


def test_project_unproject_roundtrip():
    cam = camera(chase_camera_to_ego(up=5.0, back=8.0, pitch_deg=30.0, yaw_deg=20.0))
    uv = torch.rand(1, 100, 2) * torch.tensor([W, H])
    depth = torch.rand(1, 100) * 50 + 1
    uv2, depth2 = cam.project(cam.unproject(uv, depth))
    assert torch.allclose(uv2, uv, atol=1e-3)
    assert torch.allclose(depth2, depth, atol=1e-3)


def test_chase_camera_is_a_rigid_pose_behind_and_above():
    pose = chase_camera_to_ego(up=6.0, back=10.0, pitch_deg=30.0)
    R = pose[:3, :3]
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-6)
    assert torch.det(R).item() == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(pose[:3, 3], torch.tensor([-10.0, 0.0, 6.0]))


def test_chase_camera_aims_at_ego_when_pitch_matches_geometry():
    pitch = math.degrees(math.atan2(6.0, 10.0))
    cam = camera(chase_camera_to_ego(up=6.0, back=10.0, pitch_deg=pitch))
    uv, z = cam.project(torch.zeros(1, 1, 3))
    assert torch.allclose(uv[0, 0], torch.tensor([W / 2, H / 2]), atol=1e-3)
    assert z.item() == pytest.approx(math.hypot(6.0, 10.0), rel=1e-5)


def test_top_down_view_puts_driving_direction_at_the_top():
    cam = camera(chase_camera_to_ego(up=20.0, back=0.0, pitch_deg=90.0))
    uv, _ = cam.project(torch.tensor([[[5.0, 0.0, 0.0]]]))
    assert uv[0, 0, 0].item() == pytest.approx(W / 2, abs=1e-3)
    assert uv[0, 0, 1].item() < H / 2


def test_family_samples_within_ranges():
    family = ChaseCameraFamily(top_down_prob=0.0)
    ego = torch.eye(4).expand(2, 4, 4)
    cams = family.sample(ego, num_views=3, width=W, height=H, rng=random.Random(0))
    assert cams.K.shape == (2, 3, 3, 3)
    assert cams.cam_to_world.shape == (2, 3, 4, 4)
    height = cams.cam_to_world[..., 2, 3]
    assert ((height >= 3.0) & (height <= 15.0)).all()


def test_warp_into_the_same_view_is_identity():
    cam = camera(chase_camera_to_ego(up=5.0, back=8.0, pitch_deg=30.0))
    image = torch.rand(1, 3, H, W)
    warped, valid = warp_to_view(image, cam, cam, torch.full((1, 1, H, W), 10.0))
    assert valid.all()
    assert torch.allclose(warped, image, atol=1e-3)
