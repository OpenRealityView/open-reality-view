import random

import pytest
import torch

from orv.data import synthetic_sample
from orv.geometry import ChaseCameraFamily
from orv.models import OpenRealityView

# One fixed chase camera 10 m behind and 6 m above the ego vehicle.
FIXED = ChaseCameraFamily(
    up=(6.0, 6.0),
    back=(10.0, 10.0),
    pitch_deg=(30.0, 30.0),
    yaw_deg=(0.0, 0.0),
    fov_deg=(90.0, 90.0),
    top_down_prob=0.0,
)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return OpenRealityView(
        encoder="conv", latent_dim=8, render_scale=4, decoder_width=32, renderer="naive"
    )


@pytest.fixture
def sample():
    return synthetic_sample(batch=2, frames=2, cameras=3, height=48, width=64)


def views_for(sample, width=64, height=32):
    ego = sample.ego_to_world[:, -1]
    return FIXED.sample(ego, num_views=2, width=width, height=height, rng=random.Random(0))


def test_forward_shapes_and_recurrence(model, sample):
    views = views_for(sample)
    first = model(sample, views)
    assert first.rgb.shape == (2, 2, 3, 32, 64)
    assert first.render.depth.shape == (2, 2, 1, 8, 16)
    second = model(sample, views, prev=first)
    assert second.rgb.shape == first.rgb.shape
    assert torch.isfinite(second.rgb).all()


def test_gradients_reach_lifter_and_decoder(model, sample):
    model(sample, views_for(sample)).rgb.mean().backward()
    assert model.decoder.out[0].weight.grad.abs().sum() > 0
    assert model.lifter.head[-1].weight.grad.abs().sum() > 0


def test_view_size_must_be_divisible_by_render_scale(model, sample):
    with pytest.raises(ValueError):
        model(sample, views_for(sample, width=30))
