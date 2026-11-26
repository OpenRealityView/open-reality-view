import torch

from orv.eval import match_detections, psnr


def test_psnr():
    x = torch.rand(2, 3, 8, 8)
    assert (psnr(x, x) >= 99).all()
    zeros, tenth = torch.zeros(1, 3, 4, 4), torch.full((1, 3, 4, 4), 0.1)
    assert torch.allclose(psnr(zeros, tenth), torch.tensor([20.0]))


def test_masked_psnr_ignores_masked_pixels():
    pred, target = torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 4, 4)
    target[..., :2, :] = 1.0
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., 2:, :] = 1.0
    assert psnr(pred, target, mask).item() >= 99


def test_match_detections_counts_misses_and_hallucinations():
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    pred = torch.tensor([[1.0, 1.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
    counts = match_detections(pred, gt)
    assert (counts.matched, counts.missed, counts.hallucinated) == (1, 1, 1)
    assert counts.recall == 0.5
    assert counts.hallucination_rate == 0.5
    total = counts + match_detections(torch.empty(0, 4), gt)
    assert (total.matched, total.missed, total.hallucinated) == (1, 3, 1)
