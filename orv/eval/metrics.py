"""Evaluation metrics. Faithfulness compares detections on rendered views with ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def psnr(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """PSNR in dB per image for [..., C, H, W] in [0, 1]; ``mask`` [..., 1, H, W] selects pixels."""
    err = (pred - target) ** 2
    if mask is None:
        mse = err.mean(dim=(-3, -2, -1))
    else:
        mask = mask.expand_as(err)
        mse = (err * mask).sum(dim=(-3, -2, -1)) / mask.sum(dim=(-3, -2, -1)).clamp(min=1)
    return -10 * torch.log10(mse.clamp(min=1e-10))


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU of boxes [P, 4] and [G, 4] given as (x1, y1, x2, y2) -> [P, G]."""
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    return inter / (area_a[:, None] + area_b[None] - inter).clamp(min=1e-9)


@dataclass
class FaithfulnessCounts:
    """Detect-on-render counts for one class; sum over frames before reading the rates."""

    matched: int = 0
    missed: int = 0  # ground-truth objects absent from the render
    hallucinated: int = 0  # objects in the render that do not exist

    def __add__(self, other: FaithfulnessCounts) -> FaithfulnessCounts:
        return FaithfulnessCounts(
            self.matched + other.matched,
            self.missed + other.missed,
            self.hallucinated + other.hallucinated,
        )

    @property
    def recall(self) -> float:
        return self.matched / max(self.matched + self.missed, 1)

    @property
    def hallucination_rate(self) -> float:
        return self.hallucinated / max(self.matched + self.hallucinated, 1)


def match_detections(
    pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, iou_threshold: float = 0.5
) -> FaithfulnessCounts:
    """Greedily match detections on a rendered view to ground truth projected into that view.

    ``pred_boxes`` [P, 4] must be sorted by descending detector score; ``gt_boxes`` is [G, 4].
    """
    taken = torch.zeros(len(gt_boxes), dtype=torch.bool, device=gt_boxes.device)
    matched = 0
    if len(pred_boxes) and len(gt_boxes):
        for row in box_iou(pred_boxes, gt_boxes):
            ious = row.masked_fill(taken, -1.0)
            j = int(ious.argmax())
            if ious[j] >= iou_threshold:
                taken[j] = True
                matched += 1
    return FaithfulnessCounts(matched, len(gt_boxes) - matched, len(pred_boxes) - matched)
