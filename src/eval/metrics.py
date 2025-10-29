"""Metric utilities for evaluating ARC in-context models."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, Optional

import torch
from torch import Tensor


def _ensure_rank(logits: Tensor) -> None:
    """Validate logits tensor shape."""

    if logits.dim() < 3:
        raise ValueError("Expected logits to have shape (batch, queries, classes, ...).")


def _resolve_valid_mask(targets: Tensor, mask: Optional[Tensor], *, ignore_index: int) -> Tensor:
    """Return the boolean mask of valid positions."""

    valid = targets != ignore_index
    if mask is not None:
        if mask.shape != targets.shape:
            raise ValueError("Mask must have the same shape as targets.")
        valid = valid & mask
    return valid


def compute_topk_exact_match(
    logits: Tensor,
    targets: Tensor,
    *,
    mask: Optional[Tensor] = None,
    k: int = 1,
    ignore_index: int = -100,
) -> Tensor:
    """Return a boolean tensor indicating top-k exact matches per query."""

    _ensure_rank(logits)
    if logits.shape[:2] != targets.shape[:2]:
        raise ValueError("Batch and query dimensions must match between logits and targets.")
    if k <= 0:
        raise ValueError("k must be positive for top-k computation.")

    valid = _resolve_valid_mask(targets, mask, ignore_index=ignore_index)

    num_classes = logits.shape[2]
    topk = logits.topk(min(k, num_classes), dim=2).indices  # (B, Q, K, ...)
    expanded_targets = targets.unsqueeze(2).expand_as(topk)
    matches = topk == expanded_targets
    if k == 1:
        per_pixel = matches.squeeze(2)
    else:
        per_pixel = matches.any(dim=2)
    per_pixel = per_pixel | (~valid)

    batch, queries = targets.shape[:2]
    per_example = per_pixel.reshape(batch, queries, -1).all(dim=-1)
    has_valid = valid.reshape(batch, queries, -1).any(dim=-1)

    result = torch.zeros_like(per_example, dtype=torch.bool)
    if has_valid.any():
        result[has_valid] = per_example[has_valid]
    return result


@dataclass
class MetricAccumulator:
    """Accumulate evaluation metrics across batches."""

    top1_correct: float = 0.0
    top1_total: float = 0.0
    top3_correct: float = 0.0
    top3_total: float = 0.0
    exact_match_correct: float = 0.0
    exact_match_total: float = 0.0
    pixel_correct: float = 0.0
    pixel_total: float = 0.0
    _intersections: DefaultDict[int, float] = field(default_factory=lambda: defaultdict(float))
    _unions: DefaultDict[int, float] = field(default_factory=lambda: defaultdict(float))

    def update(
        self,
        *,
        logits: Tensor,
        targets: Tensor,
        mask: Optional[Tensor] = None,
        ignore_index: int = -100,
    ) -> None:
        """Update statistics using the provided batch."""

        _ensure_rank(logits)
        if logits.shape[:2] != targets.shape[:2]:
            raise ValueError("Batch and query dimensions must match between logits and targets.")

        valid = _resolve_valid_mask(targets, mask, ignore_index=ignore_index)
        predictions = logits.argmax(dim=2)

        top1 = compute_topk_exact_match(logits, targets, mask=mask, k=1, ignore_index=ignore_index)
        top3 = compute_topk_exact_match(logits, targets, mask=mask, k=3, ignore_index=ignore_index)

        top1_count = float(top1.sum().item())
        top3_count = float(top3.sum().item())

        valid_queries_mask = valid.reshape(logits.shape[0], logits.shape[1], -1).any(dim=-1)
        total_queries = float(valid_queries_mask.sum().item())

        self.top1_correct += top1_count
        self.top1_total += total_queries
        self.top3_correct += top3_count
        self.top3_total += total_queries

        self.exact_match_correct += top1_count
        self.exact_match_total += total_queries

        valid_flat = valid.reshape(-1)
        if valid_flat.any():
            pred_flat = predictions.reshape(-1)[valid_flat]
            target_flat = targets.reshape(-1)[valid_flat]
            correct = pred_flat == target_flat
            self.pixel_correct += float(correct.sum().item())
            self.pixel_total += float(correct.numel())

            num_classes = logits.shape[2]
            for cls in range(num_classes):
                target_mask = target_flat == cls
                pred_mask = pred_flat == cls
                intersection = float(torch.logical_and(target_mask, pred_mask).sum().item())
                if intersection == 0.0 and not (target_mask.any() or pred_mask.any()):
                    continue
                union = float(target_mask.sum().item() + pred_mask.sum().item() - intersection)
                if union <= 0.0:
                    continue
                self._intersections[cls] += intersection
                self._unions[cls] += union
        else:
            self.pixel_total += 0.0

    def merge(self, other: "MetricAccumulator") -> None:
        """Merge another accumulator into this one."""

        self.top1_correct += other.top1_correct
        self.top1_total += other.top1_total
        self.top3_correct += other.top3_correct
        self.top3_total += other.top3_total
        self.exact_match_correct += other.exact_match_correct
        self.exact_match_total += other.exact_match_total
        self.pixel_correct += other.pixel_correct
        self.pixel_total += other.pixel_total
        for cls, value in other._intersections.items():
            self._intersections[cls] += value
        for cls, value in other._unions.items():
            self._unions[cls] += value

    def compute(self) -> Dict[str, float]:
        """Compute final metric values."""

        def _safe_divide(numerator: float, denominator: float) -> float:
            return numerator / denominator if denominator > 0 else 0.0

        iou_values = [
            intersection / self._unions[cls]
            for cls, intersection in self._intersections.items()
            if self._unions[cls] > 0
        ]
        mean_iou = sum(iou_values) / len(iou_values) if iou_values else 0.0

        return {
            "task_top1": _safe_divide(self.top1_correct, self.top1_total),
            "task_top3": _safe_divide(self.top3_correct, self.top3_total),
            "exact_match_rate": _safe_divide(self.exact_match_correct, self.exact_match_total),
            "pixel_accuracy": _safe_divide(self.pixel_correct, self.pixel_total),
            "mean_iou": mean_iou,
        }


__all__ = ["MetricAccumulator", "compute_topk_exact_match"]
