"""Task-level loss functions for ARC outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class TaskLossOutput:
    """Container for task loss and auxiliary metrics."""

    loss: Tensor
    metrics: Dict[str, Tensor]


class TaskLoss(nn.Module):
    """Compute supervised losses for ARC query predictions."""

    def __init__(
        self,
        *,
        loss_type: str = "cross_entropy",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        exact_match_weight: float = 0.1,
    ) -> None:
        super().__init__()
        supported = {"cross_entropy", "pixel_accuracy"}
        if loss_type not in supported:
            raise ValueError(f"Unsupported loss_type '{loss_type}'. Expected one of {supported}.")
        self.loss_type = loss_type
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.exact_match_weight = exact_match_weight

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        *,
        mask: Optional[Tensor] = None,
    ) -> TaskLossOutput:
        """Return the loss and metrics for the provided predictions."""

        if logits.dim() < 3:
            raise ValueError("Expected logits to have shape (batch, queries, classes, ...).")
        if logits.shape[:2] != targets.shape[:2]:
            raise ValueError("Batch and query dimensions must match between logits and targets.")

        num_classes = logits.shape[2]
        logits_flat = logits.permute(0, 1, *range(3, logits.dim()), 2).reshape(-1, num_classes)
        targets_flat = targets.reshape(-1)

        if mask is not None:
            if mask.shape != targets.shape:
                raise ValueError("Mask must have the same shape as targets.")
            mask_flat = (mask.reshape(-1) > 0)
            logits_flat = logits_flat[mask_flat]
            targets_flat = targets_flat[mask_flat]

        valid_targets = targets_flat != self.ignore_index

        if logits_flat.shape[0] == 0:
            base_loss = logits.new_tensor(0.0)
        elif self.loss_type == "cross_entropy":
            base_loss = F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=self.ignore_index,
                reduction="mean",
                label_smoothing=self.label_smoothing,
            )
        else:
            with torch.no_grad():
                predictions = logits_flat.argmax(dim=-1)
            if valid_targets.any():
                correct = (predictions[valid_targets] == targets_flat[valid_targets]).float()
                accuracy = correct.mean()
            else:
                accuracy = torch.tensor(0.0, device=logits.device)
            base_loss = 1.0 - accuracy

        with torch.no_grad():
            predictions_full = logits.argmax(dim=2)
            valid_positions = targets != self.ignore_index
            if mask is not None:
                valid_positions = valid_positions & (mask > 0)
            per_pixel_correct = (predictions_full == targets) | (~valid_positions)
            batch, queries = targets.shape[:2]
            per_example_correct = per_pixel_correct.reshape(batch, queries, -1).all(dim=-1)
            has_valid = valid_positions.reshape(batch, queries, -1).any(dim=-1)
            if has_valid.any():
                exact_match_rate = per_example_correct[has_valid].float().mean()
            else:
                exact_match_rate = logits.new_tensor(0.0)
            exact_match_loss = 1.0 - exact_match_rate

        loss = base_loss + self.exact_match_weight * exact_match_loss

        metrics = {
            "task_loss": loss.detach(),
        }

        with torch.no_grad():
            if logits_flat.numel() == 0:
                pixel_accuracy = torch.tensor(0.0, device=logits.device)
            else:
                predictions = logits_flat.argmax(dim=-1)
                if valid_targets.any():
                    correct = (predictions[valid_targets] == targets_flat[valid_targets]).float()
                    pixel_accuracy = correct.mean()
                else:
                    pixel_accuracy = torch.tensor(0.0, device=logits.device)
            metrics["pixel_accuracy"] = pixel_accuracy
            metrics["exact_match_rate"] = exact_match_rate
            metrics["exact_match_loss"] = exact_match_loss

        return TaskLossOutput(loss=loss, metrics=metrics)


__all__ = ["TaskLoss", "TaskLossOutput"]
