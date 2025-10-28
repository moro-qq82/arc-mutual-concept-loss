"""Factory utilities for composing ARC training losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from torch import Tensor

from ..losses import (
    SAELoss,
    SAELossOutput,
    ShareRegularizerOutput,
    ShareSubspaceRegularizer,
    TaskLoss,
    TaskLossOutput,
)


@dataclass
class LossFactoryConfig:
    """Configuration values for assembling the loss functions."""

    task_loss_type: str = "cross_entropy"
    task_ignore_index: int = -100
    task_label_smoothing: float = 0.0
    task_exact_match_weight: float = 0.1
    sae_recon_weight: float = 1.0
    sae_l1_weight: float = 1e-3
    sae_target_sparsity: float = 0.05
    sae_adaptation_rate: float = 5.0
    sae_activation_threshold: float = 1e-3
    share_rank: int = 8
    share_activation_threshold: float = 1e-3
    share_min_group_size: int = 4
    alpha: float = 1.0
    beta: float = 1.0


class LossFactory:
    """Compose the different loss components required for training."""

    def __init__(self, config: Optional[LossFactoryConfig] = None) -> None:
        if config is None:
            config = LossFactoryConfig()
        self.config = config
        self.task_loss = TaskLoss(
            loss_type=config.task_loss_type,
            ignore_index=config.task_ignore_index,
            label_smoothing=config.task_label_smoothing,
            exact_match_weight=config.task_exact_match_weight,
        )
        self.sae_loss = SAELoss(
            recon_weight=config.sae_recon_weight,
            l1_weight=config.sae_l1_weight,
            target_sparsity=config.sae_target_sparsity,
            adaptation_rate=config.sae_adaptation_rate,
            activation_threshold=config.sae_activation_threshold,
        )
        self.share_regularizer = ShareSubspaceRegularizer(
            rank=config.share_rank,
            activation_threshold=config.share_activation_threshold,
            min_group_size=config.share_min_group_size,
        )
        self.alpha = config.alpha
        self.beta = config.beta

    def __call__(
        self,
        *,
        logits: Tensor,
        targets: Tensor,
        sae_reconstruction: Tensor,
        task_representation: Tensor,
        sae_latent: Tensor,
        mask: Optional[Tensor] = None,
        group_assignments: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute the aggregate loss and expose individual components."""

        task_output: TaskLossOutput = self.task_loss(logits, targets, mask=mask)
        sae_output: SAELossOutput = self.sae_loss(sae_reconstruction, task_representation, sae_latent)
        share_output: ShareRegularizerOutput = self.share_regularizer(
            sae_latent,
            assignments=group_assignments,
        )

        total_loss = task_output.loss + self.beta * sae_output.loss + self.alpha * share_output.loss

        losses: Dict[str, Tensor] = {
            "total_loss": total_loss,
            "task_loss": task_output.loss.detach(),
            "sae_loss": sae_output.loss.detach(),
            "share_loss": share_output.loss.detach(),
        }
        losses.update(task_output.metrics)
        losses.update({f"sae_{k}": v for k, v in sae_output.metrics.items()})
        losses.update({f"share_{k}": v for k, v in share_output.metrics.items()})
        return losses


__all__ = ["LossFactory", "LossFactoryConfig"]
