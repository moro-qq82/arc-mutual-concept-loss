"""Sparse autoencoder reconstruction and sparsity losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class SAELossOutput:
    """Container storing SAE loss and auxiliary statistics."""

    loss: Tensor
    metrics: Dict[str, Tensor]


class SAELoss(nn.Module):
    """Combine reconstruction and sparsity losses for sparse autoencoders."""

    def __init__(
        self,
        *,
        recon_weight: float = 1.0,
        l1_weight: float = 1e-3,
        target_sparsity: float = 0.05,
        adaptation_rate: float = 5.0,
        activation_threshold: float = 1e-3,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.recon_weight = recon_weight
        self.base_l1_weight = l1_weight
        self.target_sparsity = target_sparsity
        self.adaptation_rate = adaptation_rate
        self.activation_threshold = activation_threshold
        self.eps = eps

    def forward(
        self,
        reconstruction: Tensor,
        target: Tensor,
        latent: Tensor,
    ) -> SAELossOutput:
        """Compute the SAE objective value and diagnostics."""

        if reconstruction.shape != target.shape:
            raise ValueError("reconstruction and target tensors must share the same shape.")
        if latent.dim() != 2:
            raise ValueError("latent activations must have shape (batch, latent_dim).")

        recon_loss = F.mse_loss(reconstruction, target)
        l1_loss = latent.abs().mean()

        with torch.no_grad():
            active = (latent.abs() > self.activation_threshold).float()
            current_sparsity = active.mean()
            sparsity_error = current_sparsity - self.target_sparsity
            scaling = 1.0 + self.adaptation_rate * sparsity_error
            scaling = torch.clamp(scaling, min=0.0)
        l1_weight = self.base_l1_weight * scaling
        total_loss = self.recon_weight * recon_loss + l1_weight * l1_loss

        metrics = {
            "reconstruction_loss": recon_loss.detach(),
            "l1_loss": l1_loss.detach(),
            "l1_weight": torch.tensor(float(l1_weight), device=latent.device),
            "sparsity": current_sparsity.detach(),
        }
        return SAELossOutput(loss=total_loss, metrics=metrics)


__all__ = ["SAELoss", "SAELossOutput"]
