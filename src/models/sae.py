"""Sparse autoencoder module for task representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from torch import Tensor, nn


@dataclass
class SparseAutoencoderOutput:
    """Container for SAE forward pass results."""

    latent: Tensor
    reconstruction: Tensor


class SparseAutoencoder(nn.Module):
    """Two-layer sparse autoencoder with configurable activation."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        *,
        activation: Optional[Callable[[Tensor], Tensor]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=bias)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=bias)
        self.activation = activation if activation is not None else nn.ReLU()

    def encode(self, inputs: Tensor) -> Tensor:
        """Encode inputs into sparse latent representations."""

        latent = self.encoder(inputs)
        latent = self.activation(latent)
        return latent

    def decode(self, latent: Tensor) -> Tensor:
        """Decode latent variables back into the input space."""

        return self.decoder(latent)

    def forward(self, inputs: Tensor) -> SparseAutoencoderOutput:  # noqa: D401
        """Run the autoencoder forward pass."""

        latent = self.encode(inputs)
        reconstruction = self.decode(latent)
        return SparseAutoencoderOutput(latent=latent, reconstruction=reconstruction)

    def compute_losses(
        self,
        inputs: Tensor,
        *,
        l1_coefficient: float,
        reconstruction_weight: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute reconstruction and sparsity losses."""

        outputs = self.forward(inputs)
        reconstruction_loss = nn.functional.mse_loss(outputs.reconstruction, inputs)
        l1_loss = outputs.latent.abs().mean()
        total = reconstruction_weight * reconstruction_loss + l1_coefficient * l1_loss
        return total, reconstruction_loss, l1_loss


__all__ = ["SparseAutoencoder", "SparseAutoencoderOutput"]
