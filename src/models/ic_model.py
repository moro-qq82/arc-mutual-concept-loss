"""End-to-end ARC in-context learning model integrating encoder, context, decoder, and SAE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch import Tensor, nn

from .context_encoder import ContextEncoder
from .grid_encoder import GridEncoder, GridEncoderOutput
from .sae import SparseAutoencoder
from .solver import GridSolver


@dataclass
class ARCModelOutput:
    """Container for the forward outputs of :class:`ARCInContextModel`."""

    logits: Tensor
    task_representation: Tensor
    sae_latent: Tensor
    sae_reconstruction: Tensor


class ARCInContextModel(nn.Module):
    """ARC in-context learning model combining encoder, context encoder, SAE, and decoder."""

    def __init__(
        self,
        *,
        num_colors: int = 10,
        grid_embedding_dim: int = 32,
        grid_hidden_channels: Optional[list[int]] = None,
        grid_output_dim: int = 256,
        context_model_dim: int = 256,
        context_heads: int = 8,
        context_layers: int = 4,
        max_context_examples: int = 10,
        sae_latent_dim: int = 512,
        decoder_model_dim: int = 256,
        decoder_heads: int = 8,
        decoder_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if grid_hidden_channels is None:
            grid_hidden_channels = [64, 128]

        self.grid_encoder = GridEncoder(
            num_colors=num_colors,
            embedding_dim=grid_embedding_dim,
            hidden_channels=grid_hidden_channels,
            output_dim=grid_output_dim,
            dropout=dropout,
        )
        self.context_encoder = ContextEncoder(
            input_dim=grid_output_dim * 2,
            model_dim=context_model_dim,
            num_heads=context_heads,
            num_layers=context_layers,
            dropout=dropout,
            max_context_examples=max_context_examples,
        )
        self.sae = SparseAutoencoder(input_dim=context_model_dim, latent_dim=sae_latent_dim)
        last_hidden_channels = grid_hidden_channels[-1]
        self.decoder = GridSolver(
            input_channels=last_hidden_channels,
            task_dim=context_model_dim,
            num_colors=num_colors,
            model_dim=decoder_model_dim,
            num_heads=decoder_heads,
            num_layers=decoder_layers,
            dropout=dropout,
        )

    def _encode_support(self, grids: Tensor) -> Tensor:
        """Encode support grids using the shared grid encoder."""

        batch, shots, height, width = grids.shape
        encoded = self.grid_encoder(grids.view(batch * shots, height, width))
        embeddings = encoded.embedding.view(batch, shots, -1)
        return embeddings

    def _encode_queries(self, grids: Tensor) -> tuple[GridEncoderOutput, int, int, int, int]:
        """Encode query grids for decoding."""

        batch, queries, height, width = grids.shape
        encoded = self.grid_encoder(grids.view(batch * queries, height, width))
        return encoded, batch, queries, height, width

    def forward(
        self,
        *,
        support_inputs: Tensor,
        support_outputs: Tensor,
        query_inputs: Tensor,
        support_mask: Optional[Tensor] = None,
    ) -> ARCModelOutput:
        """Perform forward inference for ARC in-context prediction."""

        support_in_embeddings = self._encode_support(support_inputs)
        support_out_embeddings = self._encode_support(support_outputs)
        context_repr = self.context_encoder(
            support_in_embeddings,
            support_out_embeddings,
            context_mask=support_mask,
        )
        sae_outputs = self.sae(context_repr)

        query_encoded, batch, num_queries, height, width = self._encode_queries(query_inputs)
        task_repr = sae_outputs.reconstruction
        expanded_task_repr = task_repr.unsqueeze(1).expand(batch, num_queries, -1).reshape(batch * num_queries, -1)

        logits = self.decoder(expanded_task_repr, query_encoded)
        logits = logits.view(batch, num_queries, -1, height, width)
        return ARCModelOutput(
            logits=logits,
            task_representation=context_repr,
            sae_latent=sae_outputs.latent,
            sae_reconstruction=sae_outputs.reconstruction,
        )

    def predict(self, *, support_inputs: Tensor, support_outputs: Tensor, query_inputs: Tensor, support_mask: Optional[Tensor] = None) -> Tensor:
        """Return predicted color logits for the query grids."""

        outputs = self.forward(
            support_inputs=support_inputs,
            support_outputs=support_outputs,
            query_inputs=query_inputs,
            support_mask=support_mask,
        )
        return outputs.logits


__all__ = ["ARCInContextModel", "ARCModelOutput"]
