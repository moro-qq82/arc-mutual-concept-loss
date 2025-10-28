"""Transformer-based encoder for ARC k-shot context examples."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


def _build_sinusoidal_positional_encoding(max_len: int, dim: int) -> Tensor:
    """Create sinusoidal positional encodings."""

    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim))
    encoding = torch.zeros(max_len, dim, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    return encoding


class ContextEncoder(nn.Module):
    """Encode support examples into a task-level representation."""

    def __init__(
        self,
        input_dim: int,
        *,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_context_examples: int = 10,
        add_task_token: bool = True,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")

        self.add_task_token = add_task_token
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.task_token = nn.Parameter(torch.zeros(1, 1, model_dim)) if add_task_token else None
        self.register_buffer("positional_encoding", _build_sinusoidal_positional_encoding(max_context_examples + 1, model_dim), persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        input_embeddings: Tensor,
        output_embeddings: Tensor,
        *,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass producing the task representation."""

        if input_embeddings.shape != output_embeddings.shape:
            raise ValueError("Input and output embeddings must share the same shape.")
        batch_size, num_examples, _ = input_embeddings.shape

        pair_embeddings = torch.cat([input_embeddings, output_embeddings], dim=-1)
        tokens = self.input_projection(pair_embeddings)

        if context_mask is not None:
            pool_mask = context_mask
        else:
            pool_mask = None

        if self.add_task_token:
            task_token = self.task_token.expand(batch_size, -1, -1)
            tokens = torch.cat([task_token, tokens], dim=1)
            if context_mask is not None:
                ones = torch.ones(batch_size, 1, device=context_mask.device, dtype=context_mask.dtype)
                context_mask = torch.cat([ones, context_mask], dim=1)
                pool_mask = context_mask
        positions = self.positional_encoding[: tokens.size(1)].unsqueeze(0)
        tokens = tokens + positions

        if context_mask is not None:
            if context_mask.dtype != torch.bool:
                key_padding_mask = context_mask > 0
            else:
                key_padding_mask = context_mask
            src_key_padding_mask = ~key_padding_mask
        else:
            src_key_padding_mask = None

        encoded = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)
        encoded = self.layer_norm(encoded)

        if self.add_task_token:
            task_repr = encoded[:, 0]
        else:
            if context_mask is None:
                task_repr = encoded.mean(dim=1)
            else:
                mask = pool_mask
                if mask.dtype != encoded.dtype:
                    mask = mask.float()
                task_repr = (encoded * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return task_repr


__all__ = ["ContextEncoder"]
