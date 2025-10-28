"""Decoder module that generates output grids conditioned on task context."""

from __future__ import annotations

from math import log
from typing import Optional

import torch
from torch import Tensor, nn

from .grid_encoder import GridEncoderOutput


def _build_positional_encoding(length: int, dim: int, device: Optional[torch.device] = None) -> Tensor:
    """Create 1D sinusoidal positional encodings."""

    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-log(10000.0) / max(1, dim)))
    encoding = torch.zeros(length, dim, dtype=torch.float32, device=device)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    return encoding


class GridSolver(nn.Module):
    """Generate ARC grid predictions using a transformer decoder."""

    def __init__(
        self,
        *,
        input_channels: int,
        task_dim: int,
        num_colors: int = 10,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")

        self.input_projection = nn.Linear(input_channels, model_dim)
        self.task_projection = nn.Linear(task_dim, model_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(model_dim, num_colors)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, task_repr: Tensor, encoded_grid: GridEncoderOutput) -> Tensor:
        """Decode logits for each cell of the target grid."""

        feature_map = encoded_grid.feature_map
        batch_size, channels, height, width = feature_map.shape
        device = feature_map.device
        memory = self.task_projection(task_repr).unsqueeze(1)

        tokens = feature_map.flatten(2).transpose(1, 2)
        tokens = self.input_projection(tokens)
        pos = _build_positional_encoding(height * width, tokens.size(-1), device=device)
        tokens = tokens + pos.unsqueeze(0)

        decoded = self.transformer(tokens, memory)
        decoded = self.layer_norm(decoded)
        logits = self.output_layer(decoded)
        logits = logits.transpose(1, 2).view(batch_size, -1, height, width)
        return logits


__all__ = ["GridSolver"]
