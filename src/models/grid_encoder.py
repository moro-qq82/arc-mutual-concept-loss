"""Grid encoder module for converting ARC grids into latent embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn


@dataclass
class GridEncoderOutput:
    """Container holding intermediate outputs of the grid encoder."""

    embedding: Tensor
    feature_map: Tensor


class ConvBlock(nn.Module):
    """Simple convolutional processing block."""

    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        """Forward pass applying two convolutions and optional dropout."""

        x = self.block(x)
        return self.dropout(x)


class GridEncoder(nn.Module):
    """Encode discrete ARC grids into dense latent representations."""

    def __init__(
        self,
        *,
        num_colors: int = 10,
        embedding_dim: int = 32,
        hidden_channels: Sequence[int] | Iterable[int] = (64, 128),
        output_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_channels = tuple(hidden_channels)
        if not hidden_channels:
            raise ValueError("GridEncoder requires at least one hidden channel value.")

        self.embedding = nn.Embedding(num_colors, embedding_dim)

        conv_layers = []
        in_channels = embedding_dim
        for hidden in hidden_channels:
            conv_layers.append(ConvBlock(in_channels, hidden, dropout=dropout))
            in_channels = hidden
        self.conv_net = nn.Sequential(*conv_layers)
        self.projection = nn.Linear(in_channels, output_dim)

    @property
    def output_dim(self) -> int:
        """Return the dimensionality of the produced embedding."""

        return self.projection.out_features

    @staticmethod
    def _normalize_inputs(grids: Tensor) -> Tensor:
        """Normalize input tensors to ``(B, H, W)`` long indices."""

        if grids.dim() == 4:
            _, channels, _, _ = grids.shape
            if channels != 1:
                raise ValueError("Grid tensors must have a single channel or be 3D.")
            grids = grids.squeeze(1)
        elif grids.dim() != 3:
            raise ValueError("Grid tensors must have shape (B, H, W) or (B, 1, H, W).")
        if grids.dtype != torch.long:
            grids = grids.long()
        return grids

    def forward(self, grids: Tensor) -> GridEncoderOutput:  # noqa: D401
        """Encode grids into pooled embeddings and dense feature maps."""

        grids = self._normalize_inputs(grids)
        embeddings = self.embedding(grids)
        feature_map = embeddings.permute(0, 3, 1, 2).contiguous()
        feature_map = self.conv_net(feature_map)

        pooled = feature_map.mean(dim=(-2, -1))
        embedding = self.projection(pooled)
        return GridEncoderOutput(embedding=embedding, feature_map=feature_map)


__all__ = ["GridEncoder", "GridEncoderOutput"]
