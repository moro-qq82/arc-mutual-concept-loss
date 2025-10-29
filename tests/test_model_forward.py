"""Tests for the ARCInContextModel forward pass and prediction interface."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.ic_model import ARCInContextModel


def _build_model() -> ARCInContextModel:
    return ARCInContextModel(
        num_colors=3,
        grid_embedding_dim=8,
        grid_hidden_channels=[8],
        grid_output_dim=16,
        context_model_dim=16,
        context_heads=4,
        context_layers=1,
        max_context_examples=4,
        sae_latent_dim=12,
        decoder_model_dim=16,
        decoder_heads=4,
        decoder_layers=1,
        dropout=0.0,
    )


def _random_grids(batch: int, shots: int, height: int, width: int, num_colors: int) -> torch.Tensor:
    return torch.randint(0, num_colors, (batch, shots, height, width))


def test_model_forward_shapes() -> None:
    torch.manual_seed(0)
    batch, shots, queries, height, width = 2, 3, 2, 4, 4
    num_colors = 3
    model = _build_model()

    support_inputs = _random_grids(batch, shots, height, width, num_colors)
    support_outputs = _random_grids(batch, shots, height, width, num_colors)
    query_inputs = _random_grids(batch, queries, height, width, num_colors)

    outputs = model(
        support_inputs=support_inputs,
        support_outputs=support_outputs,
        query_inputs=query_inputs,
    )

    assert outputs.logits.shape == (batch, queries, num_colors, height, width)
    assert outputs.task_representation.shape[0] == batch
    latent_dim = model.sae.encoder.out_features
    input_dim = model.sae.decoder.out_features
    assert outputs.sae_latent.shape == (batch, latent_dim)
    assert outputs.sae_reconstruction.shape == (batch, input_dim)


def test_model_predict_matches_forward_logits() -> None:
    torch.manual_seed(42)
    batch, shots, queries, height, width = 1, 2, 1, 3, 3
    num_colors = 3
    model = _build_model()

    support_inputs = _random_grids(batch, shots, height, width, num_colors)
    support_outputs = _random_grids(batch, shots, height, width, num_colors)
    query_inputs = _random_grids(batch, queries, height, width, num_colors)

    forward_outputs = model(
        support_inputs=support_inputs,
        support_outputs=support_outputs,
        query_inputs=query_inputs,
    )
    predict_logits = model.predict(
        support_inputs=support_inputs,
        support_outputs=support_outputs,
        query_inputs=query_inputs,
    )

    assert torch.allclose(forward_outputs.logits, predict_logits)
