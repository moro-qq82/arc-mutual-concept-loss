"""Tests for the ShareSubspaceRegularizer module."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.losses.share_regularizer import ShareSubspaceRegularizer


def test_regularizer_returns_zero_when_insufficient_groups() -> None:
    activations = torch.randn(3, 4)
    regularizer = ShareSubspaceRegularizer(rank=2, min_group_size=3)

    output = regularizer(activations)

    assert output.loss.item() == pytest.approx(0.0, abs=1e-6)
    assert output.metrics["share_groups"].item() == pytest.approx(0.0, abs=1e-6)


def test_regularizer_with_assignments_produces_zero_loss_for_identical_subspaces() -> None:
    base = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ]
    )
    activations = torch.cat([base, base], dim=0)
    assignments = torch.tensor([0, 0, 0, 1, 1, 1])
    regularizer = ShareSubspaceRegularizer(rank=1, min_group_size=3)

    output = regularizer(activations, assignments=assignments)

    assert output.loss.item() == pytest.approx(0.0, abs=1e-6)
    assert output.metrics["share_groups"].item() == pytest.approx(2.0, abs=1e-6)


def test_regularizer_positive_loss_for_misaligned_groups() -> None:
    activations = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.5],
        ]
    )
    assignments = torch.tensor([0, 0, 1, 1, 2, 2])
    regularizer = ShareSubspaceRegularizer(rank=1, min_group_size=2)

    output = regularizer(activations, assignments=assignments)

    assert output.loss.item() > 0
    assert output.metrics["share_groups"].item() == pytest.approx(3.0, abs=1e-6)
    assert output.metrics["mean_rank"].item() == pytest.approx(1.0, abs=1e-6)
