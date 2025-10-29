"""Utilities for computing CKA similarities across tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

import torch


def _to_2d(features: torch.Tensor) -> torch.Tensor:
    """Ensure that the features tensor is two-dimensional."""
    if features.ndim == 1:
        return features.unsqueeze(1)
    if features.ndim > 2:
        return features.flatten(start_dim=1)
    return features


def _center_features(features: torch.Tensor) -> torch.Tensor:
    """Center features by subtracting the mean along the sample dimension."""
    mean = features.mean(dim=0, keepdim=True)
    return features - mean


def compute_linear_cka(
    features_x: torch.Tensor,
    features_y: torch.Tensor,
    *,
    center: bool = True,
    eps: float = 1e-12,
) -> float:
    """Compute the linear CKA similarity between two feature matrices."""
    x = _to_2d(features_x).to(dtype=torch.float64)
    y = _to_2d(features_y).to(dtype=torch.float64)
    if x.shape[0] != y.shape[0]:
        raise ValueError("Both feature matrices must have the same number of samples.")
    if center:
        x = _center_features(x)
        y = _center_features(y)
    cov_xy = x.T @ y
    numerator = torch.linalg.matrix_norm(cov_xy, ord="fro") ** 2
    cov_xx = x.T @ x
    cov_yy = y.T @ y
    denom = torch.linalg.matrix_norm(cov_xx, ord="fro") * torch.linalg.matrix_norm(cov_yy, ord="fro")
    return (numerator / (denom + eps)).item()


@dataclass
class CKAMatrix:
    """Container for task-wise CKA similarities."""

    tasks: List[str]
    matrix: torch.Tensor

    def as_dict(self) -> Dict[Tuple[str, str], float]:
        """Return the pairwise similarities as a dictionary."""
        values: Dict[Tuple[str, str], float] = {}
        for i, task_i in enumerate(self.tasks):
            for j, task_j in enumerate(self.tasks):
                values[(task_i, task_j)] = self.matrix[i, j].item()
        return values


def compute_taskwise_cka(
    task_features: Mapping[str, torch.Tensor],
    *,
    center: bool = True,
    eps: float = 1e-12,
) -> CKAMatrix:
    """Compute pairwise CKA similarities for a mapping of task features."""
    tasks: List[str] = list(task_features.keys())
    if not tasks:
        raise ValueError("`task_features` must contain at least one entry.")
    similarities = torch.zeros((len(tasks), len(tasks)), dtype=torch.float64)
    for i, task_i in enumerate(tasks):
        features_i = task_features[task_i]
        for j in range(i, len(tasks)):
            task_j = tasks[j]
            features_j = task_features[task_j]
            similarity = compute_linear_cka(
                features_i, features_j, center=center, eps=eps
            )
            similarities[i, j] = similarity
            similarities[j, i] = similarity
    return CKAMatrix(tasks=tasks, matrix=similarities)


def aggregate_task_distances(matrix: CKAMatrix) -> torch.Tensor:
    """Return 1 - CKA as a distance tensor for downstream clustering."""
    return 1.0 - matrix.matrix


__all__ = ["CKAMatrix", "aggregate_task_distances", "compute_linear_cka", "compute_taskwise_cka"]
