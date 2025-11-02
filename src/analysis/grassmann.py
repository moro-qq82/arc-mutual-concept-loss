"""Grassmannian distance utilities for subspace analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

import torch

from src.utils import autocast_disabled, promote_precision, restore_precision


def derive_subspace_basis(
    features: torch.Tensor,
    *,
    rank: int,
    center: bool = True,
) -> torch.Tensor:
    """Compute an orthonormal basis for the dominant subspace of features."""
    matrix = features
    if center:
        matrix = features - features.mean(dim=0, keepdim=True)
    promoted = promote_precision(matrix)
    working = promoted
    if working.dtype in {torch.float16, torch.bfloat16}:
        working = working.to(torch.float32)
    with autocast_disabled(working.device.type):
        u, _, _ = torch.linalg.svd(working, full_matrices=False)
    u = restore_precision(matrix, u)
    if rank > u.shape[1]:
        raise ValueError("`rank` must be smaller than the feature dimension.")
    return u[:, :rank]


def compute_principal_angles(basis_a: torch.Tensor, basis_b: torch.Tensor) -> torch.Tensor:
    """Compute principal angles between two subspaces given orthonormal bases."""
    if basis_a.shape[0] != basis_b.shape[0]:
        raise ValueError("Bases must share the same ambient dimension.")
    if basis_a.device != basis_b.device:
        raise ValueError("Bases must reside on the same device.")
    promoted_a = promote_precision(basis_a)
    promoted_b = promote_precision(basis_b)
    working_a = promoted_a
    working_b = promoted_b
    if working_a.dtype in {torch.float16, torch.bfloat16}:
        working_a = working_a.to(torch.float32)
    if working_b.dtype in {torch.float16, torch.bfloat16}:
        working_b = working_b.to(torch.float32)
    device_type = working_a.device.type
    with autocast_disabled(device_type):
        q_a = torch.linalg.qr(working_a).Q
        q_b = torch.linalg.qr(working_b).Q
    q_a = restore_precision(basis_a, q_a)
    q_b = restore_precision(basis_b, q_b)
    m = q_a.T @ q_b
    singular_values = torch.linalg.svdvals(m)
    singular_values = torch.clamp(singular_values, 0.0, 1.0)
    return torch.arccos(singular_values)


def compute_grassmann_distance(
    basis_a: torch.Tensor,
    basis_b: torch.Tensor,
    *,
    metric: str = "projection",
) -> float:
    """Compute a Grassmannian distance between two subspaces."""
    angles = compute_principal_angles(basis_a, basis_b)
    if metric == "projection":
        return torch.linalg.vector_norm(torch.sin(angles)).item()
    if metric == "chordal":
        return math.sqrt(torch.sum(torch.sin(angles) ** 2).item())
    if metric == "geodesic":
        return torch.linalg.vector_norm(angles).item()
    raise ValueError(f"Unsupported metric: {metric}")


@dataclass
class GrassmannMetrics:
    """Container for pairwise Grassmannian diagnostics."""

    tasks: List[str]
    distances: torch.Tensor
    shared_indices: torch.Tensor

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Convert metrics into a nested dictionary."""
        summary: Dict[str, Dict[str, float]] = {}
        for i, task in enumerate(self.tasks):
            summary[task] = {
                "mean_distance": self.distances[i].mean().item(),
                "shared_index": self.shared_indices[i].item(),
            }
        return summary


class GrassmannAnalyzer:
    """High-level interface for subspace comparison across tasks."""

    def __init__(
        self,
        *,
        rank: int,
        metric: str = "projection",
        center: bool = True,
    ) -> None:
        self.rank = rank
        self.metric = metric
        self.center = center

    def build_bases(self, representations: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Construct orthonormal bases from raw representations."""
        bases: Dict[str, torch.Tensor] = {}
        for task, features in representations.items():
            bases[task] = derive_subspace_basis(
                features, rank=self.rank, center=self.center
            )
        return bases

    def pairwise_metrics(self, bases: Mapping[str, torch.Tensor]) -> GrassmannMetrics:
        """Compute distances and shared indices among tasks."""
        tasks = list(bases.keys())
        if not tasks:
            raise ValueError("No bases provided for analysis.")
        distances = torch.zeros((len(tasks), len(tasks)), dtype=torch.float64)
        shared_indices = torch.zeros(len(tasks), dtype=torch.float64)
        for i, task_i in enumerate(tasks):
            basis_i = bases[task_i]
            for j in range(i, len(tasks)):
                basis_j = bases[tasks[j]]
                distance = compute_grassmann_distance(
                    basis_i, basis_j, metric=self.metric
                )
                distances[i, j] = distance
                distances[j, i] = distance
            alignment = self._shared_index(basis_i, bases.values())
            shared_indices[i] = alignment
        return GrassmannMetrics(tasks=tasks, distances=distances, shared_indices=shared_indices)

    def _shared_index(
        self,
        basis: torch.Tensor,
        others: Iterable[torch.Tensor],
        eps: float = 1e-12,
    ) -> float:
        """Compute the shared subspace index between a basis and the cohort."""
        accum = 0.0
        count = 0
        for other in others:
            if other.data_ptr() == basis.data_ptr():
                continue
            cosines = torch.cos(compute_principal_angles(basis, other)) ** 2
            accum += cosines.mean().item()
            count += 1
        if count == 0:
            return 1.0
        return accum / (count + eps)


__all__ = [
    "GrassmannAnalyzer",
    "GrassmannMetrics",
    "compute_grassmann_distance",
    "compute_principal_angles",
    "derive_subspace_basis",
]
