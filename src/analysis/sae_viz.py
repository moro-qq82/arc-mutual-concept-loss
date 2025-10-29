"""Visualization helpers for sparse autoencoder activations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ActivationHeatmapConfig:
    """Configuration for activation heatmap rendering."""

    figsize: Sequence[float] = (8.0, 6.0)
    cmap: str = "magma"
    dpi: int = 150
    annotate: bool = False


def _prepare_matrix(activations: np.ndarray) -> np.ndarray:
    """Normalize activations to zero mean and unit variance."""
    if activations.ndim != 2:
        raise ValueError("Activations must be provided as a 2D array.")
    mean = activations.mean(axis=0, keepdims=True)
    std = activations.std(axis=0, keepdims=True) + 1e-8
    return (activations - mean) / std


def plot_activation_heatmap(
    activations: np.ndarray,
    *,
    task_labels: Optional[Sequence[str]] = None,
    unit_labels: Optional[Sequence[str]] = None,
    output_path: Optional[str] = None,
    config: Optional[ActivationHeatmapConfig] = None,
) -> plt.Figure:
    """Plot a heatmap for SAE activations."""
    cfg = config or ActivationHeatmapConfig()
    matrix = _prepare_matrix(np.asarray(activations))
    if task_labels is not None and len(task_labels) != matrix.shape[0]:
        raise ValueError("`task_labels` length must match the number of samples.")
    if unit_labels is not None and len(unit_labels) != matrix.shape[1]:
        raise ValueError("`unit_labels` length must match the number of SAE units.")
    fig, ax = plt.subplots(figsize=cfg.figsize, dpi=cfg.dpi)
    im = ax.imshow(matrix.T, aspect="auto", cmap=cfg.cmap)
    ax.set_xlabel("Samples")
    ax.set_ylabel("SAE Units")
    if task_labels is not None:
        ax.set_xticks(np.arange(len(task_labels)))
        ax.set_xticklabels(task_labels, rotation=90)
    if unit_labels is not None:
        ax.set_yticks(np.arange(len(unit_labels)))
        ax.set_yticklabels(unit_labels)
    if cfg.annotate and unit_labels is not None and task_labels is not None:
        for i, unit in enumerate(unit_labels):
            for j, task in enumerate(task_labels):
                ax.text(j, i, f"{matrix[j, i]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def _cluster_tasks(similarity: np.ndarray, *, num_groups: Optional[int] = None) -> List[int]:
    """Cluster tasks using agglomerative clustering when possible."""
    similarity = np.asarray(similarity, dtype=float)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("Similarity matrix must be square.")
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for clustering.") from exc
    distance = 1.0 - similarity
    distance = (distance + distance.T) / 2.0
    n_clusters = num_groups or max(2, min(6, similarity.shape[0]))
    try:
        model = AgglomerativeClustering(
            affinity="precomputed", linkage="average", n_clusters=n_clusters
        )
    except TypeError:  # pragma: no cover - compatibility for newer sklearn
        model = AgglomerativeClustering(
            metric="precomputed", linkage="average", n_clusters=n_clusters
        )
    return model.fit_predict(distance)


def plot_task_grouping(
    similarity: np.ndarray,
    task_names: Sequence[str],
    *,
    num_groups: Optional[int] = None,
    output_path: Optional[str] = None,
    config: Optional[ActivationHeatmapConfig] = None,
) -> plt.Figure:
    """Visualize task groupings derived from a similarity matrix."""
    similarity = np.asarray(similarity, dtype=float)
    if len(task_names) != similarity.shape[0]:
        raise ValueError("`task_names` length must match similarity dimensions.")
    cfg = config or ActivationHeatmapConfig(figsize=(6.0, 6.0))
    assignments = _cluster_tasks(similarity, num_groups=num_groups)
    order = np.argsort(assignments)
    ordered_similarity = similarity[order][:, order]
    ordered_tasks = [task_names[idx] for idx in order]
    fig, ax = plt.subplots(figsize=cfg.figsize, dpi=cfg.dpi)
    im = ax.imshow(ordered_similarity, cmap=cfg.cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(ordered_tasks)))
    ax.set_xticklabels(ordered_tasks, rotation=90)
    ax.set_yticks(np.arange(len(ordered_tasks)))
    ax.set_yticklabels(ordered_tasks)
    ax.set_title("Task Similarity")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def save_figure(fig: plt.Figure, output_path: str) -> None:
    """Persist a matplotlib figure to disk, creating directories if needed."""
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")


__all__ = [
    "ActivationHeatmapConfig",
    "plot_activation_heatmap",
    "plot_task_grouping",
    "save_figure",
]
