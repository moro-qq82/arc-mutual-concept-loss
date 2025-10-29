"""Analysis utilities for representation diagnostics."""

from .cka import compute_linear_cka, compute_taskwise_cka
from .grassmann import (
    GrassmannAnalyzer,
    compute_grassmann_distance,
    compute_principal_angles,
    derive_subspace_basis,
)
from .sae_viz import (
    ActivationHeatmapConfig,
    plot_activation_heatmap,
    plot_task_grouping,
    save_figure,
)

__all__ = [
    "ActivationHeatmapConfig",
    "GrassmannAnalyzer",
    "compute_grassmann_distance",
    "compute_linear_cka",
    "compute_principal_angles",
    "compute_taskwise_cka",
    "derive_subspace_basis",
    "plot_activation_heatmap",
    "plot_task_grouping",
    "save_figure",
]
