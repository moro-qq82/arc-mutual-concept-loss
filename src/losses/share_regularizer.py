"""Share-subspace regularizer encouraging aligned SAE activations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class ShareRegularizerOutput:
    """Container holding the share-regularization loss and diagnostics."""

    loss: Tensor
    metrics: Dict[str, Tensor]


class ShareSubspaceRegularizer(nn.Module):
    """Encourage sparse autoencoder groups to share a low-dimensional subspace."""

    def __init__(
        self,
        *,
        rank: int = 8,
        activation_threshold: float = 1e-3,
        min_group_size: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.rank = rank
        self.activation_threshold = activation_threshold
        self.min_group_size = min_group_size
        self.eps = eps

    def _build_groups(self, activations: Tensor, assignments: Optional[Tensor]) -> List[Tensor]:
        if assignments is not None:
            if assignments.dim() != 1:
                raise ValueError("assignments must be a 1D tensor of group indices.")
            if assignments.shape[0] != activations.shape[0]:
                raise ValueError("assignments length must match batch dimension of activations.")
            groups: Dict[int, List[int]] = {}
            for idx, group_id in enumerate(assignments.tolist()):
                groups.setdefault(int(group_id), []).append(idx)
            return [torch.tensor(indices, device=activations.device, dtype=torch.long) for indices in groups.values()]

        with torch.no_grad():
            binary = (activations.abs() > self.activation_threshold).to(torch.uint8)
        patterns: Dict[Tuple[int, ...], List[int]] = {}
        for idx, row in enumerate(binary):
            key = tuple(row.tolist())
            patterns.setdefault(key, []).append(idx)
        return [torch.tensor(indices, device=activations.device, dtype=torch.long) for indices in patterns.values()]

    def _group_projector(self, activations: Tensor) -> Tuple[Tensor, int]:
        centered = activations - activations.mean(dim=0, keepdim=True)
        if centered.shape[0] == 1:
            norm = torch.linalg.norm(centered, dim=-1, keepdim=True) + self.eps
            unit = centered / norm
            projector = unit.transpose(0, 1) @ unit
            return projector, 1
        rank = min(self.rank, centered.shape[0], centered.shape[1])
        if rank <= 0:
            zero = torch.zeros(
                (centered.shape[1], centered.shape[1]), device=centered.device, dtype=centered.dtype
            )
            return zero, 0

        # Some CUDA linear-algebra kernels (QR/SVD) are not implemented for bfloat16.
        # To avoid NotImplementedError on e.g. geqrf_cuda for BFloat16, perform the
        # low-rank decomposition in float32 and then cast the resulting projector
        # back to the original dtype. This keeps the rest of the pipeline in the
        # model's dtype while ensuring decomposition succeeds.
        orig_dtype = centered.dtype
        need_cast = centered.is_cuda and orig_dtype == torch.bfloat16
        centered_for_svd = centered.to(torch.float32) if need_cast else centered

        _, _, v = torch.pca_lowrank(centered_for_svd, q=rank)
        components = v[:, :rank]
        projector = components @ components.transpose(0, 1)

        if need_cast:
            # cast projector back to the original dtype (e.g. bfloat16)
            projector = projector.to(orig_dtype)

        return projector, components.shape[1]

    def forward(
        self,
        activations: Tensor,
        *,
        assignments: Optional[Tensor] = None,
    ) -> ShareRegularizerOutput:
        """Compute the share-subspace regularization term."""

        if activations.dim() != 2:
            raise ValueError("activations must be a 2D tensor of shape (batch, latent_dim).")
        groups = self._build_groups(activations, assignments)
        valid_projectors: List[Tensor] = []
        weights: List[Tensor] = []
        effective_ranks: List[int] = []
        for indices in groups:
            if indices.numel() < self.min_group_size:
                continue
            group_acts = activations.index_select(0, indices)
            projector, rank = self._group_projector(group_acts)
            valid_projectors.append(projector)
            weights.append(torch.tensor(float(indices.numel()), device=activations.device, dtype=activations.dtype))
            effective_ranks.append(rank)

        if len(valid_projectors) < 2:
            zero = activations.new_tensor(0.0)
            metrics = {
                "share_loss": zero,
                "share_groups": activations.new_tensor(0.0),
            }
            return ShareRegularizerOutput(loss=zero, metrics=metrics)

        stacked = torch.stack(valid_projectors)
        weight_tensor = torch.stack(weights)
        norm_weights = weight_tensor / weight_tensor.sum()
        mean_projector = torch.einsum("g,gij->ij", norm_weights, stacked)

        losses: List[Tensor] = []
        for projector, weight in zip(stacked, norm_weights):
            diff = projector - mean_projector
            losses.append(weight * torch.linalg.matrix_norm(diff, ord="fro") ** 2)
        loss = torch.stack(losses).sum()

        metrics = {
            "share_loss": loss.detach(),
            "share_groups": torch.tensor(
                float(len(valid_projectors)), device=activations.device, dtype=activations.dtype
            ),
            "mean_rank": torch.tensor(
                float(sum(effective_ranks) / len(effective_ranks)),
                device=activations.device,
                dtype=activations.dtype,
            ),
        }
        return ShareRegularizerOutput(loss=loss, metrics=metrics)


__all__ = ["ShareSubspaceRegularizer", "ShareRegularizerOutput"]
