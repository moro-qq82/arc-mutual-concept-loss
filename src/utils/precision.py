"""Precision-handling utilities for numerically sensitive operations."""

from __future__ import annotations

import torch


def promote_precision(tensor: torch.Tensor) -> torch.Tensor:
    """Cast low-precision CUDA tensors to float32 for stable linear algebra."""
    if tensor.is_cuda and tensor.dtype in {torch.bfloat16, torch.float16}:
        return tensor.to(torch.float32)
    return tensor


def restore_precision(reference: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Return `value` cast to `reference.dtype` when they differ."""
    if reference.dtype != value.dtype:
        return value.to(reference.dtype)
    return value
