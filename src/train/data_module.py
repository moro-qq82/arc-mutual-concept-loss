"""Data loading utilities for ARC training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def _pad_grid(grid: Tensor, height: int, width: int, *, fill_value: int) -> Tensor:
    """Pad a ``(H, W)`` grid tensor to the provided spatial shape."""

    padded = torch.full((height, width), fill_value=fill_value, dtype=grid.dtype)
    padded[: grid.shape[0], : grid.shape[1]] = grid
    return padded


def _pad_stack(grids: Iterable[Tensor], height: int, width: int, *, fill_value: int) -> Tensor:
    """Pad and stack a sequence of grids into a tensor of shape ``(N, H, W)``."""

    padded = [_pad_grid(grid, height, width, fill_value=fill_value) for grid in grids]
    if not padded:
        return torch.empty(0, height, width, dtype=torch.long)
    return torch.stack(padded, dim=0)


def _load_json(path: Path) -> Mapping[str, object]:
    """Load JSON content from disk."""

    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _ensure_sequence(obj: object, *, key: str) -> Sequence[Mapping[str, object]]:
    """Validate that the provided object is a sequence of mappings."""

    if not isinstance(obj, Sequence):
        raise TypeError(f"Key '{key}' must contain a sequence.")
    sequence: List[Mapping[str, object]] = []
    for item in obj:
        if not isinstance(item, Mapping):
            raise TypeError(f"Entries under '{key}' must be mappings.")
        sequence.append(item)
    return sequence


def _grid_from_payload(payload: Mapping[str, object], *, key: str) -> Tensor:
    """Convert a nested list grid payload into a ``torch.LongTensor``."""

    if key not in payload:
        raise KeyError(f"Example payload is missing '{key}'.")
    grid = torch.tensor(payload[key], dtype=torch.long)
    if grid.dim() != 2:
        raise ValueError(f"Grid under '{key}' must be a rank-2 tensor.")
    return grid


@dataclass
class ARCDataModuleConfig:
    """Configuration for the ARC data module."""

    processed_dir: Path
    splits_dir: Path
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False
    drop_last: bool = False
    shuffle_train: bool = True
    ignore_index: int = -100

    @staticmethod
    def from_mapping(mapping: Mapping[str, object]) -> "ARCDataModuleConfig":
        """Instantiate configuration values from a generic mapping."""

        required_keys = {"processed_dir", "splits_dir"}
        missing = required_keys - mapping.keys()
        if missing:
            joined = ", ".join(sorted(missing))
            raise KeyError(f"Data configuration is missing required keys: {joined}")
        return ARCDataModuleConfig(
            processed_dir=Path(str(mapping["processed_dir"])),
            splits_dir=Path(str(mapping["splits_dir"])),
            batch_size=int(mapping.get("batch_size", 4)),
            num_workers=int(mapping.get("num_workers", 4)),
            pin_memory=bool(mapping.get("pin_memory", True)),
            persistent_workers=bool(mapping.get("persistent_workers", False)),
            drop_last=bool(mapping.get("drop_last", False)),
            shuffle_train=bool(mapping.get("shuffle_train", True)),
            ignore_index=int(mapping.get("ignore_index", -100)),
        )


class ARCProcessedTaskDataset(Dataset):
    """Dataset reading processed ARC tasks produced by the data pipeline."""

    def __init__(self, processed_dir: Path, task_ids: Sequence[str], *, ignore_index: int = -100) -> None:
        super().__init__()
        self.processed_dir = processed_dir
        self.ignore_index = ignore_index
        self._examples: List[MutableMapping[str, object]] = []
        for task_id in task_ids:
            payload_path = processed_dir / f"{task_id}.json"
            if not payload_path.is_file():
                raise FileNotFoundError(f"Processed file for task '{task_id}' not found at {payload_path}.")
            payload = _load_json(payload_path)
            if not isinstance(payload, Mapping):
                raise TypeError(f"Processed task '{task_id}' must decode to a mapping.")
            record: MutableMapping[str, object] = {"task_id": task_id}
            record["k_shot_examples"] = _ensure_sequence(payload.get("k_shot_examples", []), key="k_shot_examples")
            record["test_examples"] = _ensure_sequence(payload.get("test_examples", []), key="test_examples")
            record["metadata"] = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {}
            self._examples.append(record)

    def __len__(self) -> int:  # noqa: D401
        """Return the number of tasks."""

        return len(self._examples)

    def _tensorize_examples(
        self,
        examples: Sequence[Mapping[str, object]],
        *,
        pad_fill: int,
        target_pad: int,
        allow_missing_output: bool = False,
    ) -> Mapping[str, Tensor]:
        """Convert raw example payloads into padded tensors."""

        inputs: List[Tensor] = []
        outputs: List[Tensor] = []
        has_output: List[bool] = []
        max_height = 0
        max_width = 0
        for example in examples:
            input_grid = _grid_from_payload(example, key="input")
            output_value = example.get("output")
            if output_value is None:
                if not allow_missing_output:
                    raise KeyError("Example payload is missing 'output'.")
                output_grid = torch.full(
                    input_grid.shape,
                    fill_value=target_pad,
                    dtype=torch.long,
                )
                has_output.append(False)
            else:
                output_grid = _grid_from_payload(example, key="output")
                has_output.append(True)
            inputs.append(input_grid)
            outputs.append(output_grid)
            max_height = max(max_height, input_grid.shape[0], output_grid.shape[0])
            max_width = max(max_width, input_grid.shape[1], output_grid.shape[1])

        padded_inputs = _pad_stack(inputs, max_height, max_width, fill_value=pad_fill)
        padded_outputs = _pad_stack(outputs, max_height, max_width, fill_value=target_pad)
        masks = torch.zeros(len(examples), max_height, max_width, dtype=torch.bool)
        for index, output in enumerate(outputs):
            if has_output[index]:
                masks[index, : output.shape[0], : output.shape[1]] = True
        payload: Dict[str, Tensor] = {
            "inputs": padded_inputs,
            "outputs": padded_outputs,
            "mask": masks,
        }
        if allow_missing_output:
            payload["has_outputs"] = torch.tensor(has_output, dtype=torch.bool)
        return payload

    def __getitem__(self, index: int) -> Mapping[str, object]:  # noqa: D401
        """Load and tensorize a processed task."""

        record = self._examples[index]
        support = self._tensorize_examples(
            record["k_shot_examples"],
            pad_fill=0,
            target_pad=0,
        )
        query = self._tensorize_examples(
            record["test_examples"],
            pad_fill=0,
            target_pad=self.ignore_index,
            allow_missing_output=True,
        )

        support_mask = torch.ones(support["inputs"].shape[0], dtype=torch.bool)
        return {
            "task_id": record["task_id"],
            "support_inputs": support["inputs"],
            "support_outputs": support["outputs"],
            "support_mask": support_mask,
            "query_inputs": query["inputs"],
            "query_outputs": query["outputs"],
            "query_mask": query["mask"],
            "metadata": record["metadata"],
            "query_has_outputs": query.get("has_outputs"),
        }


def _collate_tasks(batch: Sequence[Mapping[str, object]], *, ignore_index: int) -> Mapping[str, object]:
    """Collate a batch of tasks, padding spatial dimensions as needed."""

    batch_size = len(batch)
    if batch_size == 0:
        raise ValueError("Cannot collate an empty batch.")

    max_support = max(sample["support_inputs"].shape[0] for sample in batch)
    max_queries = max(sample["query_inputs"].shape[0] for sample in batch)
    max_height = 0
    max_width = 0
    for sample in batch:
        tensors = [sample["support_inputs"], sample["support_outputs"], sample["query_inputs"], sample["query_outputs"]]
        for tensor in tensors:
            if tensor.numel() == 0:
                continue
            max_height = max(max_height, tensor.shape[-2])
            max_width = max(max_width, tensor.shape[-1])

    support_inputs = torch.zeros(batch_size, max_support, max_height, max_width, dtype=torch.long)
    support_outputs = torch.zeros(batch_size, max_support, max_height, max_width, dtype=torch.long)
    support_mask = torch.zeros(batch_size, max_support, dtype=torch.bool)
    query_inputs = torch.zeros(batch_size, max_queries, max_height, max_width, dtype=torch.long)
    query_outputs = torch.full(
        (batch_size, max_queries, max_height, max_width),
        fill_value=ignore_index,
        dtype=torch.long,
    )
    query_mask = torch.zeros(batch_size, max_queries, max_height, max_width, dtype=torch.bool)

    task_ids: List[str] = []
    metadata: List[Mapping[str, object]] = []

    for batch_index, sample in enumerate(batch):
        task_ids.append(sample["task_id"])
        metadata.append(sample.get("metadata", {}))

        support_inputs_tensor: Tensor = sample["support_inputs"]
        support_outputs_tensor: Tensor = sample["support_outputs"]
        support_count = support_inputs_tensor.shape[0]
        support_inputs[batch_index, :support_count, : support_inputs_tensor.shape[-2], : support_inputs_tensor.shape[-1]] = support_inputs_tensor
        support_outputs[batch_index, :support_count, : support_outputs_tensor.shape[-2], : support_outputs_tensor.shape[-1]] = support_outputs_tensor
        support_mask[batch_index, :support_count] = sample["support_mask"]

        query_inputs_tensor: Tensor = sample["query_inputs"]
        query_outputs_tensor: Tensor = sample["query_outputs"]
        query_count = query_inputs_tensor.shape[0]
        query_inputs[batch_index, :query_count, : query_inputs_tensor.shape[-2], : query_inputs_tensor.shape[-1]] = query_inputs_tensor
        query_outputs[batch_index, :query_count, : query_outputs_tensor.shape[-2], : query_outputs_tensor.shape[-1]] = query_outputs_tensor
        query_mask[batch_index, :query_count, : query_outputs_tensor.shape[-2], : query_outputs_tensor.shape[-1]] = sample["query_mask"]

    return {
        "task_ids": task_ids,
        "metadata": metadata,
        "support_inputs": support_inputs,
        "support_outputs": support_outputs,
        "support_mask": support_mask,
        "query_inputs": query_inputs,
        "query_outputs": query_outputs,
        "query_mask": query_mask,
    }


class ARCDataModule:
    """Simple PyTorch-Lightning inspired data module for ARC tasks."""

    def __init__(self, config: ARCDataModuleConfig) -> None:
        self.config = config
        self._train_dataset: Optional[ARCProcessedTaskDataset] = None
        self._val_dataset: Optional[ARCProcessedTaskDataset] = None
        self._test_dataset: Optional[ARCProcessedTaskDataset] = None

    def _load_split_ids(self, filename: str) -> List[str]:
        """Read a split definition file returning a list of task identifiers."""

        path = self.config.splits_dir / filename
        if not path.is_file():
            return []
        payload = _load_json(path)
        if not isinstance(payload, Mapping) or "task_ids" not in payload:
            raise ValueError(f"Split file '{filename}' must contain a mapping with key 'task_ids'.")
        task_ids = payload["task_ids"]
        if not isinstance(task_ids, Sequence):
            raise TypeError(f"Value under 'task_ids' in '{filename}' must be a sequence.")
        return [str(task_id) for task_id in task_ids]

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets for the requested stage."""

        processed_files = sorted(self.config.processed_dir.glob("*.json"))
        all_ids = [path.stem for path in processed_files]
        if not all_ids:
            raise RuntimeError(f"No processed tasks found in {self.config.processed_dir}.")

        val_ids = set(self._load_split_ids("val_tasks.json"))
        test_ids = set(self._load_split_ids("meta_eval_test.json"))

        if stage in (None, "fit"):
            train_ids = [task_id for task_id in all_ids if task_id not in val_ids]
            if not train_ids:
                raise RuntimeError("Training split is empty; verify data preparation outputs.")
            self._train_dataset = ARCProcessedTaskDataset(
                self.config.processed_dir,
                train_ids,
                ignore_index=self.config.ignore_index,
            )
            if val_ids:
                self._val_dataset = ARCProcessedTaskDataset(
                    self.config.processed_dir,
                    sorted(val_ids),
                    ignore_index=self.config.ignore_index,
                )

        if stage in (None, "validate") and self._val_dataset is None and val_ids:
            self._val_dataset = ARCProcessedTaskDataset(
                self.config.processed_dir,
                sorted(val_ids),
                ignore_index=self.config.ignore_index,
            )

        if stage in (None, "test") and test_ids:
            self._test_dataset = ARCProcessedTaskDataset(
                self.config.processed_dir,
                sorted(test_ids),
                ignore_index=self.config.ignore_index,
            )

    def train_dataloader(self) -> DataLoader[Mapping[str, object]]:
        """Return the training dataloader."""

        if self._train_dataset is None:
            raise RuntimeError("Data module has not been set up for training.")
        return DataLoader(
            self._train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_train,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers and self.config.num_workers > 0,
            drop_last=self.config.drop_last,
            collate_fn=lambda batch: _collate_tasks(batch, ignore_index=self.config.ignore_index),
        )

    def val_dataloader(self) -> Optional[DataLoader[Mapping[str, object]]]:
        """Return the validation dataloader if available."""

        if self._val_dataset is None:
            return None
        return DataLoader(
            self._val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers and self.config.num_workers > 0,
            drop_last=False,
            collate_fn=lambda batch: _collate_tasks(batch, ignore_index=self.config.ignore_index),
        )

    def test_dataloader(self) -> Optional[DataLoader[Mapping[str, object]]]:
        """Return the test dataloader if defined."""

        if self._test_dataset is None:
            return None
        return DataLoader(
            self._test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers and self.config.num_workers > 0,
            drop_last=False,
            collate_fn=lambda batch: _collate_tasks(batch, ignore_index=self.config.ignore_index),
        )


__all__ = ["ARCDataModule", "ARCDataModuleConfig", "ARCProcessedTaskDataset"]
