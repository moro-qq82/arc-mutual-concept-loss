"""Utilities for loading ARC-AGI-2 tasks from raw data files."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

TASK_LOAD_SEED = 20250214


@dataclass(frozen=True)
class GridExample:
    """Single ARC grid example consisting of an input and an output grid."""

    input: List[List[int]]
    output: List[List[int]]


@dataclass(frozen=True)
class ARCTask:
    """Container for ARC task examples and optional metadata."""

    task_id: str
    train: List[GridExample]
    test: List[GridExample]
    metadata: Mapping[str, Any]


def _normalize_example(example: Mapping[str, Any]) -> GridExample:
    """Validate and normalize a raw example into a :class:`GridExample`."""

    if "input" not in example or "output" not in example:
        raise ValueError("Each example must contain 'input' and 'output' keys.")
    input_grid = example["input"]
    output_grid = example["output"]
    if not isinstance(input_grid, list) or not all(isinstance(row, list) for row in input_grid):
        raise TypeError("Example input grid must be a 2D list of integers.")
    if not isinstance(output_grid, list) or not all(isinstance(row, list) for row in output_grid):
        raise TypeError("Example output grid must be a 2D list of integers.")
    return GridExample(input=input_grid, output=output_grid)


def _task_from_dict(task_id: str, payload: Mapping[str, Any], *, source_path: Optional[Path] = None) -> ARCTask:
    """Convert a dictionary payload into an :class:`ARCTask`."""

    if "train" not in payload or "test" not in payload:
        raise ValueError(f"Task '{task_id}' must include 'train' and 'test' keys.")
    train_examples = [_normalize_example(example) for example in payload["train"]]
    test_examples = [_normalize_example(example) for example in payload["test"]]
    metadata: MutableMapping[str, Any] = {}
    if "metadata" in payload and isinstance(payload["metadata"], Mapping):
        metadata.update(payload["metadata"])  # type: ignore[arg-type]
    if source_path is not None:
        metadata.setdefault("source_path", str(source_path))
    metadata.setdefault("num_train_examples", len(train_examples))
    metadata.setdefault("num_test_examples", len(test_examples))
    return ARCTask(task_id=task_id, train=train_examples, test=test_examples, metadata=dict(metadata))


def _load_directory_tasks(split_dir: Path) -> Dict[str, ARCTask]:
    """Load ARC tasks stored as individual JSON files in a directory."""

    tasks: Dict[str, ARCTask] = {}
    for json_path in sorted(split_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        task_id = json_path.stem
        tasks[task_id] = _task_from_dict(task_id, payload, source_path=json_path)
    return tasks


def _load_jsonl_tasks(jsonl_path: Path, split: str) -> Dict[str, ARCTask]:
    """Load ARC tasks stored in a JSON Lines file."""

    tasks: Dict[str, ARCTask] = {}
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise TypeError("Each JSONL line must decode to a mapping.")
            task_id = str(payload.get("task_id") or payload.get("id") or f"{split}_{index:05d}")
            tasks[task_id] = _task_from_dict(task_id, payload, source_path=jsonl_path)
    return tasks


def load_arc_tasks(raw_data_root: Path | str, split: str, *, seed: int = TASK_LOAD_SEED) -> Dict[str, ARCTask]:
    """Load ARC-AGI-2 tasks for a specific split.

    Parameters
    ----------
    raw_data_root:
        Path to the ARC data directory. It can either contain subdirectories per split
        (e.g. ``training/``) with JSON files, or JSONL files named ``<split>.jsonl``.
    split:
        Dataset split to load, typically ``"training"``, ``"evaluation"``, or ``"test"``.
    seed:
        Random seed used to ensure deterministic ordering when multiple files are present.

    Returns
    -------
    Dict[str, ARCTask]
        Mapping from task identifier to task contents.
    """

    root_path = Path(raw_data_root)
    if not root_path.exists():
        raise FileNotFoundError(f"Raw data root '{root_path}' does not exist.")

    split_dir = root_path / split
    if split_dir.is_dir():
        tasks = _load_directory_tasks(split_dir)
    else:
        jsonl_path = root_path / f"{split}.jsonl"
        if jsonl_path.is_file():
            tasks = _load_jsonl_tasks(jsonl_path, split)
        elif root_path.is_file() and root_path.suffix == ".jsonl":
            tasks = _load_jsonl_tasks(root_path, split)
        else:
            raise FileNotFoundError(
                f"Could not find data for split '{split}'. Checked '{split_dir}' and '{jsonl_path}'."
            )

    # Deterministic ordering with optional seeded shuffling for reproducibility.
    if tasks:
        ordered_ids = sorted(tasks.keys())
        rng = random.Random(seed)
        rng.shuffle(ordered_ids)
        return {task_id: tasks[task_id] for task_id in ordered_ids}
    return tasks


__all__ = ["ARCTask", "GridExample", "load_arc_tasks"]
