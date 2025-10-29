"""Utilities for loading ARC-AGI-2 tasks from raw data files."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence as ABCSequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

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


def _normalize_test_examples(
    task_id: str,
    challenge_examples: Iterable[Mapping[str, Any]],
    solutions: Optional[Sequence[List[List[int]]]],
) -> List[Mapping[str, Any]]:
    """Merge challenge examples with optional solution outputs."""

    normalized: List[Mapping[str, Any]] = []
    for index, example in enumerate(challenge_examples):
        if "output" in example and example["output"] is not None:
            normalized.append(example)
            continue
        if solutions is None:
            raise ValueError(
                f"Task '{task_id}' is missing test outputs and no solutions were provided."
            )
        if index >= len(solutions):
            raise ValueError(
                f"Solutions for task '{task_id}' do not cover test example index {index}."
            )
        merged = dict(example)
        merged["output"] = solutions[index]
        normalized.append(merged)
    return normalized


def _normalize_task_collection(
    raw_tasks: object,
    *,
    source: Path,
    kind: str,
) -> Dict[str, Mapping[str, Any]]:
    """Convert heterogeneous task containers into a task-id keyed mapping."""

    def _is_sequence(value: object) -> bool:
        return isinstance(value, ABCSequence) and not isinstance(value, (str, bytes, bytearray))

    tasks: Dict[str, Mapping[str, Any]] = {}
    if isinstance(raw_tasks, Mapping):
        for task_id, payload in raw_tasks.items():
            if not isinstance(payload, Mapping):
                raise TypeError(
                    f"Task '{task_id}' payload in {source} must be a mapping, "
                    f"found {type(payload).__name__}."
                )
            tasks[str(task_id)] = payload
        return tasks

    if _is_sequence(raw_tasks):
        for index, entry in enumerate(raw_tasks):
            if not isinstance(entry, Mapping):
                raise TypeError(
                    f"Entry {index} in {source} for {kind} tasks must be a mapping, "
                    f"found {type(entry).__name__}."
                )
            task_id = entry.get("task_id") or entry.get("id") or entry.get("name")
            if not task_id:
                raise KeyError(
                    f"Entry {index} in {source} for {kind} tasks is missing a task identifier."
                )
            payload: Mapping[str, Any]
            if "task" in entry and isinstance(entry["task"], Mapping):
                payload = entry["task"]  # type: ignore[assignment]
            else:
                payload = {k: v for k, v in entry.items() if k not in {"task_id", "id", "name"}}
            tasks[str(task_id)] = payload
        return tasks

    raise TypeError(
        f"Unsupported container type '{type(raw_tasks).__name__}' for ARC tasks in {source}."
    )


def _normalize_solution_collection(
    raw_solutions: object,
    *,
    source: Path,
) -> Dict[str, Sequence[List[List[int]]]]:
    """Normalize solution payloads into a mapping keyed by task id."""

    def _is_sequence(value: object) -> bool:
        return isinstance(value, ABCSequence) and not isinstance(value, (str, bytes, bytearray))

    solutions: Dict[str, Sequence[List[List[int]]]] = {}
    if isinstance(raw_solutions, Mapping):
        for task_id, outputs in raw_solutions.items():
            if not isinstance(outputs, Iterable):
                raise TypeError(
                    f"Solutions for task '{task_id}' in {source} must be iterable, "
                    f"found {type(outputs).__name__}."
                )
            solutions[str(task_id)] = outputs  # type: ignore[assignment]
        return solutions

    if _is_sequence(raw_solutions):
        for index, entry in enumerate(raw_solutions):
            if not isinstance(entry, Mapping):
                raise TypeError(
                    f"Entry {index} in {source} solutions must be a mapping, "
                    f"found {type(entry).__name__}."
                )
            task_id = entry.get("task_id") or entry.get("id") or entry.get("name")
            if not task_id:
                raise KeyError(
                    f"Entry {index} in {source} solutions is missing a task identifier."
                )
            outputs = (
                entry.get("outputs")
                or entry.get("solutions")
                or entry.get("test_outputs")
                or entry.get("test")
                or entry.get("output")
            )
            if outputs is None:
                raise KeyError(
                    f"Entry {index} in {source} solutions does not contain output data."
                )
            if not isinstance(outputs, Iterable):
                raise TypeError(
                    f"Solutions for task '{task_id}' in {source} must be iterable, "
                    f"found {type(outputs).__name__}."
                )
            solutions[str(task_id)] = outputs  # type: ignore[assignment]
        return solutions

    raise TypeError(
        f"Unsupported container type '{type(raw_solutions).__name__}' for ARC solutions in {source}."
    )


def _load_challenge_solution_tasks(
    challenge_path: Path,
    solution_path: Optional[Path],
    split: str,
) -> Dict[str, ARCTask]:
    """Load ARC tasks stored as challenge/solution JSON files."""

    with challenge_path.open("r", encoding="utf-8") as fh:
        challenge_payload = json.load(fh)
    challenge_tasks = _normalize_task_collection(challenge_payload, source=challenge_path, kind=split)

    solutions_data: Optional[Mapping[str, Sequence[List[List[int]]]]] = None
    if solution_path is not None and solution_path.is_file():
        with solution_path.open("r", encoding="utf-8") as fh:
            raw_solutions = json.load(fh)
        solutions_data = _normalize_solution_collection(raw_solutions, source=solution_path)
    elif split != "test":
        raise FileNotFoundError(
            f"Expected solutions for split '{split}' at '{solution_path}'."
        )

    tasks: Dict[str, ARCTask] = {}
    for task_id, payload in challenge_tasks.items():
        payload_dict: Dict[str, Any] = dict(payload)
        challenge_test = payload_dict.get("test", [])
        if not isinstance(challenge_test, list):
            raise TypeError(f"Task '{task_id}' test examples must be provided as a list.")
        solutions = solutions_data.get(task_id) if solutions_data is not None else None
        payload_dict["test"] = _normalize_test_examples(task_id, challenge_test, solutions)

        metadata = payload_dict.get("metadata")
        if isinstance(metadata, Mapping):
            payload_dict["metadata"] = dict(metadata)
        else:
            payload_dict["metadata"] = {}
        payload_dict["metadata"].setdefault("source_path", str(challenge_path))
        if solution_path is not None and solution_path.is_file():
            payload_dict["metadata"].setdefault("solution_source_path", str(solution_path))

        tasks[task_id] = _task_from_dict(task_id, payload_dict)
    return tasks


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
        challenge_path = root_path / f"arc-agi_{split}_challenges.json"
        solution_path = root_path / f"arc-agi_{split}_solutions.json"
        if jsonl_path.is_file():
            tasks = _load_jsonl_tasks(jsonl_path, split)
        elif challenge_path.is_file():
            tasks = _load_challenge_solution_tasks(challenge_path, solution_path, split)
        elif root_path.is_file() and root_path.suffix == ".jsonl":
            tasks = _load_jsonl_tasks(root_path, split)
        else:
            raise FileNotFoundError(
                "Could not find data for split "
                f"'{split}'. Checked '{split_dir}', '{jsonl_path}', and '{challenge_path}'."
            )

    # Deterministic ordering with optional seeded shuffling for reproducibility.
    if tasks:
        ordered_ids = sorted(tasks.keys())
        rng = random.Random(seed)
        rng.shuffle(ordered_ids)
        return {task_id: tasks[task_id] for task_id in ordered_ids}
    return tasks


__all__ = ["ARCTask", "GridExample", "load_arc_tasks"]
