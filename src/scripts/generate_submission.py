"""Utility script for generating ARC-AGI submission files from trained checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

from src.models.ic_model import ARCInContextModel


def _load_yaml(path: Path) -> Mapping[str, object]:
    """Load a YAML configuration file into a mapping."""

    import yaml

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise TypeError("Configuration file must decode to a mapping.")
    return data


def _parse_model_kwargs(
    *,
    config_path: Optional[Path],
    model_kwargs_arg: Optional[str],
) -> Dict[str, object]:
    """Resolve model keyword arguments from CLI inputs."""

    payload: Dict[str, object] = {}
    if config_path is not None:
        config_data = _load_yaml(config_path)
        if "evaluation" in config_data and isinstance(config_data["evaluation"], Mapping):
            eval_section = config_data["evaluation"]
            model_section = eval_section.get("model")
        else:
            model_section = config_data.get("model")
        if isinstance(model_section, Mapping):
            payload.update(model_section)
    if model_kwargs_arg:
        try:
            extra = json.loads(model_kwargs_arg)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive branch
            raise ValueError("--model-kwargs must be valid JSON") from exc
        if not isinstance(extra, Mapping):
            raise TypeError("Parsed --model-kwargs JSON must describe a mapping.")
        payload.update(extra)
    return payload


def _ensure_grid(grid: object, *, key: str) -> List[List[int]]:
    """Validate that a payload entry contains a 2D integer grid."""

    if not isinstance(grid, Sequence):
        raise TypeError(f"Value under '{key}' must be a sequence of rows.")
    rows: List[List[int]] = []
    for row_index, row in enumerate(grid):
        if not isinstance(row, Sequence):
            raise TypeError(f"Row {row_index} in '{key}' must be a sequence of integers.")
        row_values: List[int] = []
        for col_index, value in enumerate(row):
            if not isinstance(value, int):
                raise TypeError(
                    f"Entry ({row_index}, {col_index}) in '{key}' must be an integer."
                )
            row_values.append(value)
        rows.append(row_values)
    return rows


def _grid_shape(grid: Sequence[Sequence[int]]) -> Tuple[int, int]:
    """Return ``(height, width)`` for the provided grid."""

    if not grid:
        return 0, 0
    return len(grid), len(grid[0]) if grid[0] else 0


def _pad_grid(
    grid: Sequence[Sequence[int]],
    height: int,
    width: int,
    *,
    fill_value: int,
) -> torch.Tensor:
    """Pad a grid to the requested spatial size."""

    tensor = torch.full((height, width), fill_value, dtype=torch.long)
    if not grid:
        return tensor
    original = torch.tensor(grid, dtype=torch.long)
    tensor[: original.shape[0], : original.shape[1]] = original
    return tensor


def _prepare_task_tensors(
    task: Mapping[str, object],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[Tuple[int, int]],
    List[Tuple[int, int]],
]:
    """Convert a raw ARC task into padded tensors suitable for inference."""

    train_examples = task.get("train")
    test_examples = task.get("test")
    if not isinstance(train_examples, Sequence) or not train_examples:
        raise ValueError("Each task must provide a non-empty 'train' sequence.")
    if not isinstance(test_examples, Sequence) or not test_examples:
        raise ValueError("Each task must provide a non-empty 'test' sequence.")

    support_inputs: List[List[List[int]]] = []
    support_outputs: List[List[List[int]]] = []
    query_inputs: List[List[List[int]]] = []
    support_output_shapes: List[Tuple[int, int]] = []
    query_input_shapes: List[Tuple[int, int]] = []

    for example in train_examples:
        if not isinstance(example, Mapping):
            raise TypeError("Entries in 'train' must be mappings.")
        input_grid = _ensure_grid(example.get("input"), key="train.input")
        output_grid = _ensure_grid(example.get("output"), key="train.output")
        support_inputs.append(input_grid)
        support_outputs.append(output_grid)
        support_output_shapes.append(_grid_shape(output_grid))

    for example in test_examples:
        if not isinstance(example, Mapping):
            raise TypeError("Entries in 'test' must be mappings.")
        input_grid = _ensure_grid(example.get("input"), key="test.input")
        query_inputs.append(input_grid)
        query_input_shapes.append(_grid_shape(input_grid))

    height_candidates = [shape[0] for shape in support_output_shapes + query_input_shapes]
    width_candidates = [shape[1] for shape in support_output_shapes + query_input_shapes]
    pad_height = max(height_candidates) if height_candidates else 0
    pad_width = max(width_candidates) if width_candidates else 0
    if pad_height == 0 or pad_width == 0:
        raise ValueError("Unable to determine padding dimensions for task grids.")

    support_inputs_tensor = torch.stack(
        [_pad_grid(grid, pad_height, pad_width, fill_value=0) for grid in support_inputs],
        dim=0,
    )
    support_outputs_tensor = torch.stack(
        [_pad_grid(grid, pad_height, pad_width, fill_value=0) for grid in support_outputs],
        dim=0,
    )
    query_inputs_tensor = torch.stack(
        [_pad_grid(grid, pad_height, pad_width, fill_value=0) for grid in query_inputs],
        dim=0,
    )

    return (
        support_inputs_tensor,
        support_outputs_tensor,
        query_inputs_tensor,
        torch.ones(len(train_examples), dtype=torch.bool),
        support_output_shapes,
        query_input_shapes,
    )


def _trim_grid(
    grid: Sequence[Sequence[int]],
    height: int,
    width: int,
) -> List[List[int]]:
    """Return the top-left ``height`` × ``width`` slice of ``grid``."""

    return [list(row[:width]) for row in grid[:height]]


def _select_majority_shape(shapes: Sequence[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """Return the most common ``(height, width)`` tuple from ``shapes``."""

    filtered = [shape for shape in shapes if shape[0] > 0 and shape[1] > 0]
    if not filtered:
        return None
    counter = Counter(filtered)
    return counter.most_common(1)[0][0]


def generate_submission(
    *,
    challenge_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    device_arg: Optional[str],
    model_kwargs: Mapping[str, object],
) -> None:
    """Run inference on ARC-AGI test challenges and emit a submission file."""

    with challenge_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if isinstance(payload, Mapping):
        task_items = list(payload.items())
    elif isinstance(payload, Sequence):  # pragma: no cover - compatibility branch
        task_items = []
        for entry in payload:
            if not isinstance(entry, Mapping):
                raise TypeError("Sequence entries must be mappings containing 'task_id'.")
            task_id = entry.get("task_id") or entry.get("id") or entry.get("name")
            if not task_id:
                raise KeyError("Task entry is missing an identifier.")
            task_items.append((str(task_id), entry))
    else:  # pragma: no cover - defensive branch
        raise TypeError("Challenges file must contain a mapping or a sequence of tasks.")

    device = torch.device(device_arg) if device_arg else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ARCInContextModel(**dict(model_kwargs))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state") if isinstance(checkpoint, Mapping) else None
    if state_dict is None:
        if isinstance(checkpoint, Mapping) and all(
            isinstance(value, torch.Tensor) for value in checkpoint.values()
        ):
            state_dict = checkpoint
        else:
            raise KeyError("Checkpoint does not contain a 'model_state' entry.")
    model.load_state_dict(state_dict)  # type: ignore[arg-type]
    model.to(device)
    model.eval()

    submission: MutableMapping[str, List[Mapping[str, List[List[int]]]]] = {}

    for task_id, task_payload in task_items:
        support_inputs, support_outputs, query_inputs, support_mask, support_shapes, query_shapes = (
            _prepare_task_tensors(task_payload)
        )

        support_inputs = support_inputs.unsqueeze(0).to(device)
        support_outputs = support_outputs.unsqueeze(0).to(device)
        query_inputs = query_inputs.unsqueeze(0).to(device)
        support_mask_tensor = support_mask.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model.predict(
                support_inputs=support_inputs,
                support_outputs=support_outputs,
                query_inputs=query_inputs,
                support_mask=support_mask_tensor,
            )
        predictions = logits.argmax(dim=2).cpu()[0]

        majority_shape = _select_majority_shape(support_shapes)
        task_attempts: List[Mapping[str, List[List[int]]]] = []
        for query_index, prediction in enumerate(predictions):
            full_grid = prediction.tolist()
            attempt_payload: MutableMapping[str, List[List[int]]] = {}

            query_shape = query_shapes[query_index]
            if query_shape[0] > 0 and query_shape[1] > 0:
                attempt_payload["attempt_1"] = _trim_grid(full_grid, *query_shape)
            else:
                attempt_payload["attempt_1"] = full_grid

            if majority_shape and majority_shape != query_shape:
                attempt_payload["attempt_2"] = _trim_grid(full_grid, *majority_shape)
            else:
                attempt_payload["attempt_2"] = attempt_payload["attempt_1"]

            task_attempts.append(attempt_payload)

        submission[str(task_id)] = task_attempts

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(submission, fh, ensure_ascii=False)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse CLI arguments and execute submission generation."""

    parser = argparse.ArgumentParser(description="Generate ARC-AGI submission JSON from a checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--challenge-path",
        type=Path,
        default=Path("data/raw/arc-agi_test_challenges.json"),
        help="Path to the ARC-AGI test challenges JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.json"),
        help="Destination path for the generated submission.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device (e.g., 'cuda', 'cuda:0', or 'cpu'). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML file providing model configuration under 'evaluation.model'.",
    )
    parser.add_argument(
        "--model-kwargs",
        type=str,
        default=None,
        help="Additional JSON-encoded keyword arguments for model construction.",
    )

    args = parser.parse_args(argv)
    model_kwargs = _parse_model_kwargs(config_path=args.config, model_kwargs_arg=args.model_kwargs)
    generate_submission(
        challenge_path=args.challenge_path,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device_arg=args.device,
        model_kwargs=model_kwargs,
    )


if __name__ == "__main__":
    main()
