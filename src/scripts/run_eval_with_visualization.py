"""Command-line tool for exporting ARC predictions and generating visual summaries."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from matplotlib import colors

from src.eval.ic_evaluator import EvaluationConfig, InContextEvaluator
from src.models.ic_model import ARCInContextModel
from src.train.data_module import ARCDataModule, ARCDataModuleConfig

ARC_COLOR_HEX: Sequence[str] = (
    "#000000",
    "#0074D9",
    "#FF4136",
    "#2ECC40",
    "#FFDC00",
    "#AAAAAA",
    "#F012BE",
    "#FF851B",
    "#7FDBFF",
    "#870C25",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Configuration at {path} must be a mapping.")
    return payload


def _stringify_paths(obj: Any) -> Any:
    """Recursively convert ``pathlib.Path`` objects to strings."""

    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: _stringify_paths(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_stringify_paths(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_stringify_paths(item) for item in obj)
    return obj


def _select_dataloader(module: ARCDataModule, split: str):
    """Return the dataloader for the requested split."""

    normalized = split.lower()
    if normalized in {"val", "validation"}:
        dataloader = module.val_dataloader()
        if dataloader is None:
            raise RuntimeError("Validation dataloader is not available; check split configuration.")
        return dataloader
    if normalized in {"test", "evaluation"}:
        dataloader = module.test_dataloader()
        if dataloader is None:
            raise RuntimeError("Test dataloader is not available; ensure meta_eval split exists.")
        return dataloader
    if normalized in {"train", "training"}:
        return module.train_dataloader()
    raise ValueError("Split must be one of 'train', 'val', or 'test'.")


def _resolve_figures_dir(path: Path | None) -> Path:
    """Return the directory for storing rendered figures."""

    if path is not None:
        return path
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return Path("reports/ic_eval/figures") / timestamp


def _prepare_colormap() -> tuple[colors.ListedColormap, colors.BoundaryNorm]:
    """Create the discrete colormap used for ARC grids."""

    cmap = colors.ListedColormap(ARC_COLOR_HEX, name="arc")
    bounds = np.arange(len(ARC_COLOR_HEX) + 1) - 0.5
    norm = colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def _sanitize_grid(grid: Sequence[Sequence[int]]) -> np.ndarray:
    """Convert nested lists into an integer ndarray safe for visualization."""

    if not grid or not grid[0]:
        return np.zeros((1, 1), dtype=np.int16)
    array = np.asarray(grid, dtype=np.int16)
    array[array < 0] = 0
    return array


def _load_support_examples(processed_dir: Path, task_id: str) -> list[Mapping[str, Any]]:
    """Load support examples for ``task_id`` from the processed dataset."""

    task_path = processed_dir / f"{task_id}.json"
    if not task_path.is_file():
        return []
    with task_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    examples = payload.get("k_shot_examples")
    if not isinstance(examples, Sequence):
        return []
    support_examples: list[Mapping[str, Any]] = []
    for example in examples:
        if not isinstance(example, Mapping):
            continue
        input_grid = example.get("input")
        output_grid = example.get("output")
        if isinstance(input_grid, Sequence) and isinstance(output_grid, Sequence):
            support_examples.append({"input": input_grid, "output": output_grid})
    return support_examples


def _render_task_summary(
    input_grid: Sequence[Sequence[int]],
    prediction: Sequence[Sequence[int]],
    target: Sequence[Sequence[int]],
    support_examples: Sequence[Mapping[str, Sequence[Sequence[int]]]],
    *,
    title: str,
    cmap: colors.ListedColormap,
    norm: colors.BoundaryNorm,
    dpi: int,
    output_path: Path,
) -> None:
    """Render a side-by-side comparison figure and save it to disk."""

    os.environ.setdefault("MPLBACKEND", "Agg")
    from matplotlib import pyplot as plt

    input_arr = _sanitize_grid(input_grid)
    pred_arr = _sanitize_grid(prediction)
    target_arr = _sanitize_grid(target)

    mismatch = pred_arr != target_arr
    accuracy = float((~mismatch).sum() / mismatch.size)
    is_perfect = bool(np.array_equal(pred_arr, target_arr))

    support_to_show = list(support_examples[:2])
    total_rows = len(support_to_show) + 1
    fig = plt.figure(figsize=(9, 3 * total_rows), constrained_layout=True)
    grid_spec = fig.add_gridspec(total_rows, 3)

    for row_index, example in enumerate(support_to_show):
        support_input = _sanitize_grid(example.get("input", []))
        support_output = _sanitize_grid(example.get("output", []))

        input_ax = fig.add_subplot(grid_spec[row_index, 0])
        input_ax.imshow(support_input, cmap=cmap, norm=norm, interpolation="nearest")
        input_ax.set_title(f"Support {row_index + 1} Input")
        input_ax.axis("off")

        output_ax = fig.add_subplot(grid_spec[row_index, 1])
        output_ax.imshow(support_output, cmap=cmap, norm=norm, interpolation="nearest")
        output_ax.set_title(f"Support {row_index + 1} Output")
        output_ax.axis("off")

        placeholder_ax = fig.add_subplot(grid_spec[row_index, 2])
        placeholder_ax.axis("off")

    query_row = len(support_to_show)
    query_axes = [
        fig.add_subplot(grid_spec[query_row, col_index])
        for col_index in range(3)
    ]
    panels = [
        (query_axes[0], input_arr, "Query Input"),
        (query_axes[1], pred_arr, "Model Prediction"),
        (query_axes[2], target_arr, "Ground Truth"),
    ]
    for ax, grid_arr, subtitle in panels:
        ax.imshow(grid_arr, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(subtitle)
        ax.axis("off")

    if mismatch.any():
        query_axes[1].imshow(
            np.ma.masked_where(~mismatch, mismatch),
            cmap="Reds",
            alpha=0.35,
            interpolation="nearest",
        )

    fig.suptitle(f"{title}\nMatch: {'✔' if is_perfect else '✘'}  Accuracy: {accuracy * 100:.1f}%")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _visualize_predictions(
    predictions: Iterable[Mapping[str, Any]],
    *,
    figures_dir: Path,
    processed_dir: Path,
    max_tasks: int | None,
    max_queries: int | None,
    dpi: int,
) -> list[Mapping[str, Any]]:
    """Render predictions to disk and return an index describing the artifacts."""

    cmap, norm = _prepare_colormap()
    rendered: list[Mapping[str, Any]] = []
    cached_support: dict[str, list[Mapping[str, Any]]] = {}
    for task_index, task_record in enumerate(predictions):
        if max_tasks is not None and task_index >= max_tasks:
            break
        task_id = task_record.get("task_id", f"task_{task_index:04d}")
        queries = task_record.get("queries", [])
        if not isinstance(queries, Sequence):
            continue
        task_id_str = str(task_id)
        if task_id_str not in cached_support:
            cached_support[task_id_str] = _load_support_examples(processed_dir, task_id_str)
        support_examples = cached_support[task_id_str]
        for query in list(queries)[: max_queries or None]:
            query_index = int(query.get("query_index", 0))
            input_grid = query.get("input", [])
            prediction_grid = query.get("prediction", [])
            target_grid = query.get("target", [])
            title = f"{task_id} / Query {query_index}"
            filename = f"{task_id}_query_{query_index:02d}.png"
            output_path = figures_dir / task_id / filename
            _render_task_summary(
                input_grid,
                prediction_grid,
                target_grid,
                support_examples,
                title=title,
                cmap=cmap,
                norm=norm,
                dpi=dpi,
                output_path=output_path,
            )
            rendered.append(
                {
                    "task_id": task_id,
                    "query_index": query_index,
                    "figure_path": str(output_path),
                }
            )
    return rendered


def main(argv: Sequence[str] | None = None) -> None:
    """Run evaluation, export predictions, and render comparison figures."""

    parser = argparse.ArgumentParser(description="Run ARC evaluation and visualize predictions.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval.yaml"),
        help="Path to the evaluation YAML configuration file.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Directory where raw prediction JSON files will be stored.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory where visualization images will be written.",
    )
    parser.add_argument(
        "--max-visualized-tasks",
        type=int,
        default=None,
        help="Optional limit on the number of tasks to visualize.",
    )
    parser.add_argument(
        "--max-queries-per-task",
        type=int,
        default=None,
        help="Optional limit on the number of queries rendered per task.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI when saving images.",
    )
    args = parser.parse_args(argv)

    config_payload = _load_yaml(args.config)
    if "data" not in config_payload:
        raise KeyError("Evaluation configuration must contain a 'data' section.")

    data_config = ARCDataModuleConfig.from_mapping(config_payload["data"])
    module = ARCDataModule(data_config)
    module.setup(None)

    evaluation_payload = dict(config_payload.get("evaluation", {}))
    evaluation_payload["save_predictions"] = True
    if args.predictions_dir is not None:
        evaluation_payload["predictions_dir"] = str(args.predictions_dir)

    eval_config = EvaluationConfig.from_mapping(evaluation_payload)
    dataloader = _select_dataloader(module, eval_config.split)

    model_kwargs = dict(eval_config.model_kwargs)
    model = ARCInContextModel(**model_kwargs)
    checkpoint = torch.load(eval_config.checkpoint_path, map_location="cpu")
    if "model_state" not in checkpoint:
        raise KeyError("Checkpoint is missing 'model_state'.")
    model.load_state_dict(checkpoint["model_state"])

    output_path = eval_config.resolve_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions_dir = eval_config.resolve_predictions_dir()

    evaluator = InContextEvaluator(
        model,
        dataloader,
        device=eval_config.device,
        ignore_index=data_config.ignore_index,
        save_predictions=True,
        predictions_dir=predictions_dir,
    )
    metrics = evaluator.evaluate(max_batches=eval_config.max_batches)

    result: dict[str, Any] = {
        "config": {
            "checkpoint_path": str(eval_config.checkpoint_path),
            "split": eval_config.split,
            "device": eval_config.device or "auto",
            "max_batches": eval_config.max_batches,
            "model_kwargs": model_kwargs,
            "data": {
                "processed_dir": str(data_config.processed_dir),
                "splits_dir": str(data_config.splits_dir),
            },
        },
        "metrics": metrics,
    }

    predictions = evaluator.saved_predictions
    if predictions:
        result["predictions"] = predictions

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    figures_dir = _resolve_figures_dir(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    index_records = _visualize_predictions(
        predictions,
        figures_dir=figures_dir,
        processed_dir=data_config.processed_dir,
        max_tasks=args.max_visualized_tasks,
        max_queries=args.max_queries_per_task,
        dpi=args.dpi,
    )

    index_payload: dict[str, Any] = {
        "metrics": metrics,
        "figures_dir": str(figures_dir),
        "rendered": index_records,
        "config": {
            "evaluation": _stringify_paths(asdict(eval_config)),
            "data": _stringify_paths(asdict(data_config)),
        },
    }
    index_path = figures_dir / "visualization_index.json"
    with index_path.open("w", encoding="utf-8") as fh:
        json.dump(index_payload, fh, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

