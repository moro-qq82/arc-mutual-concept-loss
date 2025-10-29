"""Preprocessing utilities for preparing ARC tasks for k-shot learning."""

from __future__ import annotations

import argparse
import json
import logging
import random
from hashlib import blake2b
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .raw_loader import ARCTask, GridExample, load_arc_tasks

LOGGER = logging.getLogger(__name__)


@dataclass
class DataPrepConfig:
    """Configuration values for the data preparation pipeline."""

    raw_data_dir: Path
    processed_dir: Path
    splits_dir: Path
    kshot_indices_dir: Path
    log_file: Path
    k_shot: int
    val_fraction: float = 0.1
    seed: int = 20250214
    meta_eval_split: str = "evaluation"

    @staticmethod
    def from_mapping(mapping: Mapping[str, object]) -> "DataPrepConfig":
        """Create a configuration instance from a generic mapping."""

        def _require(name: str) -> object:
            if name not in mapping:
                raise KeyError(f"Configuration is missing required key '{name}'.")
            return mapping[name]

        return DataPrepConfig(
            raw_data_dir=Path(str(_require("raw_data_dir"))),
            processed_dir=Path(str(_require("processed_dir"))),
            splits_dir=Path(str(_require("splits_dir"))),
            kshot_indices_dir=Path(str(_require("kshot_indices_dir"))),
            log_file=Path(str(_require("log_file"))),
            k_shot=int(_require("k_shot")),
            val_fraction=float(mapping.get("val_fraction", 0.1)),
            seed=int(mapping.get("seed", 20250214)),
            meta_eval_split=str(mapping.get("meta_eval_split", "evaluation")),
        )

    def ensure_directories(self) -> None:
        """Create all necessary output directories if they do not exist."""

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.kshot_indices_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)


def _select_k_indices(num_examples: int, k: int, seed: int) -> List[int]:
    """Select ``k`` indices deterministically using the provided seed."""

    if num_examples <= 0:
        raise ValueError("Tasks must contain at least one training example.")
    if num_examples <= k:
        return list(range(num_examples))
    indices = list(range(num_examples))
    rng = random.Random(seed)
    rng.shuffle(indices)
    chosen = sorted(indices[:k])
    return chosen


def _task_specific_seed(global_seed: int, task_id: str) -> int:
    """Create a deterministic per-task seed from the global seed and task identifier."""

    digest = blake2b(task_id.encode("utf-8"), digest_size=4, person=b"arc-kshot").digest()
    offset = int.from_bytes(digest, "big")
    return (global_seed + offset) % (2**32)


def _grid_example_to_dict(example: GridExample) -> Dict[str, List[List[int]]]:
    """Convert a :class:`GridExample` dataclass to a JSON-serializable dictionary."""

    return {"input": example.input, "output": example.output}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Persist a mapping to disk as JSON."""

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _append_log(log_file: Path, message: str) -> None:
    """Append a timestamped message to the data preparation log."""

    timestamp = datetime.utcnow().isoformat()
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(f"[{timestamp}] {message}\n")


def _create_validation_split(task_ids: Sequence[str], val_fraction: float) -> List[str]:
    """Create a deterministic validation split based on the provided fraction."""

    if not task_ids:
        return []
    count = max(1, int(len(task_ids) * val_fraction))
    return list(task_ids[:count])


def _summarize_task(task: ARCTask, k_indices: Sequence[int]) -> Mapping[str, object]:
    """Construct the processed representation for a single task."""

    k_examples = [_grid_example_to_dict(task.train[index]) for index in k_indices]
    test_examples = [_grid_example_to_dict(example) for example in task.test]
    return {
        "task_id": task.task_id,
        "k_shot_examples": k_examples,
        "test_examples": test_examples,
        "metadata": dict(task.metadata),
    }


def prepare_data_pipeline(config: DataPrepConfig) -> Dict[str, List[str]]:
    """Run the complete data preparation pipeline and return summary info."""

    config.ensure_directories()

    LOGGER.info("Loading training tasks from %s", config.raw_data_dir)
    train_tasks = load_arc_tasks(config.raw_data_dir, "training", seed=config.seed)

    LOGGER.info("Loading meta-evaluation tasks from split '%s'", config.meta_eval_split)
    meta_tasks: Dict[str, ARCTask] = {}
    try:
        meta_tasks = load_arc_tasks(config.raw_data_dir, config.meta_eval_split, seed=config.seed)
    except FileNotFoundError:
        LOGGER.warning("Meta-evaluation split '%s' not found; continuing without it.", config.meta_eval_split)

    for task in train_tasks.values():
        seed = _task_specific_seed(config.seed, task.task_id)
        k_indices = _select_k_indices(len(task.train), config.k_shot, seed)
        processed_payload = _summarize_task(task, k_indices)
        processed_path = config.processed_dir / f"{task.task_id}.json"
        _write_json(processed_path, processed_payload)
        kshot_path = config.kshot_indices_dir / f"{task.task_id}.json"
        _write_json(kshot_path, {"task_id": task.task_id, "indices": k_indices})

    task_ids = sorted(train_tasks.keys())
    validation_ids = _create_validation_split(task_ids, config.val_fraction)
    val_path = config.splits_dir / "val_tasks.json"
    _write_json(val_path, {"task_ids": validation_ids})

    if meta_tasks:
        meta_eval_path = config.splits_dir / "meta_eval_test.json"
        _write_json(meta_eval_path, {"task_ids": sorted(meta_tasks.keys())})

    message = (
        f"Prepared {len(train_tasks)} training tasks with k={config.k_shot}; "
        f"validation set size {len(validation_ids)}; meta-eval tasks {len(meta_tasks)}."
    )
    _append_log(config.log_file, message)
    LOGGER.info(message)

    return {
        "train_tasks": task_ids,
        "validation_tasks": validation_ids,
        "meta_eval_tasks": sorted(meta_tasks.keys()),
    }


def _load_config(path: Path) -> DataPrepConfig:
    """Load a :class:`DataPrepConfig` from a YAML file."""

    import yaml

    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, Mapping):
        raise TypeError("Configuration file must contain a mapping at the top level.")
    return DataPrepConfig.from_mapping(payload)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the data preparation script."""

    parser = argparse.ArgumentParser(description="Prepare ARC data for k-shot learning.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_prep.yaml"),
        help="Path to the data preparation YAML configuration file.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    config = _load_config(args.config)
    prepare_data_pipeline(config)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
