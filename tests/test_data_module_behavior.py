"""Tests for :mod:`src.train.data_module` split handling."""

from __future__ import annotations

import json
from pathlib import Path

from src.train.data_module import ARCDataModule, ARCDataModuleConfig


def _write_processed_task(directory: Path, task_id: str) -> None:
    """Create a minimal processed ARC task file for testing."""

    payload = {
        "k_shot_examples": [
            {"input": [[0, 0], [0, 0]], "output": [[1, 1], [1, 1]]},
        ],
        "test_examples": [
            {"input": [[2, 2], [2, 2]], "output": [[3, 3], [3, 3]]},
        ],
        "metadata": {"source": "unit_test"},
    }
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{task_id}.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def test_data_module_ignores_missing_validation_ids(tmp_path: Path) -> None:
    """Ensure evaluation data can be loaded without training split definitions."""

    processed_dir = tmp_path / "processed_evaluation"
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True)

    _write_processed_task(processed_dir, "eval_task_a")
    _write_processed_task(processed_dir, "eval_task_b")

    val_payload = {"task_ids": ["train_task_unused"]}
    test_payload = {"task_ids": ["eval_task_a", "eval_task_b"]}

    with (splits_dir / "val_tasks.json").open("w", encoding="utf-8") as fh:
        json.dump(val_payload, fh)
    with (splits_dir / "meta_eval_test.json").open("w", encoding="utf-8") as fh:
        json.dump(test_payload, fh)

    config = ARCDataModuleConfig(
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        shuffle_train=False,
    )

    module = ARCDataModule(config)
    module.setup(None)

    assert module.val_dataloader() is None

    test_loader = module.test_dataloader()
    assert test_loader is not None
    assert hasattr(test_loader, "dataset")
    assert len(test_loader.dataset) == 2  # type: ignore[arg-type]

    batch = next(iter(test_loader))
    assert batch["task_ids"] == ["eval_task_a"]
