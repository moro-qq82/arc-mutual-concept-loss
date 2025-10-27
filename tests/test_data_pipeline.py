import json
from pathlib import Path

import pytest

from src.data.preprocess import DataPrepConfig, prepare_data_pipeline
from src.data.raw_loader import load_arc_tasks


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    data_root = tmp_path / "arc"
    training_dir = data_root / "training"
    evaluation_dir = data_root / "evaluation"
    training_dir.mkdir(parents=True)
    evaluation_dir.mkdir(parents=True)

    task_payload = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
            {"input": [[5, 6], [7, 8]], "output": [[5, 6], [7, 8]]},
            {"input": [[9, 0], [1, 2]], "output": [[9, 0], [1, 2]]},
        ],
        "test": [
            {"input": [[1, 1], [1, 1]], "output": [[2, 2], [2, 2]]}
        ],
        "metadata": {"difficulty": "easy"},
    }
    with (training_dir / "task_a.json").open("w", encoding="utf-8") as fh:
        json.dump(task_payload, fh)
    with (evaluation_dir / "task_b.json").open("w", encoding="utf-8") as fh:
        json.dump(task_payload, fh)
    return data_root


def test_load_arc_tasks_from_directory(sample_dataset: Path) -> None:
    tasks = load_arc_tasks(sample_dataset, "training")
    assert "task_a" in tasks
    task = tasks["task_a"]
    assert task.metadata["num_train_examples"] == 3
    assert task.metadata["num_test_examples"] == 1
    assert task.train[0].input == [[1, 2], [3, 4]]


def test_prepare_data_pipeline_outputs(tmp_path: Path, sample_dataset: Path) -> None:
    config = DataPrepConfig(
        raw_data_dir=sample_dataset,
        processed_dir=tmp_path / "processed",
        splits_dir=tmp_path / "splits",
        kshot_indices_dir=tmp_path / "splits" / "kshot",
        log_file=tmp_path / "logs" / "data.log",
        k_shot=2,
        val_fraction=0.5,
        seed=20250214,
        meta_eval_split="evaluation",
    )

    summary = prepare_data_pipeline(config)

    processed_file = config.processed_dir / "task_a.json"
    assert processed_file.exists()
    payload = json.loads(processed_file.read_text(encoding="utf-8"))
    assert payload["task_id"] == "task_a"
    assert len(payload["k_shot_examples"]) == 2

    kshot_file = config.kshot_indices_dir / "task_a.json"
    assert kshot_file.exists()
    indices = json.loads(kshot_file.read_text(encoding="utf-8"))
    assert indices["indices"] == sorted(indices["indices"])

    val_file = config.splits_dir / "val_tasks.json"
    assert val_file.exists()
    val_payload = json.loads(val_file.read_text(encoding="utf-8"))
    assert "task_ids" in val_payload
    assert len(val_payload["task_ids"]) >= 1

    meta_file = config.splits_dir / "meta_eval_test.json"
    assert meta_file.exists()
    meta_payload = json.loads(meta_file.read_text(encoding="utf-8"))
    assert "task_b" in meta_payload["task_ids"]

    log_text = config.log_file.read_text(encoding="utf-8")
    assert "Prepared" in log_text

    assert summary["train_tasks"] == ["task_a"]
