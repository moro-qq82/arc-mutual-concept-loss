"""Command-line entry point for running few-shot meta adaptation."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import yaml

from src.models.ic_model import ARCInContextModel
from src.train.data_module import ARCDataModule, ARCDataModuleConfig
from src.train.loss_factory import LossFactoryConfig
from src.train.meta_adapter import AdapterConfig, MetaAdaptationConfig, MetaAdapter


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""

    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Configuration at {path} must be a mapping.")
    return payload


def _loss_factory_config_from_mapping(mapping: Dict[str, Any]) -> LossFactoryConfig:
    """Construct a :class:`LossFactoryConfig` from a mapping."""

    config = LossFactoryConfig()
    field_info = {field.name: field for field in fields(LossFactoryConfig)}
    for key, value in mapping.items():
        if key not in field_info:
            continue
        field = field_info[key]
        if field.type is float:
            setattr(config, key, float(value))
        elif field.type is int:
            setattr(config, key, int(value))
        elif field.type is bool:
            setattr(config, key, bool(value))
        else:
            setattr(config, key, value)
    return config


def _aggregate_metrics(results: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """Compute the mean of scalar metrics across tasks."""

    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for entry in results:
        for key, value in entry.items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in totals if counts[key] > 0}


def _dataclass_to_serialisable(obj: Any) -> Dict[str, Any]:
    """Convert dataclass-like objects into JSON-friendly dictionaries."""

    result: Dict[str, Any] = {}
    for key, value in getattr(obj, "__dict__", {}).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def main(argv: list[str] | None = None) -> None:
    """Execute meta adaptation based on the provided configuration."""

    parser = argparse.ArgumentParser(description="Run few-shot meta adaptation on ARC tasks.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/meta_adaptation.yaml"),
        help="Path to the meta-adaptation YAML configuration file.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional limit on the number of tasks to adapt.",
    )
    args = parser.parse_args(argv)

    config_payload = _load_yaml(args.config)
    if "data" not in config_payload:
        raise KeyError("Meta adaptation configuration must contain a 'data' section.")

    data_config = ARCDataModuleConfig.from_mapping(config_payload["data"])
    module = ARCDataModule(data_config)
    module.setup("test")
    dataloader = module.test_dataloader()
    if dataloader is None:
        raise RuntimeError("Test dataloader is not available; ensure meta_eval_test split exists.")
    if dataloader.batch_size != 1:
        raise ValueError("Meta adaptation requires batch_size=1 to operate per task.")

    model_section = config_payload.get("model", {})
    checkpoint_path = Path(model_section.get("checkpoint_path", ""))
    if not checkpoint_path:
        raise KeyError("Model checkpoint path must be specified under 'model.checkpoint_path'.")
    model_kwargs = dict(model_section.get("kwargs", {}))
    model = ARCInContextModel(**model_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" not in checkpoint:
        raise KeyError("Checkpoint is missing 'model_state'.")
    model.load_state_dict(checkpoint["model_state"])

    loss_config = _loss_factory_config_from_mapping(config_payload.get("loss", {}))
    loss_config.task_ignore_index = data_config.ignore_index

    adapter_config = AdapterConfig.from_mapping(config_payload.get("adapter", {}))
    meta_config = MetaAdaptationConfig.from_mapping(config_payload.get("meta_adaptation", {}))

    meta_adapter = MetaAdapter(
        model,
        adapter_config=adapter_config,
        adaptation_config=meta_config,
        loss_factory=loss_config,
    )

    max_tasks = args.max_tasks
    results = []
    pre_metrics_collection = []
    post_metrics_collection = []

    for index, batch in enumerate(dataloader):
        if max_tasks is not None and index >= max_tasks:
            break
        task_result = meta_adapter.adapt_task(batch)
        results.append(task_result.to_serialisable())
        pre_metrics_collection.append(task_result.pre_adaptation)
        post_metrics_collection.append(task_result.post_adaptation)

    summary = {
        "num_tasks": len(results),
        "pre_adaptation": _aggregate_metrics(pre_metrics_collection),
        "post_adaptation": _aggregate_metrics(post_metrics_collection),
    }

    output_section = config_payload.get("output", {})
    report_path = Path(output_section.get("report_path", "reports/meta_adaptation/results.json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": {
            "config_path": str(args.config),
            "checkpoint_path": str(checkpoint_path),
            "adapter": _dataclass_to_serialisable(adapter_config),
            "meta_adaptation": _dataclass_to_serialisable(meta_config),
            "loss": _dataclass_to_serialisable(loss_config),
            "data": {
                "processed_dir": str(data_config.processed_dir),
                "splits_dir": str(data_config.splits_dir),
            },
        },
        "summary": summary,
        "tasks": results,
    }

    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
