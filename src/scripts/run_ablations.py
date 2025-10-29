"""Utility for executing the predefined ablation experiments."""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import types

try:  # Optional dependency; seeding is skipped when unavailable.
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - defensive fallback
    np = None  # type: ignore[assignment]
import torch
import yaml

from src.models.ic_model import ARCInContextModel
from src.train import (
    ARCDataModule,
    ARCDataModuleConfig,
    ARCTrainer,
    AdapterConfig,
    LossFactoryConfig,
    MetaAdaptationConfig,
    MetaAdapter,
    OptimizerConfig,
    SchedulerConfig,
    TaskAdaptationResult,
    TrainerConfig,
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""

    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"Configuration at {path} must decode into a mapping.")
    return dict(payload)


def _deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge ``updates`` into ``base`` without mutating inputs."""

    for key, value in updates.items():
        if isinstance(value, Mapping):
            current = base.get(key)
            if isinstance(current, Mapping):
                base[key] = _deep_update(dict(current), value)
            else:
                base[key] = _deep_update({}, value)
        else:
            base[key] = value
    return base


def _merge_payload(base_payload: Mapping[str, Any], *overrides: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Create a merged dictionary applying the provided overrides in order."""

    result: Dict[str, Any] = copy.deepcopy(dict(base_payload))
    for patch in overrides:
        if not patch:
            continue
        result = _deep_update(result, patch)  # type: ignore[assignment]
    return result


def _convert_bool(value: Any) -> bool:
    """Convert arbitrary inputs into booleans."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered in {"1", "true", "yes", "on"}
    return bool(value)


def _convert_value(field_type: Any, value: Any) -> Any:
    """Convert ``value`` to match the target ``field_type``."""

    from typing import get_args, get_origin  # Local import to avoid global dependency.

    origin = get_origin(field_type)
    if origin is None:
        if field_type is float:
            return float(value)
        if field_type is int:
            return int(value)
        if field_type is bool:
            return _convert_bool(value)
        if field_type is Path:
            return Path(str(value))
        return value

    union_types = {Union}
    union_type_attr = getattr(types, "UnionType", None)
    if union_type_attr is not None:
        union_types.add(union_type_attr)
    if origin in union_types:
        args = [arg for arg in get_args(field_type) if arg is not type(None)]
        if value is None:
            return None
        if not args:
            return value
        return _convert_value(args[0], value)

    if origin in {list, List, Sequence, Iterable}:
        args = get_args(field_type)
        if not isinstance(value, (list, tuple)):
            raise TypeError("Expected a list or tuple value.")
        if not args:
            return list(value)
        return [
            _convert_value(args[0], item) for item in value
        ]

    if origin in {tuple, Tuple}:
        args = get_args(field_type)
        if not isinstance(value, (list, tuple)):
            raise TypeError("Expected a list or tuple value.")
        if not args:
            return tuple(value)
        converted = [
            _convert_value(args[min(index, len(args) - 1)], item)
            for index, item in enumerate(value)
        ]
        return tuple(converted)

    return value


def _build_dataclass(cls: Any, mapping: Optional[Mapping[str, Any]]) -> Any:
    """Instantiate ``cls`` and populate fields using ``mapping``."""

    instance = cls()
    if not mapping:
        return instance
    for field in fields(instance):
        if field.name not in mapping:
            continue
        value = mapping[field.name]
        setattr(instance, field.name, _convert_value(field.type, value))
    return instance


def _dataclass_to_serialisable(obj: Any) -> Dict[str, Any]:
    """Convert dataclass-like objects into JSON serialisable dictionaries."""

    if not is_dataclass(obj):
        raise TypeError("Expected a dataclass instance.")
    payload: Dict[str, Any] = {}
    for field in fields(obj):
        value = getattr(obj, field.name)
        if isinstance(value, Path):
            payload[field.name] = str(value)
        elif isinstance(value, tuple):
            payload[field.name] = list(value)
        else:
            payload[field.name] = value
    return payload


def _set_seed(seed: Optional[int]) -> None:
    """Initialise RNG state across Python, NumPy, and PyTorch."""

    if seed is None:
        return
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_stage_payload(
    stage_spec: Mapping[str, Any],
    variant_overrides: Optional[Mapping[str, Any]],
) -> tuple[Dict[str, Any], Path]:
    """Load the base configuration and apply overrides for a stage."""

    base_path = Path(stage_spec.get("base_config", ""))
    if base_path:
        base_payload = _load_yaml(base_path)
    else:
        base_payload = {}
    stage_overrides = stage_spec.get("overrides") or {}
    payload = _merge_payload(base_payload, stage_overrides, variant_overrides)
    return payload, base_path


def _ensure_parent(path: Path) -> None:
    """Create the parent directory for ``path`` if it does not exist."""

    path.parent.mkdir(parents=True, exist_ok=True)


def _run_training_stage(
    payload: Mapping[str, Any],
    *,
    ablation: str,
    variant: str,
    description: str,
    base_config_path: Path,
) -> Dict[str, Any]:
    """Execute the training stage for a given ablation variant."""

    seed = payload.get("seed")
    _set_seed(int(seed) if seed is not None else None)

    if "data" not in payload:
        raise KeyError("Training configuration must include a 'data' section.")
    data_config = ARCDataModuleConfig.from_mapping(payload["data"])
    module = ARCDataModule(data_config)
    module.setup("fit")
    train_loader = module.train_dataloader()
    val_loader = module.val_dataloader()

    loss_config = _build_dataclass(LossFactoryConfig, payload.get("loss"))
    loss_config.task_ignore_index = data_config.ignore_index
    optimizer_config = _build_dataclass(OptimizerConfig, payload.get("optimizer"))
    scheduler_config = _build_dataclass(SchedulerConfig, payload.get("scheduler"))
    trainer_config = _build_dataclass(TrainerConfig, payload.get("trainer"))

    trainer_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer_config.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    _ensure_parent(trainer_config.history_path)

    model_kwargs = payload.get("model", {})
    model = ARCInContextModel(**model_kwargs)

    trainer = ARCTrainer(
        model,
        loss_factory=loss_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        trainer_config=trainer_config,
    )
    trainer.fit(train_loader, val_loader=val_loader)

    metrics: Dict[str, Any] = {}
    if val_loader is not None:
        metrics["validation"] = trainer.evaluate(val_loader)

    output_section = payload.get("output", {})
    report_path = Path(output_section.get("report_path", f"reports/ablations/{ablation}_{variant}.json"))
    _ensure_parent(report_path)

    report = {
        "ablation": ablation,
        "variant": variant,
        "description": description,
        "stage": "train",
        "base_config": str(base_config_path) if base_config_path else None,
        "seed": seed,
        "data": {
            "processed_dir": str(data_config.processed_dir),
            "splits_dir": str(data_config.splits_dir),
            "batch_size": data_config.batch_size,
        },
        "model_kwargs": model_kwargs,
        "loss": _dataclass_to_serialisable(loss_config),
        "optimizer": _dataclass_to_serialisable(optimizer_config),
        "scheduler": _dataclass_to_serialisable(scheduler_config),
        "trainer": _dataclass_to_serialisable(trainer_config),
        "metrics": metrics,
    }

    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"[train] {ablation}/{variant} -> {report_path}")

    return {
        "report_path": str(report_path),
        "metrics": metrics,
        "checkpoint_dir": str(trainer_config.checkpoint_dir),
        "history_path": str(trainer_config.history_path),
    }


def _aggregate_metrics(results: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    """Compute mean values for matching metric keys."""

    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for entry in results:
        for key, value in entry.items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in totals if counts[key] > 0}


def _run_meta_adaptation_stage(
    payload: Mapping[str, Any],
    *,
    ablation: str,
    variant: str,
    description: str,
    base_config_path: Path,
    default_seed: Optional[int],
) -> Dict[str, Any]:
    """Execute the meta-adaptation stage for a given ablation variant."""

    seed = payload.get("seed", default_seed)
    _set_seed(int(seed) if seed is not None else None)

    if "data" not in payload:
        raise KeyError("Meta-adaptation configuration must include a 'data' section.")
    data_config = ARCDataModuleConfig.from_mapping(payload["data"])
    module = ARCDataModule(data_config)
    module.setup("test")
    dataloader = module.test_dataloader()
    if dataloader is None:
        raise RuntimeError("Test dataloader is not available; ensure meta_eval split exists.")
    if dataloader.batch_size != 1:
        raise ValueError("Meta adaptation requires batch_size=1 to operate per task.")

    model_section = payload.get("model", {})
    checkpoint_path = Path(model_section.get("checkpoint_path", ""))
    if not checkpoint_path:
        raise KeyError("Meta-adaptation stage requires 'model.checkpoint_path'.")
    model_kwargs = dict(model_section.get("kwargs", {}))
    model = ARCInContextModel(**model_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" not in checkpoint:
        raise KeyError("Checkpoint is missing 'model_state'.")
    model.load_state_dict(checkpoint["model_state"])

    loss_config = _build_dataclass(LossFactoryConfig, payload.get("loss"))
    loss_config.task_ignore_index = data_config.ignore_index
    adapter_config = AdapterConfig.from_mapping(payload.get("adapter", {}))
    meta_config = MetaAdaptationConfig.from_mapping(payload.get("meta_adaptation", {}))
    meta_config.ignore_index = data_config.ignore_index

    meta_adapter = MetaAdapter(
        model,
        adapter_config=adapter_config,
        adaptation_config=meta_config,
        loss_factory=loss_config,
    )

    results: List[Mapping[str, Any]] = []
    pre_collection: List[Mapping[str, float]] = []
    post_collection: List[Mapping[str, float]] = []

    for batch in dataloader:
        task_result: TaskAdaptationResult = meta_adapter.adapt_task(batch)  # type: ignore[assignment]
        results.append(task_result.to_serialisable())
        pre_collection.append(task_result.pre_adaptation)
        post_collection.append(task_result.post_adaptation)

    summary = {
        "num_tasks": len(results),
        "pre_adaptation": _aggregate_metrics(pre_collection),
        "post_adaptation": _aggregate_metrics(post_collection),
    }

    output_section = payload.get("output", {})
    report_path = Path(
        output_section.get("report_path", f"reports/meta_adaptation/{ablation}_{variant}.json")
    )
    _ensure_parent(report_path)

    report = {
        "ablation": ablation,
        "variant": variant,
        "description": description,
        "stage": "meta_adaptation",
        "base_config": str(base_config_path) if base_config_path else None,
        "seed": seed,
        "data": {
            "processed_dir": str(data_config.processed_dir),
            "splits_dir": str(data_config.splits_dir),
        },
        "model": {
            "checkpoint_path": str(checkpoint_path),
            "kwargs": model_kwargs,
        },
        "loss": _dataclass_to_serialisable(loss_config),
        "adapter": _dataclass_to_serialisable(adapter_config),
        "meta_adaptation": _dataclass_to_serialisable(meta_config),
        "summary": summary,
        "tasks": results,
    }

    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"[meta]  {ablation}/{variant} -> {report_path}")

    return {
        "report_path": str(report_path),
        "summary": summary,
    }


def _normalise_variant_description(
    base_description: Optional[str],
    variant_description: Optional[str],
) -> str:
    """Prefer variant-specific descriptions when available."""

    if variant_description:
        return variant_description.strip()
    if base_description:
        return base_description.strip()
    return ""


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse CLI arguments and execute the requested ablation suite."""

    parser = argparse.ArgumentParser(description="Run ARC ablation experiments.")
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("configs/ablations"),
        help="Directory containing ablation YAML files.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("reports/ablations/summary.json"),
        help="Path to save the aggregated summary JSON.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional list of ablation names to execute.",
    )
    args = parser.parse_args(argv)

    ablation_files = sorted(args.configs_dir.glob("*.yaml"))
    if not ablation_files:
        raise FileNotFoundError(f"No YAML files found in {args.configs_dir}.")

    results: List[Dict[str, Any]] = []

    for config_path in ablation_files:
        config = _load_yaml(config_path)
        ablation_name = str(config.get("name", config_path.stem))
        if args.only and ablation_name not in args.only:
            continue
        base_description = config.get("description")
        label = config.get("label")

        variants = config.get("variants")
        if not variants:
            variants = [
                {
                    "id": ablation_name,
                    "description": base_description,
                    "overrides": {},
                }
            ]

        for entry in variants:
            variant_id = str(entry.get("id", f"{ablation_name}_{len(results)+1}"))
            variant_description = _normalise_variant_description(
                base_description, entry.get("description")
            )
            overrides = entry.get("overrides") or {}

            train_overrides: Optional[Mapping[str, Any]]
            if "train" in overrides:
                train_overrides = overrides.get("train")
            elif overrides:
                train_overrides = overrides
            else:
                train_overrides = None

            meta_overrides: Optional[Mapping[str, Any]] = None
            if "meta_adaptation" in overrides:
                meta_overrides = overrides.get("meta_adaptation")

            stage_results: Dict[str, Any] = {}
            base_seed: Optional[int] = None

            if "train" in config:
                train_payload, base_train_path = _prepare_stage_payload(
                    config["train"], train_overrides
                )
                stage_results["train"] = _run_training_stage(
                    train_payload,
                    ablation=ablation_name,
                    variant=variant_id,
                    description=variant_description,
                    base_config_path=base_train_path,
                )
                if "seed" in train_payload:
                    base_seed = int(train_payload["seed"])

            if "meta_adaptation" in config:
                meta_payload, base_meta_path = _prepare_stage_payload(
                    config["meta_adaptation"], meta_overrides
                )
                stage_results["meta_adaptation"] = _run_meta_adaptation_stage(
                    meta_payload,
                    ablation=ablation_name,
                    variant=variant_id,
                    description=variant_description,
                    base_config_path=base_meta_path,
                    default_seed=base_seed,
                )

            results.append(
                {
                    "ablation": ablation_name,
                    "variant": variant_id,
                    "label": label,
                    "description": variant_description,
                    "config_path": str(config_path),
                    "stages": stage_results,
                }
            )

    _ensure_parent(args.summary_path)
    with args.summary_path.open("w", encoding="utf-8") as fh:
        json.dump({"results": results}, fh, ensure_ascii=False, indent=2)

    print(f"Aggregated summary saved to {args.summary_path}")


if __name__ == "__main__":
    main()
