"""ARCアブレーション実験を評価データで実行するスクリプト。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from src.eval.ic_evaluator import EvaluationConfig, InContextEvaluator
from src.models.ic_model import ARCInContextModel
from src.scripts.run_ablations import (
    _ensure_parent,
    _load_yaml,
    _merge_payload,
    _normalise_variant_description,
    _prepare_stage_payload,
    _run_meta_adaptation_stage,
    _run_training_stage,
)
from src.train import ARCDataModule, ARCDataModuleConfig
from src.train.data_module import ARCProcessedTaskDataset
from src.train.data_module import _collate_tasks  # type: ignore


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


def _run_evaluation_stage(
    payload: Mapping[str, Any],
    *,
    ablation: str,
    variant: str,
    description: str,
    base_config_path: Path,
    checkpoint_override: Optional[Path],
    default_output_path: Path,
    default_predictions_dir: Path,
) -> Dict[str, Any]:
    """Execute evaluation for a single ablation variant."""

    if "data" not in payload:
        raise KeyError("Evaluation configuration must include a 'data' section.")
    if "evaluation" not in payload:
        raise KeyError("Evaluation configuration must include an 'evaluation' section.")

    data_section = payload["data"]
    evaluation_section = dict(payload["evaluation"])
    task_source = str(payload.get("task_source", "evaluation")).lower()

    if checkpoint_override is not None:
        evaluation_section["checkpoint_path"] = str(checkpoint_override)
    if "checkpoint_path" not in evaluation_section:
        raise KeyError("Evaluation requires a checkpoint path; provide one or enable training stage.")

    if "output_path" not in evaluation_section:
        evaluation_section["output_path"] = str(default_output_path)
    if evaluation_section.get("save_predictions"):
        predictions_dir = evaluation_section.get("predictions_dir")
        if predictions_dir is None:
            evaluation_section["predictions_dir"] = str(default_predictions_dir)

    data_config = ARCDataModuleConfig.from_mapping(data_section)
    eval_config = EvaluationConfig.from_mapping(evaluation_section)

    if task_source == "evaluation":
        all_paths = sorted(data_config.processed_dir.glob("*.json"))
        if not all_paths:
            raise RuntimeError(f"No processed tasks found in {data_config.processed_dir}.")
        task_ids = [path.stem for path in all_paths]
        dataset = ARCProcessedTaskDataset(
            data_config.processed_dir,
            task_ids,
            ignore_index=data_config.ignore_index,
        )
        dataloader: Iterable[Mapping[str, Any]] = DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            shuffle=False,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
            persistent_workers=data_config.persistent_workers and data_config.num_workers > 0,
            drop_last=False,
            collate_fn=lambda batch: _collate_tasks(batch, ignore_index=data_config.ignore_index),
        )
    else:
        module = ARCDataModule(data_config)
        module.setup(None)
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
        save_predictions=eval_config.save_predictions,
        predictions_dir=predictions_dir,
    )
    metrics = evaluator.evaluate(max_batches=eval_config.max_batches)

    report = {
        "ablation": ablation,
        "variant": variant,
        "description": description,
        "stage": "evaluation",
        "base_config": str(base_config_path) if base_config_path else None,
        "checkpoint_path": str(eval_config.checkpoint_path),
        "task_source": task_source,
        "data": {
            "processed_dir": str(data_config.processed_dir),
            "splits_dir": str(data_config.splits_dir),
            "batch_size": data_config.batch_size,
        },
        "evaluation": {
            "split": eval_config.split,
            "device": eval_config.device,
            "max_batches": eval_config.max_batches,
            "save_predictions": eval_config.save_predictions,
        },
        "model_kwargs": model_kwargs,
        "metrics": metrics,
    }

    output_path = Path(evaluation_section["output_path"])
    _ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"[eval] {ablation}/{variant} -> {output_path}")

    stage_result = {
        "report_path": str(output_path),
        "metrics": metrics,
    }
    if predictions_dir is not None:
        stage_result["predictions_dir"] = str(predictions_dir)
    return stage_result


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point for running ablations evaluated on the ARC evaluation set."""

    parser = argparse.ArgumentParser(description="Run ARC ablation experiments evaluated on the ARC evaluation set.")
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("configs/ablations"),
        help="Directory containing ablation YAML files.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("reports/ablations_eval/summary.json"),
        help="Path to save the aggregated summary JSON.",
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=Path("configs/eval.yaml"),
        help="Base evaluation configuration used when a config omits an evaluation stage.",
    )
    parser.add_argument(
        "--task-source",
        choices=["evaluation", "split"],
        default="evaluation",
        help="Task source when using the default evaluation configuration.",
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
            variant_description = _normalise_variant_description(base_description, entry.get("description"))
            overrides = entry.get("overrides") or {}

            train_overrides: Optional[Mapping[str, Any]]
            if "train" in overrides:
                train_overrides = overrides.get("train")
            elif overrides:
                train_overrides = overrides
            else:
                train_overrides = None

            meta_overrides: Optional[Mapping[str, Any]] = overrides.get("meta_adaptation") if "meta_adaptation" in overrides else None
            evaluation_overrides: Optional[Mapping[str, Any]] = overrides.get("evaluation") if "evaluation" in overrides else None

            stage_results: Dict[str, Any] = {}
            base_seed: Optional[int] = None
            checkpoint_override: Optional[Path] = None

            if "train" in config:
                train_payload, base_train_path = _prepare_stage_payload(config["train"], train_overrides)
                train_stage = _run_training_stage(
                    train_payload,
                    ablation=ablation_name,
                    variant=variant_id,
                    description=variant_description,
                    base_config_path=base_train_path,
                )
                stage_results["train"] = train_stage
                if "seed" in train_payload:
                    base_seed = int(train_payload["seed"])
                checkpoint_override = Path(train_stage["checkpoint_dir"]) / "best.pt"

            if "meta_adaptation" in config:
                meta_payload, base_meta_path = _prepare_stage_payload(config["meta_adaptation"], meta_overrides)
                stage_results["meta_adaptation"] = _run_meta_adaptation_stage(
                    meta_payload,
                    ablation=ablation_name,
                    variant=variant_id,
                    description=variant_description,
                    base_config_path=base_meta_path,
                    default_seed=base_seed,
                )

            if "evaluation" in config:
                evaluation_payload, base_eval_path = _prepare_stage_payload(config["evaluation"], evaluation_overrides)
            else:
                base_eval_path = args.eval_config
                base_payload = _load_yaml(args.eval_config)
                evaluation_payload = _merge_payload(base_payload, evaluation_overrides)
                evaluation_payload.setdefault("task_source", args.task_source)

            default_output = Path(f"reports/ablations_eval/{ablation_name}_{variant_id}.json")
            default_predictions_dir = Path(f"reports/ablations_eval/predictions/{ablation_name}_{variant_id}")

            stage_results["evaluation"] = _run_evaluation_stage(
                evaluation_payload,
                ablation=ablation_name,
                variant=variant_id,
                description=variant_description,
                base_config_path=base_eval_path,
                checkpoint_override=checkpoint_override,
                default_output_path=default_output,
                default_predictions_dir=default_predictions_dir,
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
