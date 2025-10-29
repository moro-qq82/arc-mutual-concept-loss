"""Command-line entry point for running ARC in-context evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from src.eval.ic_evaluator import EvaluationConfig, InContextEvaluator
from src.models.ic_model import ARCInContextModel
from src.train.data_module import ARCDataModule, ARCDataModuleConfig


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""

    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Configuration at {path} must be a mapping.")
    return payload


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


def main(argv: list[str] | None = None) -> None:
    """Execute evaluation based on the provided configuration."""

    parser = argparse.ArgumentParser(description="Run ARC in-context evaluation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval.yaml"),
        help="Path to the evaluation YAML configuration file.",
    )
    args = parser.parse_args(argv)

    config_payload = _load_yaml(args.config)
    if "data" not in config_payload:
        raise KeyError("Evaluation configuration must contain a 'data' section.")

    data_config = ARCDataModuleConfig.from_mapping(config_payload["data"])
    module = ARCDataModule(data_config)
    module.setup(None)

    eval_config = EvaluationConfig.from_mapping(config_payload.get("evaluation", {}))
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

    result = {
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

    if eval_config.save_predictions and evaluator.saved_predictions:
        result["predictions"] = evaluator.saved_predictions

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
