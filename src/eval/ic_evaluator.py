"""In-context evaluation utilities for ARC models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import torch
from torch import Tensor, nn

from .metrics import MetricAccumulator


@dataclass
class EvaluationConfig:
    """Configuration container for the in-context evaluator."""

    checkpoint_path: Path
    output_path: Optional[Path] = None
    device: Optional[str] = None
    split: str = "test"
    max_batches: Optional[int] = None
    save_predictions: bool = False
    predictions_dir: Optional[Path] = None
    model_kwargs: Mapping[str, object] = field(default_factory=dict)

    @staticmethod
    def from_mapping(mapping: Mapping[str, object]) -> "EvaluationConfig":
        """Create a configuration instance from a generic mapping."""

        if "checkpoint_path" not in mapping:
            raise KeyError("Evaluation configuration requires 'checkpoint_path'.")
        checkpoint = Path(str(mapping["checkpoint_path"]))
        output_path = mapping.get("output_path")
        predictions_dir = mapping.get("predictions_dir")
        return EvaluationConfig(
            checkpoint_path=checkpoint,
            output_path=Path(str(output_path)) if output_path is not None else None,
            device=str(mapping.get("device")) if mapping.get("device") is not None else None,
            split=str(mapping.get("split", "test")),
            max_batches=int(mapping["max_batches"]) if "max_batches" in mapping and mapping["max_batches"] is not None else None,
            save_predictions=bool(mapping.get("save_predictions", False)),
            predictions_dir=Path(str(predictions_dir)) if predictions_dir is not None else None,
            model_kwargs=dict(mapping.get("model", {})) if isinstance(mapping.get("model"), Mapping) else {},
        )

    def resolve_output_path(self) -> Path:
        """Return the concrete output path for evaluation results."""

        if self.output_path is not None:
            return self.output_path
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return Path("reports/ic_eval") / f"eval_{timestamp}.json"

    def resolve_predictions_dir(self) -> Optional[Path]:
        """Return the directory for saving predictions if enabled."""

        if not self.save_predictions:
            return None
        if self.predictions_dir is not None:
            return self.predictions_dir
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return Path("reports/ic_eval") / "predictions" / timestamp


class InContextEvaluator:
    """Utility class for running in-context evaluation loops."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: Iterable[Mapping[str, object]],
        *,
        device: Optional[str] = None,
        ignore_index: int = -100,
        save_predictions: bool = False,
        predictions_dir: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ignore_index = ignore_index
        self.metric_accumulator = MetricAccumulator()
        self.save_predictions = save_predictions
        self.predictions_dir = predictions_dir
        self._saved_predictions: List[Mapping[str, object]] = []
        self.model.to(self.device)

    def _move_to_device(self, batch: Mapping[str, object]) -> Dict[str, object]:
        """Move tensor values in the batch to the configured device."""

        moved: Dict[str, object] = {}
        for key, value in batch.items():
            if isinstance(value, Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _extract_prediction_records(
        self,
        batch: Mapping[str, object],
        logits: Tensor,
    ) -> List[Mapping[str, object]]:
        """Convert logits into serializable prediction records."""

        predictions = logits.argmax(dim=2).cpu()
        targets = batch["query_outputs"].cpu()
        mask = batch.get("query_mask")
        mask_cpu = mask.cpu() if isinstance(mask, Tensor) else None
        records: List[Mapping[str, object]] = []
        task_ids = batch.get("task_ids")
        metadata_list = batch.get("metadata")

        batch_size = predictions.shape[0]
        num_queries = predictions.shape[1]
        for task_index in range(batch_size):
            task_id = str(task_ids[task_index]) if task_ids is not None else f"task_{task_index:04d}"
            task_metadata = metadata_list[task_index] if isinstance(metadata_list, list) else {}
            query_records: List[Mapping[str, object]] = []
            for query_index in range(num_queries):
                prediction = predictions[task_index, query_index]
                target = targets[task_index, query_index]
                if mask_cpu is not None:
                    query_mask = mask_cpu[task_index, query_index]
                    if not query_mask.any():
                        continue
                    valid_rows = query_mask.any(dim=1)
                    valid_cols = query_mask.any(dim=0)
                    height = int(valid_rows.sum().item())
                    width = int(valid_cols.sum().item())
                    prediction_grid = prediction[:height, :width].tolist()
                    target_grid = target[:height, :width].tolist()
                else:
                    prediction_grid = prediction.tolist()
                    target_grid = target.tolist()
                query_records.append(
                    {
                        "query_index": query_index,
                        "prediction": prediction_grid,
                        "target": target_grid,
                    }
                )
            if query_records:
                records.append(
                    {
                        "task_id": task_id,
                        "metadata": task_metadata,
                        "queries": query_records,
                    }
                )
        return records

    def evaluate(self, *, max_batches: Optional[int] = None) -> Dict[str, float]:
        """Run evaluation over the dataloader and return aggregated metrics."""

        self.model.eval()
        saved_predictions: List[Mapping[str, object]] = []
        with torch.no_grad():
            for batch_index, batch in enumerate(self.dataloader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                batch_moved = self._move_to_device(batch)
                outputs = self.model(
                    support_inputs=batch_moved["support_inputs"],
                    support_outputs=batch_moved["support_outputs"],
                    query_inputs=batch_moved["query_inputs"],
                    support_mask=batch_moved.get("support_mask"),
                )
                logits = outputs.logits
                self.metric_accumulator.update(
                    logits=logits,
                    targets=batch_moved["query_outputs"],
                    mask=batch_moved.get("query_mask"),
                    ignore_index=self.ignore_index,
                )
                if self.save_predictions:
                    saved_predictions.extend(self._extract_prediction_records(batch, logits))

        if self.save_predictions and saved_predictions:
            if self.predictions_dir is not None:
                self.predictions_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                predictions_path = self.predictions_dir / f"predictions_{timestamp}.json"
                with predictions_path.open("w", encoding="utf-8") as fh:
                    json.dump(saved_predictions, fh, ensure_ascii=False, indent=2)
            self._saved_predictions = saved_predictions

        return self.metric_accumulator.compute()

    @property
    def saved_predictions(self) -> List[Mapping[str, object]]:
        """Return the saved prediction records."""

        return self._saved_predictions


__all__ = ["EvaluationConfig", "InContextEvaluator"]
