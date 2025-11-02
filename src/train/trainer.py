"""Training loop utilities for ARC models."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from .loss_factory import LossFactory, LossFactoryConfig


def _default_device(requested: Optional[str] = None) -> torch.device:
    """Resolve the training device."""

    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _amp_dtype(name: str) -> torch.dtype:
    """Map configuration strings to torch dtypes."""

    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported AMP dtype '{name}'. Use 'bf16' or 'fp16'.")


@dataclass
class OptimizerConfig:
    """Hyperparameters for the AdamW optimizer."""

    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = field(default_factory=lambda: (0.9, 0.95))


@dataclass
class SchedulerConfig:
    """Cosine scheduler with warmup configuration."""

    warmup_steps: int = 0
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.1


@dataclass
class TrainerConfig:
    """Configuration for the training loop."""

    epochs: int = 50
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    amp_dtype: str = "bf16"
    log_interval: int = 50
    eval_interval: int = 1
    checkpoint_dir: Path = Path("checkpoints")
    tensorboard_dir: Path = Path("logs/train/tensorboard")
    history_path: Path = Path("logs/train/history.jsonl")
    device: Optional[str] = None


class ARCTrainer:
    """Minimal trainer handling optimization, evaluation, and logging."""

    def __init__(
        self,
        model: nn.Module,
        loss_factory: LossFactory | LossFactoryConfig | None = None,
        *,
        optimizer_config: Optional[OptimizerConfig] = None,
        scheduler_config: Optional[SchedulerConfig] = None,
        trainer_config: Optional[TrainerConfig] = None,
    ) -> None:
        self.model = model
        self.loss_factory = loss_factory if isinstance(loss_factory, LossFactory) else LossFactory(loss_factory)
        self.optimizer_config = optimizer_config or OptimizerConfig()
        self.scheduler_config = scheduler_config or SchedulerConfig()
        self.config = trainer_config or TrainerConfig()
        self.device = _default_device(self.config.device)
        self.model.to(self.device)

    def _create_optimizer(self) -> AdamW:
        """Instantiate the AdamW optimizer."""

        return AdamW(
            self.model.parameters(),
            lr=self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay,
            betas=self.optimizer_config.betas,
        )

    def _create_scheduler(self, optimizer: AdamW, total_updates: int) -> LambdaLR:
        """Create a warmup + cosine decay scheduler."""

        warmup_steps = self.scheduler_config.warmup_steps
        if warmup_steps <= 0 and self.scheduler_config.warmup_ratio > 0:
            warmup_steps = int(total_updates * self.scheduler_config.warmup_ratio)
        warmup_steps = max(warmup_steps, 0)
        min_lr_ratio = self.scheduler_config.min_lr_ratio

        def lr_lambda(step: int) -> float:
            if total_updates == 0:
                return 1.0
            if step < warmup_steps and warmup_steps > 0:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = min(1.0, (step - warmup_steps) / max(1, total_updates - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return LambdaLR(optimizer, lr_lambda)

    def _move_to_device(self, batch: Mapping[str, object]) -> Dict[str, object]:
        """Move tensors in the batch to the configured device."""

        moved: Dict[str, object] = {}
        for key, value in batch.items():
            if isinstance(value, Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _forward_step(self, batch: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        """Run the model forward pass and compute losses."""

        outputs = self.model(
            support_inputs=batch["support_inputs"],
            support_outputs=batch["support_outputs"],
            query_inputs=batch["query_inputs"],
            support_mask=batch.get("support_mask"),
        )
        losses = self.loss_factory(
            logits=outputs.logits,
            targets=batch["query_outputs"],
            sae_reconstruction=outputs.sae_reconstruction,
            task_representation=outputs.task_representation,
            sae_latent=outputs.sae_latent,
            mask=batch.get("query_mask"),
        )
        losses["total_loss"] = losses["total_loss"].to(outputs.logits.dtype)
        return losses

    def _append_history(self, payload: Mapping[str, object]) -> None:
        """Persist metrics to the JSONL training history file."""

        self.config.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def fit(
        self,
        train_loader: Iterable[Mapping[str, object]],
        *,
        val_loader: Optional[Iterable[Mapping[str, object]]] = None,
    ) -> None:
        """Execute the training loop."""

        accumulation = max(1, self.config.gradient_accumulation_steps)
        optimizer = self._create_optimizer()
        total_batches = len(train_loader) if hasattr(train_loader, "__len__") else None
        if total_batches is None:
            raise ValueError("train_loader must define __len__ for scheduler setup.")
        total_updates = math.ceil(total_batches / accumulation) * self.config.epochs
        scheduler = self._create_scheduler(optimizer, total_updates)

        use_amp = self.config.use_amp and self.device.type == "cuda"
        amp_dtype = _amp_dtype(self.config.amp_dtype)
        scaler = GradScaler(self.device.type, enabled=use_amp)

        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(self.config.tensorboard_dir))

        global_step = 0
        best_val = float("inf")

        optimizer.zero_grad(set_to_none=True)

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            running: Dict[str, float] = defaultdict(float)
            running_steps = 0

            for batch_index, batch in enumerate(train_loader, start=1):
                batch_moved = self._move_to_device(batch)
                with autocast(self.device.type, dtype=amp_dtype, enabled=use_amp):
                    losses = self._forward_step(batch_moved)
                    loss = losses["total_loss"] / accumulation
                scaler.scale(loss).backward()

                if batch_index % accumulation == 0:
                    if self.config.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

                for key, value in losses.items():
                    running[key] += float(value.detach().cpu().item()) if isinstance(value, Tensor) else float(value)
                running_steps += 1

                if batch_index % self.config.log_interval == 0:
                    averages = {key: val / running_steps for key, val in running.items()}
                    for metric, metric_value in averages.items():
                        writer.add_scalar(f"train/{metric}", metric_value, global_step)
                    history_payload = {
                        "step": global_step,
                        "epoch": epoch,
                        "split": "train",
                        **{k: round(v, 6) for k, v in averages.items()},
                    }
                    self._append_history(history_payload)
                    running.clear()
                    running_steps = 0

            if total_batches % accumulation != 0:
                if self.config.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            if running_steps > 0:
                averages = {key: val / running_steps for key, val in running.items()}
                for metric, metric_value in averages.items():
                    writer.add_scalar(f"train/{metric}", metric_value, global_step)
                history_payload = {
                    "step": global_step,
                    "epoch": epoch,
                    "split": "train",
                    **{k: round(v, 6) for k, v in averages.items()},
                }
                self._append_history(history_payload)

            if val_loader is not None and epoch % self.config.eval_interval == 0:
                val_metrics = self.evaluate(val_loader)
                for metric, metric_value in val_metrics.items():
                    writer.add_scalar(f"val/{metric}", metric_value, global_step)
                history_payload = {
                    "step": global_step,
                    "epoch": epoch,
                    "split": "val",
                    **{k: round(v, 6) for k, v in val_metrics.items()},
                }
                self._append_history(history_payload)
                current_val = val_metrics.get("total_loss", float("nan"))
                if math.isnan(current_val):
                    current_val = float("inf")
                if current_val < best_val:
                    best_val = current_val
                    self._save_checkpoint(optimizer, scheduler, global_step, tag="best")

            self._save_checkpoint(optimizer, scheduler, global_step, tag=f"epoch_{epoch:03d}")

        writer.flush()
        writer.close()

    def evaluate(self, data_loader: Iterable[Mapping[str, object]]) -> Dict[str, float]:
        """Run evaluation over the provided dataloader."""

        self.model.eval()
        metrics: Dict[str, float] = defaultdict(float)
        count = 0
        with torch.no_grad():
            for batch in data_loader:
                batch_moved = self._move_to_device(batch)
                losses = self._forward_step(batch_moved)
                for key, value in losses.items():
                    metrics[key] += float(value.detach().cpu().item()) if isinstance(value, Tensor) else float(value)
                count += 1
        if count == 0:
            return {}
        return {key: value / count for key, value in metrics.items()}

    def _save_checkpoint(
        self,
        optimizer: AdamW,
        scheduler: LambdaLR,
        step: int,
        *,
        tag: str,
    ) -> None:
        """Persist model, optimizer, and scheduler states."""

        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "step": step,
        }
        path = self.config.checkpoint_dir / f"{tag}.pt"
        torch.save(checkpoint, path)


__all__ = ["ARCTrainer", "TrainerConfig", "OptimizerConfig", "SchedulerConfig"]
