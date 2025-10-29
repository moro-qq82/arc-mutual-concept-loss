"""Few-shot meta-adaptation utilities with LoRA and adapters."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .loss_factory import LossFactory, LossFactoryConfig


def _amp_dtype(name: str) -> torch.dtype:
    """Map configuration strings to the corresponding AMP dtype."""

    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported AMP dtype '{name}'. Use 'bf16' or 'fp16'.")


class LoRAInjectedLinear(nn.Module):
    """Wrapper injecting a LoRA branch into a linear layer."""

    def __init__(self, module: nn.Linear, *, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base = module
        self.rank = rank
        self.scaling = alpha / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.down = nn.Linear(module.in_features, rank, bias=False)
        self.up = nn.Linear(rank, module.out_features, bias=False)
        self.reset_parameters()
        self.freeze_base_parameters()

    def reset_parameters(self) -> None:
        """Initialise LoRA parameters following the standard recipe."""

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def freeze_base_parameters(self) -> None:
        """Freeze the underlying linear weights."""

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, inputs: Tensor) -> Tensor:  # noqa: D401
        """Apply the wrapped linear layer with the LoRA residual."""

        base_out = self.base(inputs)
        lora_update = self.up(self.dropout(self.down(inputs))) * self.scaling
        return base_out + lora_update


class AdapterInjectedLinear(nn.Module):
    """Wrapper inserting a bottleneck adapter on the output of a linear layer."""

    def __init__(self, module: nn.Linear, *, bottleneck_dim: int, dropout: float) -> None:
        super().__init__()
        if bottleneck_dim <= 0:
            raise ValueError("Adapter bottleneck dimension must be positive.")
        self.base = module
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.down = nn.Linear(module.out_features, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, module.out_features, bias=False)
        self.activation = nn.ReLU(inplace=True)
        self.reset_parameters()
        self.freeze_base_parameters()

    def reset_parameters(self) -> None:
        """Initialise adapter weights for stable fine-tuning."""

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def freeze_base_parameters(self) -> None:
        """Disable gradients for the wrapped linear layer."""

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, inputs: Tensor) -> Tensor:  # noqa: D401
        """Apply the wrapped layer and adapter residual."""

        base_out = self.base(inputs)
        adapted = self.up(self.dropout(self.activation(self.down(base_out))))
        return base_out + adapted


@dataclass
class AdapterConfig:
    """Configuration describing how to insert adaptation modules."""

    mode: str = "lora"
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    bottleneck_dim: int = 64
    target_modules: Optional[Sequence[str]] = None

    @staticmethod
    def from_mapping(mapping: Mapping[str, object]) -> "AdapterConfig":
        """Create an adapter configuration from a generic mapping."""

        mode = str(mapping.get("mode", "lora")).lower()
        rank = int(mapping.get("rank", 8))
        alpha = float(mapping.get("alpha", 16.0))
        dropout = float(mapping.get("dropout", 0.05))
        bottleneck_dim = int(mapping.get("bottleneck_dim", 64))
        target_modules = mapping.get("target_modules")
        targets: Optional[List[str]] = None
        if target_modules is not None:
            if not isinstance(target_modules, Sequence) or isinstance(target_modules, (str, bytes)):
                raise TypeError("'target_modules' must be a sequence of module name substrings.")
            targets = [str(item) for item in target_modules]
        return AdapterConfig(
            mode=mode,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            target_modules=targets,
        )


@dataclass
class MetaAdaptationConfig:
    """Hyperparameters governing the inner-loop fine-tuning."""

    max_steps: int = 200
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    warmup_steps: int = 10
    min_lr_ratio: float = 0.1
    gradient_clip: float = 1.0
    log_interval: int = 25
    eval_interval: int = 50
    device: Optional[str] = None
    use_amp: bool = True
    amp_dtype: str = "bf16"
    ignore_index: int = -100
    history_dir: Optional[Path] = Path("logs/meta_adaptation")

    @staticmethod
    def from_mapping(mapping: Mapping[str, object]) -> "MetaAdaptationConfig":
        """Build a configuration object from a mapping."""

        history_dir = mapping.get("history_dir")
        history_path: Optional[Path]
        if history_dir is None:
            history_path = Path("logs/meta_adaptation")
        elif str(history_dir):
            history_path = Path(str(history_dir))
        else:
            history_path = None
        return MetaAdaptationConfig(
            max_steps=int(mapping.get("max_steps", 200)),
            learning_rate=float(mapping.get("learning_rate", 5e-4)),
            weight_decay=float(mapping.get("weight_decay", 0.0)),
            warmup_steps=int(mapping.get("warmup_steps", 10)),
            min_lr_ratio=float(mapping.get("min_lr_ratio", 0.1)),
            gradient_clip=float(mapping.get("gradient_clip", 1.0)),
            log_interval=int(mapping.get("log_interval", 25)),
            eval_interval=int(mapping.get("eval_interval", 50)),
            device=str(mapping.get("device")) if mapping.get("device") is not None else None,
            use_amp=bool(mapping.get("use_amp", True)),
            amp_dtype=str(mapping.get("amp_dtype", "bf16")),
            ignore_index=int(mapping.get("ignore_index", -100)),
            history_dir=history_path,
        )


@dataclass
class AdaptationLog:
    """Single log entry for a meta-adaptation step."""

    step: int
    loss: float
    learning_rate: float

    def to_mapping(self) -> Mapping[str, float]:
        """Return a serialisable mapping."""

        return {"step": self.step, "loss": self.loss, "learning_rate": self.learning_rate}


@dataclass
class TaskAdaptationResult:
    """Summary of pre/post metrics and adaptation statistics."""

    task_id: str
    pre_adaptation: Mapping[str, float]
    post_adaptation: Mapping[str, float]
    steps: int
    wall_time: float
    history: Sequence[AdaptationLog]

    def to_serialisable(self) -> Mapping[str, object]:
        """Convert the result into JSON serialisable primitives."""

        return {
            "task_id": self.task_id,
            "pre_adaptation": dict(self.pre_adaptation),
            "post_adaptation": dict(self.post_adaptation),
            "steps": self.steps,
            "wall_time": self.wall_time,
            "history": [entry.to_mapping() for entry in self.history],
        }


def _should_target(module_path: str, targets: Optional[Sequence[str]]) -> bool:
    """Return ``True`` when the module path matches the configured filters."""

    if not targets:
        return True
    return any(token in module_path for token in targets)


def _replace_linear_modules(
    module: nn.Module,
    *,
    adapter_config: AdapterConfig,
    prefix: str = "",
) -> List[nn.Module]:
    """Recursively wrap linear submodules with the requested adapters."""

    injected: List[nn.Module] = []
    for name, child in list(module.named_children()):
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and _should_target(path, adapter_config.target_modules):
            if adapter_config.mode == "lora":
                wrapped = LoRAInjectedLinear(
                    child,
                    rank=adapter_config.rank,
                    alpha=adapter_config.alpha,
                    dropout=adapter_config.dropout,
                )
            elif adapter_config.mode == "adapter":
                wrapped = AdapterInjectedLinear(
                    child,
                    bottleneck_dim=adapter_config.bottleneck_dim,
                    dropout=adapter_config.dropout,
                )
            else:
                raise ValueError("adapter_config.mode must be either 'lora' or 'adapter'.")
            setattr(module, name, wrapped)
            injected.append(wrapped)
        else:
            injected.extend(
                _replace_linear_modules(child, adapter_config=adapter_config, prefix=path)
            )
    return injected


def _gather_trainable_parameters(modules: Iterable[nn.Module]) -> List[nn.Parameter]:
    """Collect parameters that require gradients from the provided modules."""

    params: List[nn.Parameter] = []
    for module in modules:
        params.extend([param for param in module.parameters() if param.requires_grad])
    return params


class MetaAdapter:
    """Run short inner-loop fine-tuning with lightweight adapters."""

    def __init__(
        self,
        model: nn.Module,
        *,
        adapter_config: Optional[AdapterConfig] = None,
        adaptation_config: Optional[MetaAdaptationConfig] = None,
        loss_factory: LossFactory | LossFactoryConfig | None = None,
    ) -> None:
        self.model = model
        self.adapter_config = adapter_config or AdapterConfig()
        self.config = adaptation_config or MetaAdaptationConfig()
        if isinstance(loss_factory, LossFactory):
            self.loss_factory = loss_factory
        else:
            self.loss_factory = LossFactory(loss_factory)
        injected = _replace_linear_modules(self.model, adapter_config=self.adapter_config)
        if not injected:
            raise RuntimeError("No linear modules were wrapped; adjust 'target_modules' or model architecture.")
        self._injected_modules: List[nn.Module] = injected
        self.trainable_parameters = _gather_trainable_parameters(injected)
        if not self.trainable_parameters:
            raise RuntimeError("No trainable parameters found after adapter injection.")
        self.device = self._resolve_device()
        self.model.to(self.device)
        if self.config.history_dir is not None:
            self.config.history_dir.mkdir(parents=True, exist_ok=True)
        self._reset_adapters()

    def _resolve_device(self) -> torch.device:
        """Determine which device to use for adaptation."""

        if self.config.device is not None:
            return torch.device(self.config.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _move_to_device(self, batch: Mapping[str, object]) -> Dict[str, object]:
        """Move tensor values inside the batch to the configured device."""

        result: Dict[str, object] = {}
        for key, value in batch.items():
            if isinstance(value, Tensor):
                result[key] = value.to(self.device)
            else:
                result[key] = value
        return result

    def _build_training_batch(self, batch: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        """Construct an inner-loop batch using the support examples as supervision."""

        return {
            "support_inputs": batch["support_inputs"],
            "support_outputs": batch["support_outputs"],
            "support_mask": batch.get("support_mask"),
            "query_inputs": batch["support_inputs"],
            "query_outputs": batch["support_outputs"],
            "query_mask": torch.ones_like(batch["support_outputs"], dtype=torch.bool),
        }

    def _evaluate(self, batch: Mapping[str, Tensor]) -> Mapping[str, float]:
        """Run the model in evaluation mode and compute scalar metrics."""

        self.model.eval()
        with torch.no_grad():
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
        metrics: Dict[str, float] = {}
        for key, tensor in losses.items():
            if tensor.dim() == 0:
                metrics[key] = float(tensor.detach().cpu().item())
        return metrics

    def _log_history(self, task_id: str, history: Sequence[AdaptationLog]) -> None:
        """Write step-level history to the configured directory."""

        if self.config.history_dir is None:
            return
        path = self.config.history_dir / f"{task_id}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for entry in history:
                fh.write(json.dumps(entry.to_mapping(), ensure_ascii=False) + "\n")

    def _reset_adapters(self) -> None:
        """Reset adapter parameters between tasks."""

        for module in self._injected_modules:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def adapt_task(self, batch: Mapping[str, object]) -> TaskAdaptationResult:
        """Run meta-adaptation for a single task batch."""

        if "task_ids" not in batch:
            raise KeyError("Batch is missing 'task_ids'; ensure the data module returns task identifiers.")
        task_ids = batch["task_ids"]
        if not isinstance(task_ids, Sequence) or not task_ids:
            raise ValueError("Batch must contain at least one task identifier.")
        if len(task_ids) != 1:
            raise ValueError("MetaAdapter expects batches of size 1 during adaptation.")
        task_id = str(task_ids[0])

        self._reset_adapters()

        tensor_batch = {key: value for key, value in batch.items() if isinstance(value, Tensor)}
        moved_batch = self._move_to_device(tensor_batch)
        eval_batch = {
            "support_inputs": moved_batch["support_inputs"],
            "support_outputs": moved_batch["support_outputs"],
            "support_mask": moved_batch.get("support_mask"),
            "query_inputs": moved_batch["query_inputs"],
            "query_outputs": moved_batch["query_outputs"],
            "query_mask": moved_batch.get("query_mask"),
        }

        pre_metrics = self._evaluate(eval_batch)

        training_batch = self._build_training_batch(moved_batch)
        optimizer = AdamW(
            self.trainable_parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = LambdaLR(
            optimizer,
            lambda step: self._lr_schedule(
                step=step,
                max_steps=self.config.max_steps,
                warmup_steps=self.config.warmup_steps,
                min_ratio=self.config.min_lr_ratio,
            ),
        )
        use_amp = self.config.use_amp and self.device.type == "cuda"
        amp_dtype = _amp_dtype(self.config.amp_dtype)
        scaler = GradScaler(enabled=use_amp)

        history: List[AdaptationLog] = []
        start_time = time.perf_counter()
        self.model.train()
        for step in range(1, self.config.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            with autocast(dtype=amp_dtype, enabled=use_amp):
                outputs = self.model(
                    support_inputs=training_batch["support_inputs"],
                    support_outputs=training_batch["support_outputs"],
                    query_inputs=training_batch["query_inputs"],
                    support_mask=training_batch.get("support_mask"),
                )
                losses = self.loss_factory(
                    logits=outputs.logits,
                    targets=training_batch["query_outputs"],
                    sae_reconstruction=outputs.sae_reconstruction,
                    task_representation=outputs.task_representation,
                    sae_latent=outputs.sae_latent,
                    mask=training_batch.get("query_mask"),
                )
                loss = losses["total_loss"]
            scaler.scale(loss).backward()
            if self.config.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.trainable_parameters, self.config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % self.config.log_interval == 0 or step == 1 or step == self.config.max_steps:
                history.append(
                    AdaptationLog(
                        step=step,
                        loss=float(loss.detach().cpu().item()),
                        learning_rate=float(optimizer.param_groups[0]["lr"]),
                    )
                )
        wall_time = time.perf_counter() - start_time

        post_metrics = self._evaluate(eval_batch)
        self._log_history(task_id, history)

        return TaskAdaptationResult(
            task_id=task_id,
            pre_adaptation=pre_metrics,
            post_adaptation=post_metrics,
            steps=self.config.max_steps,
            wall_time=wall_time,
            history=history,
        )

    @staticmethod
    def _lr_schedule(*, step: int, max_steps: int, warmup_steps: int, min_ratio: float) -> float:
        """Simple warmup + cosine decay schedule."""

        if max_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step <= warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = min(1.0, max(0, step - warmup_steps) / max(1, max_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine


__all__ = [
    "AdapterConfig",
    "MetaAdaptationConfig",
    "MetaAdapter",
    "TaskAdaptationResult",
]
