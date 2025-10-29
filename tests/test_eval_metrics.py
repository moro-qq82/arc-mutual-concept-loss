import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.eval.ic_evaluator import InContextEvaluator
from src.eval.metrics import MetricAccumulator, compute_topk_exact_match


class DummyModel(torch.nn.Module):
    def forward(self, *, support_inputs, support_outputs, query_inputs, support_mask=None):  # type: ignore[override]
        batch, queries, height, width = query_inputs.shape
        num_classes = 3
        logits = torch.zeros(batch, queries, num_classes, height, width, device=query_inputs.device)
        for b in range(batch):
            for q in range(queries):
                for i in range(height):
                    for j in range(width):
                        value = float((i + j) % num_classes)
                        logits[b, q, int(value), i, j] = 5.0
        return types.SimpleNamespace(logits=logits)


def test_compute_topk_exact_match_basic() -> None:
    logits = torch.full((1, 1, 3, 2, 2), -5.0)
    targets = torch.tensor([[[[0, 1], [1, 2]]]])
    for i in range(2):
        for j in range(2):
            cls = targets[0, 0, i, j]
            logits[0, 0, cls, i, j] = 5.0
    result_top1 = compute_topk_exact_match(logits, targets, k=1)
    result_top3 = compute_topk_exact_match(logits, targets, k=3)
    assert result_top1.bool().item() is True
    assert result_top3.bool().item() is True


def test_metric_accumulator_perfect_case() -> None:
    logits = torch.full((1, 1, 3, 2, 2), -5.0)
    targets = torch.tensor([[[[0, 1], [1, 2]]]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    for i in range(2):
        for j in range(2):
            cls = targets[0, 0, i, j]
            logits[0, 0, cls, i, j] = 10.0
    accumulator = MetricAccumulator()
    accumulator.update(logits=logits, targets=targets, mask=mask)
    metrics = accumulator.compute()
    assert metrics["task_top1"] == pytest.approx(1.0)
    assert metrics["task_top3"] == pytest.approx(1.0)
    assert metrics["pixel_accuracy"] == pytest.approx(1.0)
    assert metrics["mean_iou"] == pytest.approx(1.0)
    assert metrics["exact_match_rate"] == pytest.approx(1.0)


def test_metric_accumulator_top3_tolerant() -> None:
    logits = torch.zeros((1, 1, 3, 1, 1))
    targets = torch.tensor([[[[1]]]])
    logits[0, 0, 2, 0, 0] = 5.0
    logits[0, 0, 1, 0, 0] = 4.0
    accumulator = MetricAccumulator()
    accumulator.update(logits=logits, targets=targets)
    metrics = accumulator.compute()
    assert metrics["task_top1"] == pytest.approx(0.0)
    assert metrics["task_top3"] == pytest.approx(1.0)
    assert metrics["pixel_accuracy"] == pytest.approx(0.0)
    assert metrics["mean_iou"] == pytest.approx(0.0)


def test_incontext_evaluator_runs(tmp_path: Path) -> None:
    batch = {
        "task_ids": ["task_a"],
        "metadata": [{}],
        "support_inputs": torch.zeros(1, 1, 2, 2, dtype=torch.long),
        "support_outputs": torch.zeros(1, 1, 2, 2, dtype=torch.long),
        "support_mask": torch.ones(1, dtype=torch.bool),
        "query_inputs": torch.zeros(1, 1, 2, 2, dtype=torch.long),
        "query_outputs": torch.tensor([[[[0, 1], [1, 2]]]], dtype=torch.long),
        "query_mask": torch.ones(1, 1, 2, 2, dtype=torch.bool),
    }
    dataloader = [batch]
    predictions_dir = tmp_path / "predictions"
    evaluator = InContextEvaluator(
        DummyModel(),
        dataloader,
        device="cpu",
        ignore_index=-100,
        save_predictions=True,
        predictions_dir=predictions_dir,
    )
    metrics = evaluator.evaluate()
    assert metrics["task_top1"] == pytest.approx(1.0)
    assert evaluator.saved_predictions
    assert predictions_dir.exists()
