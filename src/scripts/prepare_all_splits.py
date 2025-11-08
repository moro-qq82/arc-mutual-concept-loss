"""CLI utility for preparing processed ARC data across all splits."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from src.data.preprocess import prepare_split

LOGGER = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the multi-split preparation tool."""

    parser = argparse.ArgumentParser(description="Prepare ARC splits into processed JSON files.")
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to the directory containing ARC raw JSON files.",
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=Path("data/processed_training-k-shot"),
        help="Directory where processed training tasks will be stored.",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path("data/processed_evaluation"),
        help="Directory where processed evaluation tasks will be stored.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("data/processed_test"),
        help="Directory where processed test tasks will be stored.",
    )
    parser.add_argument(
        "--k-shot",
        type=int,
        default=3,
        help="Number of support examples to sample for each training task.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20250214,
        help="Random seed used for deterministic sampling.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("data/processed/preparation.log"),
        help="Optional log file to append preparation summaries to.",
    )
    return parser.parse_args(argv)


def _append_log(log_path: Path, message: str) -> None:
    """Append a human-readable message to the provided log file."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def main(argv: Sequence[str] | None = None) -> None:
    """Prepare processed datasets for training, evaluation, and test splits."""

    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    LOGGER.info("Preparing training split (k=%d) -> %s", args.k_shot, args.training_dir)
    training_ids = prepare_split(
        args.raw_data_dir,
        "training",
        args.training_dir,
        k_shot=args.k_shot,
        seed=args.seed,
    )

    evaluation_ids: list[str] = []
    try:
        LOGGER.info("Preparing evaluation split -> %s", args.evaluation_dir)
        evaluation_ids = prepare_split(
            args.raw_data_dir,
            "evaluation",
            args.evaluation_dir,
            seed=args.seed,
        )
    except FileNotFoundError:
        LOGGER.warning("Evaluation split not found; skipping processed_evaluation generation.")

    test_ids: list[str] = []
    try:
        LOGGER.info("Preparing test split -> %s", args.test_dir)
        test_ids = prepare_split(
            args.raw_data_dir,
            "test",
            args.test_dir,
            seed=args.seed,
            allow_incomplete_test=True,
        )
    except FileNotFoundError:
        LOGGER.warning("Test split not found; skipping processed_test generation.")

    summary = (
        f"training={len(training_ids)} tasks, "
        f"evaluation={len(evaluation_ids)} tasks, "
        f"test={len(test_ids)} tasks"
    )
    LOGGER.info(summary)
    _append_log(args.log, summary)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
