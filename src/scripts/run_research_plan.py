"""Command-line utility to execute the research experimentation plan."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

try:  # Optional dependency; provide helpful error message when absent.
    import yaml  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - executed when PyYAML is missing
    yaml = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


class _StrictFormatDict(dict):
    """Mapping that raises ``KeyError`` when a placeholder is missing."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - defensive branch
        raise KeyError(f"Undefined template variable: {key}")


@dataclass
class PlanTask:
    """Single actionable item inside a stage."""

    name: str
    kind: str
    command: list[str] = field(default_factory=list)
    module: Optional[str] = None
    args: list[str] = field(default_factory=list)
    cwd: Optional[Path] = None
    directory: Optional[Path] = None
    log_file: Optional[Path] = None
    message: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    continue_on_error: bool = False

    @staticmethod
    def from_mapping(payload: Mapping[str, object], base_dir: Path) -> "PlanTask":
        """Create a task from a configuration mapping."""

        name = str(payload.get("name", ""))
        if not name:
            raise ValueError("Each task must define a non-empty 'name'.")
        kind = str(payload.get("kind", payload.get("type", "")))
        if not kind:
            raise ValueError(f"Task '{name}' is missing a 'kind'.")

        command_list: list[str] = []
        if "command" in payload:
            raw_command = payload["command"]
            if isinstance(raw_command, (str, bytes)):
                raise TypeError(f"Task '{name}' command must not be a string.")
            if not isinstance(raw_command, Iterable):
                raise TypeError(f"Task '{name}' command must be a sequence of strings.")
            command_list = [str(item) for item in raw_command]  # type: ignore[arg-type]

        args_list: list[str] = []
        if "args" in payload:
            raw_args = payload["args"]
            if isinstance(raw_args, (str, bytes)):
                raise TypeError(f"Task '{name}' args must not be a string.")
            if not isinstance(raw_args, Iterable):
                raise TypeError(f"Task '{name}' args must be a sequence of strings.")
            args_list = [str(item) for item in raw_args]  # type: ignore[arg-type]

        cwd_path = None
        if "cwd" in payload and payload["cwd"] is not None:
            cwd_path = (base_dir / Path(str(payload["cwd"])) ).resolve()

        directory = None
        if "directory" in payload and payload["directory"] is not None:
            directory = (base_dir / Path(str(payload["directory"])) ).resolve()

        log_file = None
        if "log_file" in payload and payload["log_file"] is not None:
            log_file = (base_dir / Path(str(payload["log_file"])) ).resolve()

        env_payload = payload.get("env") or {}
        if not isinstance(env_payload, Mapping):
            raise TypeError(f"Task '{name}' env must be a mapping of key/value pairs.")
        env = {str(key): str(value) for key, value in env_payload.items()}

        return PlanTask(
            name=name,
            kind=kind.lower(),
            command=command_list,
            module=str(payload.get("module")) if payload.get("module") else None,
            args=args_list,
            cwd=cwd_path,
            directory=directory,
            log_file=log_file,
            message=str(payload.get("message")) if payload.get("message") else None,
            env=env,
            continue_on_error=bool(payload.get("continue_on_error", False)),
        )


@dataclass
class PlanStage:
    """Collection of ordered tasks representing one logical stage."""

    name: str
    description: Optional[str]
    tasks: list[PlanTask]

    @staticmethod
    def from_mapping(name: str, payload: Mapping[str, object], base_dir: Path) -> "PlanStage":
        """Create a stage from YAML mapping."""

        description = str(payload.get("description")) if payload.get("description") else None
        raw_tasks = payload.get("tasks")
        if isinstance(raw_tasks, (str, bytes)):
            raise TypeError(f"Stage '{name}' tasks must not be a string.")
        if not isinstance(raw_tasks, Iterable):
            raise TypeError(f"Stage '{name}' must contain a 'tasks' list.")
        tasks: list[PlanTask] = []
        for item in raw_tasks:  # type: ignore[assignment]
            if not isinstance(item, Mapping):
                raise TypeError(f"Stage '{name}' tasks must be mappings; got {type(item)!r}.")
            tasks.append(PlanTask.from_mapping(item, base_dir))
        if not tasks:
            raise ValueError(f"Stage '{name}' must define at least one task.")
        return PlanStage(name=name, description=description, tasks=tasks)


@dataclass
class PlanConfig:
    """Top-level configuration container for the plan runner."""

    stages: Dict[str, PlanStage]
    default_cwd: Path
    base_dir: Path

    @staticmethod
    def from_mapping(payload: Mapping[str, object], *, base_dir: Path) -> "PlanConfig":
        """Create configuration from mapping."""

        raw_default_cwd = payload.get("default_cwd")
        if raw_default_cwd:
            default_cwd = (base_dir / Path(str(raw_default_cwd))).resolve()
        else:
            default_cwd = base_dir

        stages_payload = payload.get("stages")
        if not isinstance(stages_payload, Mapping):
            raise TypeError("Configuration must include a 'stages' mapping.")
        stages: Dict[str, PlanStage] = {}
        for name, stage_payload in stages_payload.items():
            if not isinstance(stage_payload, Mapping):
                raise TypeError(f"Stage '{name}' definition must be a mapping.")
            stages[name] = PlanStage.from_mapping(name, stage_payload, base_dir)
        if not stages:
            raise ValueError("Plan configuration must define at least one stage.")
        return PlanConfig(stages=stages, default_cwd=default_cwd, base_dir=base_dir)


def _load_yaml(path: Path) -> MutableMapping[str, object]:
    """Load YAML content from path."""

    if yaml is None:
        raise ModuleNotFoundError(
            "PyYAML is required to load research plan configurations. Install it via 'pip install PyYAML'."
        )
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, MutableMapping):
        raise TypeError("Plan configuration must decode into a mapping.")
    return payload


class PlanRunner:
    """Execute stages defined by :class:`PlanConfig`."""

    def __init__(self, config: PlanConfig, *, variables: Optional[Mapping[str, str]] = None) -> None:
        self.config = config
        defaults = {
            "python": sys.executable,
            "project_root": str(config.default_cwd),
            "config_dir": str(config.base_dir),
        }
        if variables:
            defaults.update(variables)
        self.variables = defaults

    def _format(self, value: str) -> str:
        """Apply template expansion to a single string."""

        return value.format_map(_StrictFormatDict(self.variables))

    def _expand_sequence(self, values: Iterable[str]) -> list[str]:
        """Apply template expansion to command arguments."""

        return [self._format(value) for value in values]

    def _resolve_cwd(self, task: PlanTask) -> Path:
        """Determine the working directory for a task."""

        if task.cwd is not None:
            return task.cwd
        return self.config.default_cwd

    def _run_command(self, task: PlanTask, dry_run: bool) -> None:
        """Execute a shell command task."""

        if not task.command:
            raise ValueError(f"Task '{task.name}' of kind 'command' must specify 'command'.")
        command = self._expand_sequence(task.command)
        cwd = self._resolve_cwd(task)
        env = {key: self._format(value) for key, value in task.env.items()}
        LOGGER.info("[%s] Running command: %s", task.name, " ".join(command))
        if dry_run:
            return
        env_map = os.environ.copy()
        env_map.update(env)
        result = subprocess.run(command, cwd=cwd, env=env_map, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Command task '{task.name}' failed with exit code {result.returncode}.")

    def _run_module(self, task: PlanTask, dry_run: bool) -> None:
        """Execute a Python module using ``sys.executable``."""

        if not task.module:
            raise ValueError(f"Task '{task.name}' of kind 'module' must specify 'module'.")
        args = [sys.executable, "-m", task.module]
        args.extend(self._expand_sequence(task.args))
        cwd = self._resolve_cwd(task)
        env = {key: self._format(value) for key, value in task.env.items()}
        LOGGER.info("[%s] Running module: python -m %s", task.name, task.module)
        if dry_run:
            return
        env_map = os.environ.copy()
        env_map.update(env)
        result = subprocess.run(args, cwd=cwd, env=env_map, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Module task '{task.name}' failed with exit code {result.returncode}.")

    def _run_mkdir(self, task: PlanTask, dry_run: bool) -> None:
        """Create a directory recursively."""

        if not task.directory:
            raise ValueError(f"Task '{task.name}' of kind 'mkdir' must define 'directory'.")
        target = task.directory
        LOGGER.info("[%s] Ensuring directory exists: %s", task.name, target)
        if dry_run:
            return
        target.mkdir(parents=True, exist_ok=True)

    def _run_log(self, task: PlanTask, dry_run: bool) -> None:
        """Append a message to the specified log file."""

        if not task.log_file:
            raise ValueError(f"Task '{task.name}' of kind 'log' must define 'log_file'.")
        if task.message is None:
            raise ValueError(f"Task '{task.name}' of kind 'log' must define 'message'.")
        log_file = task.log_file
        message = self._format(task.message)
        LOGGER.info("[%s] Appending message to %s", task.name, log_file)
        if dry_run:
            return
        log_file.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().isoformat()
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(f"[{timestamp}] {message}\n")

    def run_stage(self, stage_name: str, *, dry_run: bool = False) -> None:
        """Execute all tasks from the given stage."""

        if stage_name not in self.config.stages:
            raise KeyError(f"Unknown stage '{stage_name}'. Available: {', '.join(self.config.stages)}")
        stage = self.config.stages[stage_name]
        LOGGER.info("=== Stage: %s ===", stage.name)
        if stage.description:
            LOGGER.info("%s", stage.description)
        for task in stage.tasks:
            try:
                if task.kind == "command":
                    self._run_command(task, dry_run)
                elif task.kind == "module":
                    self._run_module(task, dry_run)
                elif task.kind == "mkdir":
                    self._run_mkdir(task, dry_run)
                elif task.kind == "log":
                    self._run_log(task, dry_run)
                else:
                    raise ValueError(f"Unsupported task kind '{task.kind}' for task '{task.name}'.")
            except Exception as exc:
                LOGGER.error("Task '%s' failed: %s", task.name, exc)
                if not task.continue_on_error:
                    raise

    def list_stages(self) -> list[str]:
        """Return available stage names."""

        return list(self.config.stages.keys())


def load_plan_config(path: Path) -> PlanConfig:
    """Read configuration from disk."""

    payload = _load_yaml(path)
    return PlanConfig.from_mapping(payload, base_dir=path.parent)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the plan runner."""

    parser = argparse.ArgumentParser(description="Execute the research experimentation plan.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research_plan.yaml"),
        help="Path to the research plan YAML configuration.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        help="Name of a stage to execute. Can be passed multiple times.",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="List available stages and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the tasks without executing them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    """Entry point for command-line execution."""

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG)

    config_path = args.config.resolve()
    config = load_plan_config(config_path)
    runner = PlanRunner(config)

    if args.list_stages:
        for name in runner.list_stages():
            print(name)
        return

    stages = args.stage or runner.list_stages()
    for stage_name in stages:
        runner.run_stage(stage_name, dry_run=args.dry_run)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
