from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from src.scripts.run_research_plan import PlanRunner, load_plan_config


def _write_plan(tmp_path: Path) -> Path:
    helper_module = tmp_path / "helper_module.py"
    helper_module.write_text(
        "from pathlib import Path\n"
        "Path('module_output.txt').write_text('module', encoding='utf-8')\n",
        encoding="utf-8",
    )

    config_text = """
default_cwd: .
stages:
  sample:
    description: sample stage
    tasks:
      - name: create_directory
        kind: mkdir
        directory: generated
      - name: run_command
        kind: command
        command:
          - {python}
          - -c
          - from pathlib import Path; Path('command_output.txt').write_text('command', encoding='utf-8')
      - name: run_module
        kind: module
        module: helper_module
      - name: append_log
        kind: log
        log_file: logs/plan.log
        message: executed stage
"""

    path = tmp_path / "plan.yaml"
    path.write_text(config_text.strip(), encoding="utf-8")
    return path


def test_plan_runner_executes_tasks(tmp_path, monkeypatch):
    plan_path = _write_plan(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    config = load_plan_config(plan_path)
    runner = PlanRunner(config, variables={"python": sys.executable})
    runner.run_stage("sample")

    assert (tmp_path / "generated").is_dir()
    assert (tmp_path / "command_output.txt").read_text(encoding="utf-8") == "command"
    assert (tmp_path / "module_output.txt").read_text(encoding="utf-8") == "module"
    log_lines = (tmp_path / "logs/plan.log").read_text(encoding="utf-8").splitlines()
    assert any("executed stage" in line for line in log_lines)


def test_plan_runner_dry_run(tmp_path, monkeypatch):
    plan_path = _write_plan(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    config = load_plan_config(plan_path)
    runner = PlanRunner(config, variables={"python": sys.executable})
    runner.run_stage("sample", dry_run=True)

    assert not (tmp_path / "generated").exists()
    assert not (tmp_path / "command_output.txt").exists()
    assert not (tmp_path / "module_output.txt").exists()
    assert not (tmp_path / "logs/plan.log").exists()
