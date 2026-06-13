"""Observability: every task leaves a durable record (even with no admission), crashes are
diagnosable, and the rolling checkpoint restores the exact latest state. The fix for runs that
used to leave empty directories whenever nothing new was shelved."""

import json
from pathlib import Path

from ardevo.dataset.icarus import Task
from ardevo.evolution.multitask import task_entry
from ardevo.orchestrator import Attempt
from ardevo.trials.orchestrated_trial import OrchestratedTrial
from tests.test_orchestrator import _orchestrator


class _StubScheduler:
    def state_dict(self) -> dict:
        return {"cursor": 2}

    def load_state_dict(self, data: dict) -> None:
        self.loaded = data


def _trial(tmp_path: Path, orchestrator) -> OrchestratedTrial:
    trial = object.__new__(OrchestratedTrial)
    trial.config = {"orchestrator": {}, "schedule": {}, "evolution": {}, "fitness": {}, "dataset": "synthetic"}
    trial.library = orchestrator.library
    trial.loop = orchestrator.loop
    trial.scheduler = _StubScheduler()
    trial.run_dir = tmp_path / "run"
    trial.run_dir.mkdir(parents=True, exist_ok=True)
    trial.task = None
    trial.rungs = [1, 2, 3]
    trial.skipped_rungs = []
    trial.task_records = []
    trial.checkpoint_every = 1
    trial.tasks_to_run = 10
    return trial


def test_run_summary_records_a_task_that_admits_nothing(tmp_path: Path, xor_task: Task) -> None:
    orchestrator = _orchestrator(tmp_path)
    trial = _trial(tmp_path, orchestrator)
    entry = task_entry(xor_task)
    attempt = Attempt(task=entry.name, depth=0, outcome="failed", metric=0.42, generations=11, strategy="direct")

    trial._record_task(entry, attempt, new_library_keys=[], library_size=len(trial.library))
    trial._write_run_summary(orchestrator, orchestrator.state, task_cursor=1, status="running")

    summary = json.loads((trial.run_dir / "run_summary.json").read_text())
    assert summary["status"] == "running"
    assert summary["tasks_attempted"] == 1
    assert summary["outcomes"] == {"failed": 1}
    assert len(summary["tasks"]) == 1
    row = summary["tasks"][0]
    assert row["task"] == entry.name and row["rung"] == entry.rung
    assert row["outcome"] == "failed" and row["strategy"] == "direct" and row["new_library_keys"] == []


def test_crash_leaves_a_diagnosable_summary(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    trial = _trial(tmp_path, orchestrator)
    trial._write_run_summary(orchestrator, orchestrator.state, task_cursor=0, status="crashed: ValueError: boom")
    summary = json.loads((trial.run_dir / "run_summary.json").read_text())
    assert summary["status"].startswith("crashed") and "boom" in summary["status"]


def test_rolling_checkpoint_restores_exact_state(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    trial = _trial(tmp_path, orchestrator)
    orchestrator.state.generation = 7
    orchestrator.counters["accepts"] = 3
    orchestrator.attempts = [Attempt(task="xor", depth=0, outcome="evolved", metric=1.0, generations=5, strategy="direct")]

    trial._persist_resume_state(orchestrator, orchestrator.state, task_cursor=5)
    assert (trial.run_dir / "checkpoint.json").exists()

    state, cursor, attempts, counters = trial._restore()
    assert cursor == 5
    assert counters["accepts"] == 3
    assert state.generation == 7
    assert len(attempts) == 1 and attempts[0].outcome == "evolved"


def test_resume_reloads_prior_task_records(tmp_path: Path, xor_task: Task) -> None:
    orchestrator = _orchestrator(tmp_path)
    trial = _trial(tmp_path, orchestrator)
    entry = task_entry(xor_task)
    trial._record_task(entry, Attempt(task=entry.name, depth=0, outcome="evolved", metric=1.0, generations=3), [], 1)
    trial._write_run_summary(orchestrator, orchestrator.state, task_cursor=1, status="running")

    recovered = trial._load_prior_records()
    assert len(recovered) == 1 and recovered[0]["task"] == entry.name


def test_load_prior_records_tolerates_missing_summary(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    trial = _trial(tmp_path, orchestrator)
    assert trial._load_prior_records() == []
