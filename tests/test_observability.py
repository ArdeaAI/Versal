"""Observability: every task leaves a durable record (even with no admission), crashes are
diagnosable, and the rolling checkpoint restores the exact latest state. The fix for runs that
used to leave empty directories whenever nothing new was shelved."""

import hashlib
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


def test_attempts_carry_wall_clock_and_stage_forensics(xor_task: Task, tmp_path: Path) -> None:
    """Every solve stamps seconds + per-stage seconds so a wedged stage shows in run_summary.json
    instead of requiring a live process sample (the 2026-07-05 CIFAR mutation wedge)."""
    orchestrator = _orchestrator(tmp_path)
    orchestrator.solve(xor_task)
    attempt = orchestrator.attempts[-1]
    assert attempt.seconds > 0.0
    assert attempt.stage_seconds  # at least one strategy timed
    assert all(value >= 0.0 for value in attempt.stage_seconds.values())
    row = attempt.to_dict()
    assert row["seconds"] == attempt.seconds and row["stage_seconds"] == attempt.stage_seconds
    assert Attempt.from_dict(row).stage_seconds == attempt.stage_seconds


def test_attempts_from_old_checkpoints_without_timing_still_restore() -> None:
    legacy = {"task": "xor", "depth": 0, "outcome": "evolved", "metric": 1.0, "generations": 2}
    attempt = Attempt.from_dict(legacy)
    assert attempt.seconds == 0.0 and attempt.stage_seconds == {}
    assert "seconds" not in attempt.to_dict()  # zero stays absent: old summaries byte-identical


def test_attempt_sample_metrics_round_trip_and_absent_when_empty() -> None:
    diagnostics = {"mean_sample_accuracy": 0.55, "max_sample_accuracy": 0.9, "best_sample_weight": -1.0, "weight_robustness": 0.4}
    attempt = Attempt(task="two_spirals", depth=0, outcome="failed", metric=0.67, generations=28, sample_metrics=dict(diagnostics))
    row = attempt.to_dict()
    assert row["sample_metrics"] == diagnostics
    assert Attempt.from_dict(row).sample_metrics == diagnostics
    legacy = {"task": "xor", "depth": 0, "outcome": "evolved", "metric": 1.0, "generations": 2}
    restored = Attempt.from_dict(legacy)
    assert restored.sample_metrics == {}
    assert "sample_metrics" not in restored.to_dict()  # standard-eval rows stay byte-identical


def test_run_summary_row_carries_sample_metrics(tmp_path: Path, xor_task: Task) -> None:
    orchestrator = _orchestrator(tmp_path)
    trial = _trial(tmp_path, orchestrator)
    entry = task_entry(xor_task)
    diagnostics = {"mean_sample_accuracy": 0.55, "max_sample_accuracy": 0.9}
    trial._record_task(entry, Attempt(task=entry.name, depth=0, outcome="failed", metric=0.67, generations=28, sample_metrics=diagnostics), [], 0)
    trial._record_task(entry, Attempt(task=entry.name, depth=0, outcome="failed", metric=0.5, generations=3), [], 0)
    assert trial.task_records[0]["sample_metrics"] == diagnostics
    assert "sample_metrics" not in trial.task_records[1]


def test_attempt_size_metrics_round_trip_and_absent_when_empty() -> None:
    sizes = {"champion_nodes": 56.0, "champion_connections": 292.0, "champion_complexity": 344.0, "pop_median_nodes": 200.0}
    attempt = Attempt(task="two_spirals", depth=0, outcome="failed", metric=0.9, generations=120, size_metrics=dict(sizes))
    row = attempt.to_dict()
    assert row["size_metrics"] == sizes
    assert Attempt.from_dict(row).size_metrics == sizes
    legacy = {"task": "xor", "depth": 0, "outcome": "evolved", "metric": 1.0, "generations": 2}
    restored = Attempt.from_dict(legacy)
    assert restored.size_metrics == {}
    assert "size_metrics" not in restored.to_dict()  # old checkpoints and empty rows stay byte-identical


def test_run_summary_row_carries_size_metrics_and_module_pool(tmp_path: Path, xor_task: Task) -> None:
    orchestrator = _orchestrator(tmp_path)
    trial = _trial(tmp_path, orchestrator)
    entry = task_entry(xor_task)
    sizes = {"champion_nodes": 56.0, "pop_median_nodes": 200.0}
    pool_sizes = {"pool_median_nodes": 200.0, "pool_max_nodes": 258.0}
    trial._record_task(entry, Attempt(task=entry.name, depth=0, outcome="failed", metric=0.9, generations=120, size_metrics=sizes), [], 0, pool_sizes)
    trial._record_task(entry, Attempt(task=entry.name, depth=0, outcome="failed", metric=0.5, generations=3), [], 0)
    assert trial.task_records[0]["size_metrics"] == sizes
    assert trial.task_records[0]["module_pool"] == pool_sizes
    assert "size_metrics" not in trial.task_records[1] and "module_pool" not in trial.task_records[1]


def test_real_solve_stamps_size_metrics_and_pool_sizes(tmp_path: Path, xor_task: Task) -> None:
    orchestrator = _orchestrator(tmp_path)
    orchestrator.solve(xor_task)
    sizes = orchestrator.attempts[-1].size_metrics
    assert sizes and sizes["champion_complexity"] >= 0.0  # the default ladder is composition-shaped
    assert "champion_modules" in sizes
    pool_sizes = OrchestratedTrial._module_pool_sizes(orchestrator.state)
    assert pool_sizes["pool_median_nodes"] > 0.0 and pool_sizes["pool_max_nodes"] >= pool_sizes["pool_median_nodes"]


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


def test_run_summary_carries_config_provenance(tmp_path: Path, xor_task: Task) -> None:
    """A results directory identifies its exact config, seed, and library; a reviewer (or the
    matrix driver) never has to match a run to a config by schedule shape again."""
    orchestrator = _orchestrator(tmp_path)
    trial = _trial(tmp_path, orchestrator)
    trial.config = dict(trial.config, config_path="configs/recon_ladder.toml", config_sha256="ab" * 32, seed=3)
    trial._write_run_summary(orchestrator, orchestrator.state, task_cursor=0, status="running")
    summary = json.loads((trial.run_dir / "run_summary.json").read_text())
    assert summary["config_path"] == "configs/recon_ladder.toml"
    assert summary["config_sha256"] == "ab" * 32
    assert summary["seed"] == 3
    assert summary["library_dir"] == str(trial.library.root)


def test_run_summary_tolerates_configs_without_provenance(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    trial = _trial(tmp_path, orchestrator)  # the stub config has no provenance keys
    trial._write_run_summary(orchestrator, orchestrator.state, task_cursor=0, status="running")
    summary = json.loads((trial.run_dir / "run_summary.json").read_text())
    assert summary["config_path"] == "" and summary["config_sha256"] == "" and summary["seed"] == 0


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


def test_fresh_per_task_freezes_the_starting_library_once(tmp_path: Path) -> None:
    from ardevo.library import ModuleLibrary

    trial = object.__new__(OrchestratedTrial)
    trial.fresh_per_task = True
    trial.run_dir = tmp_path / "run"
    trial.run_dir.mkdir()
    trial.library = ModuleLibrary(tmp_path / "library")
    trial.frozen_library_dir = None
    trial.library.root.mkdir(parents=True, exist_ok=True)
    (trial.library.root / "before.txt").write_text("before")

    trial._prepare_frozen_library()
    assert trial.frozen_library_dir == trial.run_dir / "frozen_library"
    frozen_library_dir = trial.frozen_library_dir
    assert frozen_library_dir is not None
    assert (frozen_library_dir / "before.txt").read_text() == "before"

    (trial.library.root / "after.txt").write_text("after")
    trial._prepare_frozen_library()
    assert not (frozen_library_dir / "after.txt").exists()


def test_new_run_snapshots_exact_config_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.toml"
    source.write_bytes(b"[run]\nseed = 7\n")
    trial = object.__new__(OrchestratedTrial)
    trial.run_dir = tmp_path / "run"
    trial.run_dir.mkdir()
    trial.config = {"config_path": str(source), "config_sha256": "stale", "seed": 9, "orchestrator": {"library_dir": "override"}}
    trial._snapshot_config()
    assert (trial.run_dir / "config.toml").read_bytes() == source.read_bytes()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert (trial.run_dir / "config.toml.sha256").read_text() == f"{source_hash}  config.toml\n"
    effective = json.loads((trial.run_dir / "config.effective.json").read_text())
    assert effective["seed"] == 9 and effective["orchestrator"]["library_dir"] == "override"
    effective_hash = hashlib.sha256((trial.run_dir / "config.effective.json").read_bytes()).hexdigest()
    assert (trial.run_dir / "config.effective.json.sha256").read_text() == f"{effective_hash}  config.effective.json\n"


def test_require_all_rungs_gates_construction(tmp_path: Path, xor_task: Task, monkeypatch) -> None:
    """A rung that fails to LOAD aborts the run when require_all_rungs is set; the default stays
    tolerant (the pre-existing skip-and-report behavior)."""
    import pytest

    from ardevo.evolution import multitask
    from ardevo.trials import orchestrated_trial
    from tests.test_hierarchical_loop import _config as _loop_config

    report = multitask.PoolReport(entries=[task_entry(xor_task)], skipped=[multitask.SkippedRung(rung=7, error_type="RuntimeError", message="synthetic load failure")])
    monkeypatch.setattr(orchestrated_trial, "build_pool_report", lambda **_kwargs: report)

    config = _loop_config()
    config.update({"dataset": "synthetic", "n_samples": 4, "seed": 0})
    config["orchestrator"] = {"tasks": 1, "library_dir": str(tmp_path / "lib")}
    config["schedule"] = {"kind": "interleave_rungs", "rungs": [1, 7], "require_all_rungs": True}
    with pytest.raises(RuntimeError, match="require_all_rungs.*rung 7"):
        OrchestratedTrial(config)

    config["schedule"]["require_all_rungs"] = False
    trial = OrchestratedTrial(config)  # tolerant default: skips are reported, not fatal
    assert [skipped.rung for skipped in trial.skipped_rungs] == [7]


def test_schedule_task_name_filters_are_deterministic(tmp_path: Path, xor_task: Task, monkeypatch) -> None:
    from dataclasses import replace

    from ardevo.evolution import multitask
    from ardevo.trials import orchestrated_trial
    from tests.test_hierarchical_loop import _config as _loop_config

    train = replace(xor_task, meta=replace(xor_task.meta, name="arc.train.one"))
    evaluation = replace(xor_task, meta=replace(xor_task.meta, name="arc.eval.one"))
    report = multitask.PoolReport(entries=[task_entry(train), task_entry(evaluation)], skipped=[])
    monkeypatch.setattr(orchestrated_trial, "build_pool_report", lambda **_kwargs: report)
    config = _loop_config()
    config.update({"dataset": "synthetic", "n_samples": 4, "seed": 0})
    config["orchestrator"] = {"tasks": 1, "library_dir": str(tmp_path / "lib")}
    config["schedule"] = {"kind": "interleave_rungs", "rungs": [18], "task_include": ["arc.*"], "task_exclude": ["*.train.*"]}
    trial = OrchestratedTrial(config)
    assert [entry.name for entry in trial.pool] == ["arc.eval.one"]


def test_library_gc_cli_dry_run_and_checkpoint_protection(tmp_path: Path, monkeypatch) -> None:
    from ardevo.evolution.genome import genome_to_dict
    from ardevo.library import MODULE, ModuleLibrary
    from ardevo.tools.library_gc import checkpoint_macro_refs, run_gc
    from tests.test_recurrence import _running_parity_genome

    genome = _running_parity_genome()
    library = ModuleLibrary(tmp_path / "lib")
    io = {"inputs": [{"signature": "BINARY|K", "width": 2}], "output": {"signature": "BINARY|K", "width": 1}}
    doomed = library.add(entry_type=MODULE, payload=genome_to_dict(genome), io=io, provenance={"accepted_metric": 0.9})
    library.retire(doomed)

    # A resumable checkpoint whose pooled genome carries a macro ref to the doomed key protects it.
    run_dir = tmp_path / "results" / "20990101_000000_orchestrated"
    run_dir.mkdir(parents=True)
    pool_genome = genome_to_dict(genome)
    pool_genome["macros"] = [{"ref": f"library:{doomed}", "inputs": [0], "outputs": [1], "innovation": 1, "trainable": False}]
    (run_dir / "checkpoint.json").write_text(json.dumps({"loop_state": {"modules": [{"genome": pool_genome}], "species_champions": {}}}))
    assert checkpoint_macro_refs(run_dir / "checkpoint.json") == {doomed}

    protected = run_gc(tmp_path / "lib", dry_run=False, protect_checkpoint=True, results_root=tmp_path / "results")
    assert protected["swept"] == [] and len(library.keys()) == 1  # the checkpoint pinned it

    dry = run_gc(tmp_path / "lib", dry_run=True, protect_checkpoint=False, results_root=tmp_path / "results")
    assert dry["swept"] == [doomed] and (tmp_path / "lib" / "entries" / f"{doomed}.json").exists()  # dry run touched nothing

    swept = run_gc(tmp_path / "lib", dry_run=False, protect_checkpoint=False, results_root=tmp_path / "results")
    assert swept["swept"] == [doomed] and not (tmp_path / "lib" / "entries" / f"{doomed}.json").exists()
