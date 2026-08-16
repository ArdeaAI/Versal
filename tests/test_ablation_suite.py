"""Ablation orchestration stays manifest-driven, resumable, and pure under test."""

import json
from pathlib import Path

import pytest

from versal.tools import ablation_suite
from versal.tools.ablation_suite import (
    ArmSpec,
    RunSpec,
    SuiteManifest,
    _archive_complete,
    aggregate_suite,
    assign_gpus,
    confidence_interval,
    execute_gpu_queue,
    execute_run,
    load_manifest,
    materialize_run_config,
    render_slurm_script,
    requests_cuda,
    select_runs,
    state_path,
)
from versal.utils.config import Config


def test_checked_in_manifest_expands_p0_and_p1_seed_protocols() -> None:
    manifest = load_manifest()
    runs = select_runs(manifest)

    assert manifest.baseline == "full"
    assert len(manifest.arms) == 11
    assert len(runs) == 45
    assert all(len(arm.seeds) == (5 if arm.priority == "P0" else 3) for arm in manifest.arms)
    assert {arm.name for arm in manifest.arms if arm.priority == "P0"} == {
        "full",
        "no_routing",
        "no_decomposition",
        "no_hierarchy",
        "no_refinement",
        "no_library_reuse",
    }
    assert {arm.name for arm in manifest.arms if arm.priority == "P1"} == {
        "no_curriculum",
        "no_self_adaptive",
        "no_archive_diversity",
        "freeze_only",
        "no_motifs_macros",
    }

    unrouted = Config(next(arm.config for arm in manifest.arms if arm.name == "no_routing")).current
    assert "routed" not in unrouted["orchestrator"]["evolve"]
    assert [Path(source["path"]).name for source in unrouted["config_sources"]] == [
        "canary.toml",
        "full_cluster.toml",
        "base.toml",
        "no_routing.toml",
    ]

    full_campaign = load_manifest(Config.PROJECT_ROOT / "configs" / "campaigns" / "full_cluster.toml")
    full_runs = select_runs(full_campaign)
    assert len(full_runs) == 3
    assert full_campaign.slurm["max_parallel"] == 3
    assert all(requests_cuda(spec) for spec in full_runs)
    assert all(requests_cuda(run) for run in runs)


def test_selection_and_gpu_assignment_are_deterministic() -> None:
    manifest = load_manifest()
    runs = select_runs(manifest, priorities={"P1"}, arms={"freeze_only", "no_self_adaptive"})

    assert [(run.index, run.arm.name, run.seed) for run in runs] == [
        (33, "no_self_adaptive", 0),
        (34, "no_self_adaptive", 1),
        (35, "no_self_adaptive", 2),
        (39, "freeze_only", 0),
        (40, "freeze_only", 1),
        (41, "freeze_only", 2),
    ]
    assert [gpu for _run, gpu in assign_gpus(runs, ["2", "5"])] == ["2", "5", "2", "5", "2", "5"]


def test_local_lattice_cuda_profile_requests_a_gpu() -> None:
    arm = ArmSpec(
        name="local_lattice",
        priority="P0",
        config=Config.PROJECT_ROOT / "configs" / "canary-lattice.toml",
        seeds=(0,),
        lever="",
        claim="",
        measure="",
    )

    assert requests_cuda(RunSpec(index=0, arm=arm, seed=0))


def test_materialized_run_config_hashes_seed_library_and_compute(tmp_path: Path) -> None:
    manifest = load_manifest()
    spec = select_runs(manifest, arms={"full"})[2]

    config_path, effective = materialize_run_config(tmp_path / "suite", spec, compute="cuda")

    assert effective["seed"] == 2
    assert effective["compute"] == "cuda"
    assert effective["machine_env"] == "ClusterCUDA"
    assert effective["orchestrator"]["library_dir"] == str((tmp_path / "suite" / "libraries" / "full" / "seed2").resolve())
    assert Path(effective["config_sources"][-1]["path"]) == config_path
    assert Path(effective["config_sources"][-2]["path"]).name == "full.toml"
    assert effective["config_effective_sha256"] != Config(spec.arm.config).current["config_effective_sha256"]


def test_completed_run_state_is_skipped_without_launching(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = load_manifest()
    spec = select_runs(manifest, arms={"full"})[0]
    output = tmp_path / "suite"
    config_path, effective = materialize_run_config(output, spec)
    run_dir = tmp_path / "results" / "done_orchestrated"
    run_dir.mkdir(parents=True)
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "status": "done",
                "tasks": [],
                "seed": spec.seed,
                "config_path": str(config_path),
                "config_effective_sha256": effective["config_effective_sha256"],
                "library_dir": effective["orchestrator"]["library_dir"],
            }
        )
    )
    (run_dir / "archive_state.json").write_text(json.dumps({"status": "complete", "snapshot_status": "done"}))
    path = state_path(output, spec)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"status": "running", "run_dir": str(run_dir), "attempt": 1}))

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("a completed ablation run must not launch another process")

    monkeypatch.setattr(ablation_suite.subprocess, "run", forbidden)
    state = execute_run(spec, output=output, results_root=tmp_path / "results")

    assert state["status"] == "done"
    assert state["attempt"] == 1


def test_slurm_script_reenters_one_manifest_index(tmp_path: Path) -> None:
    manifest = load_manifest()
    priorities = {"P0"}
    runs = select_runs(manifest, priorities=priorities)

    script = render_slurm_script(
        manifest,
        runs,
        output=tmp_path / "suite",
        results_root=tmp_path / "results",
        max_parallel=4,
        priorities=priorities,
    )

    assert "#SBATCH --array=0-29%4" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert '--run-index "${SLURM_ARRAY_TASK_ID}"' in script
    assert "--priorities P0" in script
    assert "--compute cuda" in script
    assert f"cd {Config.PROJECT_ROOT}" in script


def test_gpu_queue_runs_each_devices_assignments_serially(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = load_manifest()
    specs = select_runs(manifest, arms={"full"})[:3]
    calls: list[tuple[str, str | None]] = []

    def fake_execute(spec: RunSpec, **kwargs: object) -> dict[str, object]:
        calls.append((spec.run_id, str(kwargs.get("gpu"))))
        return {"status": "done"}

    monkeypatch.setattr(ablation_suite, "execute_run", fake_execute)

    failures = execute_gpu_queue(
        [(spec, "3") for spec in specs],
        output=tmp_path / "suite",
        results_root=tmp_path / "results",
        compute="cuda",
        dry_run=False,
    )

    assert failures == 0
    assert calls == [(spec.run_id, "3") for spec in specs]


def test_cold_run_refuses_prepopulated_library_without_matching_resume(tmp_path: Path) -> None:
    manifest = load_manifest()
    spec = select_runs(manifest, arms={"full"})[0]
    library = tmp_path / "suite" / "libraries" / "full" / "seed0"
    library.mkdir(parents=True)
    (library / "unexpected.json").write_text("{}")

    with pytest.raises(FileExistsError, match="cold-run library is not empty"):
        execute_run(spec, output=tmp_path / "suite", results_root=tmp_path / "results", dry_run=True)


def test_resume_refuses_summary_from_different_effective_config(tmp_path: Path) -> None:
    manifest = load_manifest()
    spec = select_runs(manifest, arms={"full"})[0]
    output = tmp_path / "suite"
    config_path, effective = materialize_run_config(output, spec)
    run_dir = tmp_path / "results" / "stale_orchestrated"
    run_dir.mkdir(parents=True)
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "status": "running",
                "seed": spec.seed,
                "config_path": str(config_path),
                "config_effective_sha256": "different",
                "library_dir": effective["orchestrator"]["library_dir"],
            }
        )
    )
    path = state_path(output, spec)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"run_dir": str(run_dir), "config_effective_sha256": effective["config_effective_sha256"]}))

    with pytest.raises(ValueError, match="saved run provenance"):
        execute_run(spec, output=output, results_root=tmp_path / "results", dry_run=True)


def test_archive_enabled_run_is_done_only_after_done_snapshot(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    effective = {"archive": {"enabled": True}}
    assert not _archive_complete(run_dir, effective)
    (run_dir / "archive_state.json").write_text(json.dumps({"status": "complete", "snapshot_status": "running"}))
    assert not _archive_complete(run_dir, effective)
    (run_dir / "archive_state.json").write_text(json.dumps({"status": "complete", "snapshot_status": "done"}))
    assert _archive_complete(run_dir, effective)
    assert _archive_complete(tmp_path / "missing", {"archive": {"enabled": False}})


def test_aggregate_suite_reports_seed_confidence_intervals(tmp_path: Path) -> None:
    config = tmp_path / "arm.toml"
    config.write_text("[orchestrator]\ntasks = 2\n")
    arm = ArmSpec("full", "P0", config, (0, 1), "full", "reference", "metrics")
    manifest = SuiteManifest(tmp_path / "manifest.toml", "tiny", "", "full", tmp_path / "suite", (arm,), {})
    runs = [RunSpec(0, arm, 0), RunSpec(1, arm, 1)]
    task_sets = [
        [
            {
                "rung": 1,
                "task": "a",
                "outcome": "evolved",
                "metric": 1.0,
                "report_metric": 0.8,
                "seconds": 1.0,
                "new_library_keys": ["m1"],
                "resource_metrics": {"direct_resource_declined": 1.0},
            },
            {"rung": 2, "task": "b", "outcome": "failed", "metric": 0.2, "seconds": 2.0, "new_library_keys": []},
        ],
        [
            {"rung": 1, "task": "a", "outcome": "library_hit", "metric": 1.0, "report_metric": 0.9, "seconds": 3.0, "new_library_keys": []},
            {"rung": 2, "task": "b", "outcome": "refined", "metric": 0.6, "seconds": 4.0, "new_library_keys": []},
        ],
    ]
    for spec, tasks in zip(runs, task_sets):
        run_dir = tmp_path / f"run_{spec.seed}"
        run_dir.mkdir()
        (run_dir / "run_summary.json").write_text(
            json.dumps({"status": "done", "seed": spec.seed, "tasks": tasks, "library_size": spec.seed + 1, "counters": {"resource_declines": 1 - spec.seed}})
        )
        path = state_path(tmp_path / "suite", spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": "done", "run_dir": str(run_dir)}))

    aggregate = aggregate_suite(manifest, runs, tmp_path / "suite")

    solve_rate = aggregate["arms"]["full"]["overall"]["solve_rate"]
    assert solve_rate == confidence_interval([0.5, 1.0])
    assert solve_rate["mean"] == 0.75
    assert aggregate["arms"]["full"]["rungs"]["2"]["solve_rate"]["mean"] == 0.5
    assert aggregate["arms"]["full"]["overall"]["mean_report_metric"]["mean"] == pytest.approx(0.85)
    assert aggregate["arms"]["full"]["overall"]["report_coverage"]["mean"] == 0.5
    assert aggregate["arms"]["full"]["overall"]["library_hit_rate"]["mean"] == 0.25
    assert aggregate["arms"]["full"]["overall"]["resource_decline_rate"]["mean"] == 0.25
    assert aggregate["arms"]["full"]["overall"]["final_library_size"]["mean"] == 1.5
    assert (tmp_path / "suite" / "aggregate.json").exists()
    assert (tmp_path / "suite" / "aggregate.csv").exists()
