"""Manifest-driven, resumable ablation execution and aggregation.

Examples:
    uv run ablation_suite --priorities P0 --gpus 0,1,2,3
    uv run ablation_suite --emit-slurm results/ablations/p0.slurm --priorities P0
    uv run ablation_suite --aggregate-only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ardevo.tools.run_matrix import SOLVED_OUTCOMES, flatten_rows, load_summary
from ardevo.utils.config import Config
from ardevo.utils.logging import Logger

console = Logger.get_console()
DEFAULT_MANIFEST = Config.PROJECT_ROOT / "configs" / "ablations" / "manifest.toml"
ABLATION_SOLVED_OUTCOMES = SOLVED_OUTCOMES | {"decomposed"}
_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_THREAD_LAUNCH_LOCK = threading.Lock()


@dataclass(frozen=True)
class ArmSpec:
    name: str
    priority: str
    config: Path
    seeds: tuple[int, ...]
    lever: str
    claim: str
    measure: str


@dataclass(frozen=True)
class SuiteManifest:
    path: Path
    name: str
    description: str
    baseline: str
    results_root: Path
    arms: tuple[ArmSpec, ...]
    slurm: dict[str, Any]


@dataclass(frozen=True)
class RunSpec:
    index: int
    arm: ArmSpec
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.index:03d}_{self.arm.name}_seed{self.seed}"


def load_manifest(path: Path = DEFAULT_MANIFEST) -> SuiteManifest:
    """Parse and validate one suite manifest; config paths are relative to the manifest."""
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"ablation manifest not found: {path}")
    raw = tomllib.loads(path.read_text())
    if int(raw.get("version", 0)) != 1:
        raise ValueError(f"unsupported ablation manifest version {raw.get('version')!r}")
    suite = raw.get("suite", {})
    seed_sets = {str(key).lower(): tuple(int(seed) for seed in value) for key, value in raw.get("seeds", {}).items()}
    arms: list[ArmSpec] = []
    seen: set[str] = set()
    for item in raw.get("arms", []):
        name = str(item.get("name", ""))
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError(f"invalid ablation arm name {name!r}")
        if name in seen:
            raise ValueError(f"duplicate ablation arm {name!r}")
        seen.add(name)
        priority = str(item.get("priority", "")).upper()
        if priority not in ("P0", "P1"):
            raise ValueError(f"arm {name!r} has invalid priority {priority!r}")
        seeds = tuple(int(seed) for seed in item.get("seeds", seed_sets.get(priority.lower(), ())))
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f"arm {name!r} must have a non-empty unique seed list")
        config = (path.parent / str(item["config"])).resolve()
        if not config.exists():
            raise FileNotFoundError(f"arm {name!r} config not found: {config}")
        # Loading here catches broken inheritance and malformed arm overlays before a cluster job.
        Config(config)
        arms.append(
            ArmSpec(
                name=name,
                priority=priority,
                config=config,
                seeds=seeds,
                lever=str(item.get("lever", "")),
                claim=str(item.get("claim", "")),
                measure=str(item.get("measure", "")),
            )
        )
    baseline = str(suite.get("baseline", ""))
    if baseline not in seen:
        raise ValueError(f"suite baseline {baseline!r} is not a declared arm")
    root = Path(str(suite.get("results_root", "results/ablations"))).expanduser()
    if not root.is_absolute():
        root = Config.PROJECT_ROOT / root
    return SuiteManifest(
        path=path,
        name=str(suite.get("name", path.stem)),
        description=str(suite.get("description", "")),
        baseline=baseline,
        results_root=root.resolve(),
        arms=tuple(arms),
        slurm=dict(raw.get("slurm", {})),
    )


def select_runs(manifest: SuiteManifest, *, priorities: set[str] | None = None, arms: set[str] | None = None) -> list[RunSpec]:
    selected = [arm for arm in manifest.arms if (not priorities or arm.priority in priorities) and (not arms or arm.name in arms)]
    if arms:
        missing = arms - {arm.name for arm in selected}
        if missing:
            raise ValueError(f"unknown or priority-filtered arm(s): {', '.join(sorted(missing))}")
    selected_names = {arm.name for arm in selected}
    # Run IDs use the global manifest index, so changing a CLI filter never forks state/library
    # identity for the same arm and seed. `--run-index` remains positional within the filtered list.
    return [RunSpec(index=index, arm=arm, seed=seed) for index, (arm, seed) in enumerate((arm, seed) for arm in manifest.arms for seed in arm.seeds) if arm.name in selected_names]


def assign_gpus(runs: list[RunSpec], gpus: list[str]) -> list[tuple[RunSpec, str | None]]:
    """Deterministic round-robin device assignment; an empty list preserves the environment."""
    return [(spec, gpus[index % len(gpus)] if gpus else None) for index, spec in enumerate(runs)]


def requests_cuda(spec: RunSpec) -> bool:
    """Whether the arm resolves to CUDA without a concrete CLI compute override."""

    config = Config(spec.arm.config).current
    compute = str(config.get("compute", "auto"))
    return compute.startswith("cuda") or (compute == "auto" and config.get("machine_env") in {"LatticeCUDA", "LocalLatticeCUDA", "ClusterCUDA"})


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def state_path(output: Path, spec: RunSpec) -> Path:
    return output / "runs" / f"{spec.run_id}.json"


def load_state(output: Path, spec: RunSpec) -> dict[str, Any]:
    path = state_path(output, spec)
    return json.loads(path.read_text()) if path.exists() else {}


def _library_path(output: Path, spec: RunSpec) -> Path:
    return (output / "libraries" / spec.arm.name / f"seed{spec.seed}").resolve()


def materialize_run_config(output: Path, spec: RunSpec, *, compute: str | None = None) -> tuple[Path, dict[str, Any]]:
    """Write the concrete seed/library/device overlay whose merged hash identifies this run."""
    path = (output / "configs" / f"{spec.run_id}.toml").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    relative_parent = os.path.relpath(spec.arm.config, path.parent)
    lines = [f"extends = {json.dumps(relative_parent)}", "", "[run]", f"seed = {spec.seed}"]
    if compute is not None:
        lines.append(f"compute = {json.dumps(compute)}")
        lines.append(f"machine = {json.dumps({'cuda': 'ClusterCUDA', 'mps': 'MonadMetal', 'cpu': 'local'}[compute])}")
    lines.extend(("", "[orchestrator]", f"library_dir = {json.dumps(str(_library_path(output, spec)))}", ""))
    payload = "\n".join(lines)
    if not path.exists() or path.read_text() != payload:
        path.write_text(payload)
    effective = Config(path).current
    return path, effective


def _summary_matches(summary: dict[str, Any], *, config_path: Path, config_effective_sha256: str, spec: RunSpec, library_path: Path) -> bool:
    try:
        recorded_config = Path(str(summary.get("config_path", ""))).expanduser().resolve()
        recorded_library = Path(str(summary.get("library_dir", ""))).expanduser().resolve()
    except (OSError, RuntimeError):
        return False
    return (
        recorded_config == config_path.resolve()
        and recorded_library == library_path.resolve()
        and int(summary.get("seed", -1)) == spec.seed
        and summary.get("config_effective_sha256") == config_effective_sha256
    )


def _archive_complete(run_dir: Path, effective: dict[str, Any]) -> bool:
    if not bool(effective.get("archive", {}).get("enabled", False)):
        return True
    try:
        state = json.loads((run_dir / "archive_state.json").read_text())
    except (OSError, TypeError, json.JSONDecodeError):
        return False
    return state.get("status") == "complete" and state.get("snapshot_status") == "done"


def find_run_dir(results_root: Path, *, config_path: Path, config_effective_sha256: str, spec: RunSpec, library_path: Path) -> Path | None:
    """Find this arm by stamped provenance, never by a racy before/after directory diff."""
    matches: list[Path] = []
    if results_root.exists():
        for candidate in results_root.glob("*_orchestrated"):
            summary = load_summary(candidate)
            if summary is not None and _summary_matches(
                summary,
                config_path=config_path,
                config_effective_sha256=config_effective_sha256,
                spec=spec,
                library_path=library_path,
            ):
                matches.append(candidate)
    return max(matches, key=lambda path: path.stat().st_mtime_ns) if matches else None


def _serialize_launch(output: Path) -> None:
    """Space timestamp-named run creation across local workers and Slurm array processes."""
    import fcntl

    lock_path = output / ".launch.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _THREAD_LAUNCH_LOCK, open(lock_path, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        try:
            previous = float(handle.read().strip() or 0.0)
        except ValueError:
            previous = 0.0
        delay = 1.1 - (time.time() - previous)
        if delay > 0:
            time.sleep(delay)
        handle.seek(0)
        handle.truncate()
        handle.write(str(time.time()))
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def build_command(config_path: Path, *, resume_dir: Path | None = None) -> list[str]:
    command = [sys.executable, "-m", "ardevo.main", "--config", str(config_path)]
    if resume_dir is not None:
        command.extend(("--resume", str(resume_dir)))
    return command


def execute_run(
    spec: RunSpec,
    *,
    output: Path,
    results_root: Path,
    gpu: str | None = None,
    compute: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Serialize one arm/seed across local invocations and Slurm array retries."""
    import fcntl

    lock_path = output.resolve() / "locks" / f"{spec.run_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"{spec.run_id} is already active under {output}") from error
        try:
            return _execute_run_unlocked(spec, output=output, results_root=results_root, gpu=gpu, compute=compute, dry_run=dry_run)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def execute_gpu_queue(
    assignments: list[tuple[RunSpec, str]],
    *,
    output: Path,
    results_root: Path,
    compute: str | None,
    dry_run: bool,
) -> int:
    """Run one device's assignments serially so two processes never share that GPU."""

    failures = 0
    for spec, gpu in assignments:
        try:
            state = execute_run(spec, output=output, results_root=results_root, gpu=gpu, compute=compute, dry_run=dry_run)
        except Exception as error:
            failures += 1
            console.print(f"[red]{spec.run_id}: {type(error).__name__}: {error}[/red]")
            continue
        failures += state["status"] == "failed"
        console.print(f"{spec.run_id}: {state['status']} gpu={gpu}")
    return failures


def _execute_run_unlocked(
    spec: RunSpec,
    *,
    output: Path,
    results_root: Path,
    gpu: str | None = None,
    compute: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run or resume one isolated arm after its per-run lock has been acquired."""
    output = output.resolve()
    selected_compute = compute or ("cuda" if gpu is not None else None)
    config_path, effective = materialize_run_config(output, spec, compute=selected_compute)
    library_path = _library_path(output, spec)
    previous = load_state(output, spec)
    previous_hash = previous.get("config_effective_sha256")
    if previous_hash is not None and previous_hash != effective["config_effective_sha256"]:
        raise ValueError(f"{spec.run_id}: effective config changed since state was created; use a new --output directory")
    run_dir = (
        Path(previous["run_dir"])
        if previous.get("run_dir")
        else find_run_dir(
            results_root,
            config_path=config_path,
            config_effective_sha256=effective["config_effective_sha256"],
            spec=spec,
            library_path=library_path,
        )
    )
    previous_summary = load_summary(run_dir) if run_dir is not None else None
    if previous_summary is not None and not _summary_matches(
        previous_summary,
        config_path=config_path,
        config_effective_sha256=effective["config_effective_sha256"],
        spec=spec,
        library_path=library_path,
    ):
        raise ValueError(f"{spec.run_id}: saved run provenance does not match the effective config; use a new --output directory")
    if previous_summary is not None and previous_summary.get("status") == "done" and run_dir is not None and _archive_complete(run_dir, effective):
        state = {**previous, "status": "done", "run_dir": str(run_dir), "returncode": 0}
        _atomic_json(state_path(output, spec), state)
        return state
    if run_dir is None and library_path.exists() and any(library_path.iterdir()):
        raise FileExistsError(f"{spec.run_id}: cold-run library is not empty and no matching resumable run exists: {library_path}")

    command = build_command(config_path, resume_dir=run_dir)
    state = {
        "schema_version": 1,
        "run_id": spec.run_id,
        "index": spec.index,
        "arm": spec.arm.name,
        "priority": spec.arm.priority,
        "seed": spec.seed,
        "source_config": str(spec.arm.config),
        "run_config": str(config_path),
        "config_effective_sha256": effective["config_effective_sha256"],
        "library_dir": str(library_path),
        "gpu": gpu,
        "compute": selected_compute,
        "status": "dry_run" if dry_run else "running",
        "attempt": int(previous.get("attempt", 0)) + (0 if dry_run else 1),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "command": command,
        "started_at": time.time(),
    }
    _atomic_json(state_path(output, spec), state)
    if dry_run:
        return state

    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    log_path = output / "logs" / f"{spec.run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _serialize_launch(output)
    with open(log_path, "a") as log:
        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)

    discovered = (
        find_run_dir(
            results_root,
            config_path=config_path,
            config_effective_sha256=effective["config_effective_sha256"],
            spec=spec,
            library_path=library_path,
        )
        or run_dir
    )
    summary = load_summary(discovered) if discovered is not None else None
    done = completed.returncode == 0 and summary is not None and summary.get("status") == "done" and discovered is not None and _archive_complete(discovered, effective)
    state.update(
        status="done" if done else "failed",
        returncode=completed.returncode,
        run_dir=str(discovered) if discovered is not None else None,
        run_status=summary.get("status") if summary else "missing run_summary.json",
        finished_at=time.time(),
        log_path=str(log_path),
    )
    _atomic_json(state_path(output, spec), state)
    return state


_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def confidence_interval(values: list[float]) -> dict[str, Any]:
    """Mean and two-sided 95% Student-t interval across independent seeds."""
    if not values:
        return {"n": 0, "mean": None, "ci95": [None, None]}
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {"n": 1, "mean": mean, "ci95": [mean, mean]}
    critical = _T_CRITICAL_95.get(len(values) - 1, 1.96)
    margin = critical * statistics.stdev(values) / math.sqrt(len(values))
    return {"n": len(values), "mean": mean, "ci95": [mean - margin, mean + margin]}


def _run_metrics(summary: dict[str, Any]) -> tuple[dict[str, float | None], dict[int, dict[str, float | None]]]:
    rows = flatten_rows(summary)

    def optional_mean(selected: list[dict[str, Any]], table: str, key: str) -> float | None:
        values = [float(row[table][key]) for row in selected if row.get(table, {}).get(key) is not None]
        return statistics.fmean(values) if values else None

    def metrics(selected: list[dict[str, Any]]) -> dict[str, float | None]:
        count = len(selected)
        report_metrics = [float(row["report_metric"]) for row in selected if row.get("report_metric") is not None]
        return {
            "solve_rate": sum(row.get("outcome") in ABLATION_SOLVED_OUTCOMES for row in selected) / count if count else 0.0,
            "mean_metric": statistics.fmean(float(row.get("metric") or 0.0) for row in selected) if count else 0.0,
            "mean_report_metric": statistics.fmean(report_metrics) if report_metrics else None,
            "report_coverage": len(report_metrics) / count if count else 0.0,
            "total_seconds": sum(float(row.get("seconds") or 0.0) for row in selected),
            "admissions": float(sum(int(row.get("new_library_keys") or 0) for row in selected)),
            "library_hit_rate": sum(row.get("outcome") == "library_hit" for row in selected) / count if count else 0.0,
            "refined_rate": sum(row.get("outcome") == "refined" for row in selected) / count if count else 0.0,
            "decomposed_rate": sum(row.get("outcome") == "decomposed" for row in selected) / count if count else 0.0,
            "routed_solve_rate": sum(row.get("strategy") == "routed" and row.get("outcome") in ABLATION_SOLVED_OUTCOMES for row in selected) / count if count else 0.0,
            "time_budget_rate": sum(row.get("failure_stage") == "time_budget" for row in selected) / count if count else 0.0,
            "resource_decline_rate": sum(any(name.endswith("_declined") and float(value) > 0.0 for name, value in row.get("resource_metrics", {}).items()) for row in selected)
            / count
            if count
            else 0.0,
            "mean_champion_complexity": optional_mean(selected, "size_metrics", "champion_complexity"),
            "mean_weight_robustness": optional_mean(selected, "sample_metrics", "weight_robustness"),
        }

    rungs = sorted({int(row["rung"]) for row in rows if row.get("rung") is not None})
    overall = metrics(rows)
    counters = summary.get("counters", {})
    overall.update(
        final_library_size=float(summary.get("library_size", 0)),
        resource_declines_total=float(counters.get("resource_declines", 0)),
        routed_solved_total=float(counters.get("routed_solved", 0)),
    )
    return overall, {rung: metrics([row for row in rows if row.get("rung") == rung]) for rung in rungs}


def aggregate_suite(manifest: SuiteManifest, runs: list[RunSpec], output: Path) -> dict[str, Any]:
    """Aggregate completed run summaries into per-arm and per-rung seed confidence intervals."""
    aggregate: dict[str, Any] = {"schema_version": 1, "suite": manifest.name, "baseline": manifest.baseline, "arms": {}}
    csv_rows: list[dict[str, Any]] = []
    selected_arm_names = {spec.arm.name for spec in runs}
    for arm in (item for item in manifest.arms if item.name in selected_arm_names):
        arm_runs = [spec for spec in runs if spec.arm.name == arm.name]
        run_values: list[dict[str, float | None]] = []
        rung_values: dict[int, list[dict[str, float | None]]] = {}
        statuses: dict[str, int] = {}
        for spec in arm_runs:
            state = load_state(output, spec)
            status = str(state.get("status", "pending"))
            statuses[status] = statuses.get(status, 0) + 1
            run_dir = Path(state["run_dir"]) if state.get("run_dir") else None
            summary = load_summary(run_dir) if run_dir is not None else None
            if summary is None or summary.get("status") != "done":
                continue
            overall, per_rung = _run_metrics(summary)
            run_values.append(overall)
            for rung, values in per_rung.items():
                rung_values.setdefault(rung, []).append(values)

        def aggregate_metrics(values: list[dict[str, float | None]]) -> dict[str, Any]:
            metrics = sorted({metric for row in values for metric in row})
            aggregated: dict[str, Any] = {}
            for metric in metrics:
                samples: list[float] = []
                for row in values:
                    value = row.get(metric)
                    if value is not None:
                        samples.append(float(value))
                aggregated[metric] = confidence_interval(samples)
            return aggregated

        overall = aggregate_metrics(run_values)
        per_rung = {str(rung): aggregate_metrics(values) for rung, values in sorted(rung_values.items())}
        aggregate["arms"][arm.name] = {
            "priority": arm.priority,
            "lever": arm.lever,
            "claim": arm.claim,
            "measure": arm.measure,
            "expected_runs": len(arm_runs),
            "statuses": statuses,
            "overall": overall,
            "rungs": per_rung,
        }
        for metric, estimate in overall.items():
            csv_rows.append(
                {
                    "arm": arm.name,
                    "priority": arm.priority,
                    "rung": "all",
                    "metric": metric,
                    "n": estimate["n"],
                    "mean": estimate["mean"],
                    "ci95_low": estimate["ci95"][0],
                    "ci95_high": estimate["ci95"][1],
                }
            )
        for rung, metrics in per_rung.items():
            for metric, estimate in metrics.items():
                csv_rows.append(
                    {
                        "arm": arm.name,
                        "priority": arm.priority,
                        "rung": rung,
                        "metric": metric,
                        "n": estimate["n"],
                        "mean": estimate["mean"],
                        "ci95_low": estimate["ci95"][0],
                        "ci95_high": estimate["ci95"][1],
                    }
                )
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "aggregate.json", aggregate)
    with open(output / "aggregate.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("arm", "priority", "rung", "metric", "n", "mean", "ci95_low", "ci95_high"))
        writer.writeheader()
        writer.writerows(csv_rows)
    return aggregate


def render_slurm_script(
    manifest: SuiteManifest,
    runs: list[RunSpec],
    *,
    output: Path,
    results_root: Path,
    max_parallel: int,
    priorities: set[str] | None = None,
    arms: set[str] | None = None,
) -> str:
    if not runs:
        raise ValueError("no runs selected for Slurm emission")
    slurm = manifest.slurm
    command = [
        "uv",
        "run",
        "ablation_suite",
        "--manifest",
        str(manifest.path),
        "--output",
        str(output.resolve()),
        "--results-root",
        str(results_root.resolve()),
        "--run-index",
        '"${SLURM_ARRAY_TASK_ID}"',
        "--compute",
        "cuda",
    ]
    if priorities:
        command.extend(("--priorities", ",".join(sorted(priorities))))
    if arms:
        command.extend(("--arms", ",".join(sorted(arms))))
    rendered_command = " ".join(item if item.startswith('"${') else shlex.quote(item) for item in command)
    return "\n".join(
        (
            "#!/bin/bash",
            f"#SBATCH --job-name={slurm.get('job_name', 'ardevo-ablate')}",
            f"#SBATCH --array=0-{len(runs) - 1}%{max(1, max_parallel)}",
            f"#SBATCH --time={slurm.get('time', '24:00:00')}",
            f"#SBATCH --cpus-per-task={int(slurm.get('cpus_per_task', 16))}",
            f"#SBATCH --mem={slurm.get('mem', '64G')}",
            f"#SBATCH --gres=gpu:{int(slurm.get('gpus', 1))}",
            "set -euo pipefail",
            f"cd {shlex.quote(str(Config.PROJECT_ROOT))}",
            rendered_command,
            "",
        )
    )


def _parse_set(value: str | None, *, upper: bool = False) -> set[str] | None:
    if not value:
        return None
    values = {item.strip() for item in value.split(",") if item.strip()}
    return {item.upper() for item in values} if upper else values


def main() -> None:
    parser = argparse.ArgumentParser(description="Run, resume, aggregate, or emit Slurm jobs for the checked-in ablation manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=None, help="Suite state/artifact directory (default: manifest suite.results_root).")
    parser.add_argument("--results-root", type=Path, default=Path("results"), help="Ordinary app run root used to discover run_summary.json.")
    parser.add_argument("--priorities", default=None, help="Comma-separated P0/P1 filter.")
    parser.add_argument("--arms", default=None, help="Comma-separated arm-name filter.")
    parser.add_argument("--gpus", default=None, help="Comma-separated local CUDA device IDs; jobs are assigned round-robin.")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--compute", choices=("cpu", "mps", "cuda"), default=None, help="Concrete run-config compute override.")
    parser.add_argument("--run-index", type=int, default=None, help="Execute one selected run (Slurm array seam).")
    parser.add_argument("--emit-slurm", type=Path, default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    output = (args.output or manifest.results_root).expanduser().resolve()
    priorities = _parse_set(args.priorities, upper=True)
    arms = _parse_set(args.arms)
    runs = select_runs(manifest, priorities=priorities, arms=arms)
    if not runs:
        raise SystemExit("no ablation runs selected")

    gpus = [item.strip() for item in (args.gpus or "").split(",") if item.strip()]
    if len(set(gpus)) != len(gpus):
        raise SystemExit("--gpus must contain unique device IDs")
    default_parallel = len(gpus) if gpus else int(manifest.slurm.get("max_parallel", 1)) if args.emit_slurm else 1
    max_parallel = args.max_parallel or default_parallel
    if max_parallel < 1:
        raise SystemExit("--max-parallel must be positive")
    if gpus and max_parallel > len(gpus):
        raise SystemExit("--max-parallel cannot exceed the number of --gpus (one run per GPU)")
    if gpus and args.compute not in (None, "cuda"):
        raise SystemExit("--gpus can only be combined with --compute cuda")
    implicit_cuda = args.compute is None and any(requests_cuda(spec) for spec in runs)
    if not args.emit_slurm and args.run_index is None and not gpus and max_parallel > 1 and (args.compute == "cuda" or implicit_cuda):
        raise SystemExit("parallel CUDA runs require explicit --gpus so each process has an exclusive device")
    if args.emit_slurm:
        script = render_slurm_script(
            manifest,
            runs,
            output=output,
            results_root=args.results_root,
            max_parallel=max_parallel,
            priorities=priorities,
            arms=arms,
        )
        args.emit_slurm.parent.mkdir(parents=True, exist_ok=True)
        args.emit_slurm.write_text(script)
        args.emit_slurm.chmod(0o755)
        console.print(f"wrote Slurm array script for {len(runs)} runs: {args.emit_slurm}")
        return
    if args.aggregate_only:
        aggregate_suite(manifest, runs, output)
        console.print(f"aggregation written to {output}")
        return
    if args.run_index is not None:
        if args.run_index < 0 or args.run_index >= len(runs):
            raise SystemExit(f"--run-index must be in [0, {len(runs) - 1}]")
        state = execute_run(runs[args.run_index], output=output, results_root=args.results_root, compute=args.compute, dry_run=args.dry_run)
        console.print(f"{state['run_id']}: {state['status']}")
        return

    failures = 0
    if gpus:
        active_gpus = gpus[:max_parallel]
        assignments = assign_gpus(runs, active_gpus)
        queues: dict[str, list[tuple[RunSpec, str]]] = {gpu: [] for gpu in active_gpus}
        for spec, assigned in assignments:
            assert assigned is not None
            queues[assigned].append((spec, assigned))
        with ThreadPoolExecutor(max_workers=len(active_gpus)) as executor:
            futures = [
                executor.submit(
                    execute_gpu_queue,
                    queue,
                    output=output,
                    results_root=args.results_root,
                    compute=args.compute,
                    dry_run=args.dry_run,
                )
                for queue in queues.values()
            ]
            for future in as_completed(futures):
                failures += future.result()
    else:
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {executor.submit(execute_run, spec, output=output, results_root=args.results_root, compute=args.compute, dry_run=args.dry_run): spec for spec in runs}
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    state = future.result()
                except Exception as error:
                    failures += 1
                    console.print(f"[red]{spec.run_id}: {type(error).__name__}: {error}[/red]")
                    continue
                failures += state["status"] == "failed"
                console.print(f"{spec.run_id}: {state['status']} gpu={state.get('gpu')}")
    if not args.dry_run:
        aggregate_suite(manifest, runs, output)
        console.print(f"aggregation written to {output}")
    if failures:
        raise SystemExit(f"{failures} ablation run(s) failed; rerun the same command to resume")


if __name__ == "__main__":
    main()
