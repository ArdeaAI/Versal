"""run_matrix: the multi-seed experiment driver plus the per-rung tier scorecard.

    uv run run_matrix --config configs/canary.toml --seeds 0,1,2 --cold
    uv run run_matrix --scorecard results/<run>

One arm = (config, seed, cold or shared library). Arms execute sequentially as subprocesses of the
ordinary app entry point (crash isolation: a dead arm still leaves its run_summary.json and the
matrix moves on), and every aggregate here derives ONLY from run_summary.json, the always-on
durable record, so `--scorecard` also works on any past run directory.

Tiers, the plan's operational meaning of "meaningful result" per rung:
    T0  never attempted (or crashed before a record)
    T1  an attempt completed end to end with forensics
    T2  best champion carries real signal (metric >= t2 floor, default the wall-stone floor 0.45)
    T3  the rung contributed to the library (a solve or an admitted stepping stone)
    T4  solved at the accept bar (outcome evolved / refined / library_hit)
"""

import argparse
import csv
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from versal.utils.logging import Logger

console = Logger.get_console()

SOLVED_OUTCOMES = {"evolved", "refined", "library_hit"}
DEFAULT_T2_FLOOR = 0.45


def parse_rungs(spec: str) -> list[int]:
    rungs: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            low, high = part.split("-", 1)
            rungs.extend(range(int(low), int(high) + 1))
        elif part:
            rungs.append(int(part))
    return rungs


def load_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "run_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def flatten_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """One flat row per attempted task, stamped with the run's provenance columns."""
    provenance = {
        "run_dir": summary.get("run_dir", ""),
        "run_status": summary.get("status", ""),
        "config_path": summary.get("config_path", ""),
        "config_sha256": summary.get("config_sha256", ""),
        "seed": summary.get("seed", 0),
        "library_dir": summary.get("library_dir", ""),
    }
    rows: list[dict[str, Any]] = []
    for record in summary.get("tasks", []):
        row = dict(provenance)
        row.update(
            {
                "rung": record.get("rung"),
                "task": record.get("task"),
                "outcome": record.get("outcome"),
                "metric": record.get("metric", 0.0),
                "strategy": record.get("strategy"),
                "generations": record.get("generations", 0),
                "depth": record.get("depth", 0),
                "failure_stage": record.get("failure_stage"),
                "seconds": record.get("seconds", 0.0),
                "new_library_keys": len(record.get("new_library_keys", [])),
                "report_metric": record.get("report_metric"),
                "task_metrics": dict(record.get("task_metrics", {})),
                "sample_metrics": dict(record.get("sample_metrics", {})),
                "size_metrics": dict(record.get("size_metrics", {})),
                "resource_metrics": dict(record.get("resource_metrics", {})),
            }
        )
        rows.append(row)
    return rows


def rung_tier(rows: list[dict[str, Any]], *, t2_floor: float = DEFAULT_T2_FLOOR) -> str:
    """The tier a rung has EARNED across the given rows (max over attempts)."""
    if not rows:
        return "T0"
    if any(row.get("outcome") in SOLVED_OUTCOMES for row in rows):
        return "T4"
    if any(row.get("new_library_keys", 0) > 0 for row in rows):
        return "T3"
    if any(float(row.get("metric") or 0.0) >= t2_floor for row in rows):
        return "T2"
    return "T1"


def build_scorecard(rows: list[dict[str, Any]], rungs: list[int], *, t2_floor: float = DEFAULT_T2_FLOOR) -> dict[str, Any]:
    """Per-rung tier plus supporting numbers, from flattened rows of one or more runs."""
    per_rung: dict[str, Any] = {}
    for rung in rungs:
        rung_rows = [row for row in rows if row.get("rung") == rung]
        metrics = [float(row.get("metric") or 0.0) for row in rung_rows]
        per_rung[str(rung)] = {
            "tier": rung_tier(rung_rows, t2_floor=t2_floor),
            "attempts": len(rung_rows),
            "best_metric": max(metrics) if metrics else None,
            "solves": sum(1 for row in rung_rows if row.get("outcome") in SOLVED_OUTCOMES),
            "admissions": sum(int(row.get("new_library_keys", 0)) for row in rung_rows),
            "time_budget_hits": sum(1 for row in rung_rows if row.get("failure_stage") == "time_budget"),
            "total_seconds": round(sum(float(row.get("seconds") or 0.0) for row in rung_rows), 1),
        }
    tiers = [entry["tier"] for entry in per_rung.values()]
    return {"t2_floor": t2_floor, "rungs": per_rung, "tier_counts": {tier: tiers.count(tier) for tier in ("T0", "T1", "T2", "T3", "T4")}}


def print_scorecard(scorecard: dict[str, Any]) -> None:
    from rich.table import Table

    table = Table(title="per-rung tier scorecard")
    for column in ("rung", "tier", "attempts", "best_metric", "solves", "admissions", "budget hits", "seconds"):
        table.add_column(column)
    for rung, entry in scorecard["rungs"].items():
        best = f"{entry['best_metric']:.3f}" if entry["best_metric"] is not None else "-"
        style = {"T4": "green", "T3": "cyan", "T2": "yellow", "T1": "white"}.get(entry["tier"], "red")
        table.add_row(
            rung,
            f"[{style}]{entry['tier']}[/{style}]",
            str(entry["attempts"]),
            best,
            str(entry["solves"]),
            str(entry["admissions"]),
            str(entry["time_budget_hits"]),
            str(entry["total_seconds"]),
        )
    console.print(table)
    console.print(f"tier counts: {scorecard['tier_counts']}")


def run_arm(config: Path, seed: int, library_dir: str | None, results_root: Path) -> tuple[Path | None, int]:
    """Run one arm as a subprocess; identify its run directory by diffing results/ around the run."""
    before = {path for path in results_root.iterdir() if path.is_dir()} if results_root.exists() else set()
    command = [sys.executable, "-m", "versal.main", "--config", str(config), "--seed", str(seed)]
    if library_dir is not None:
        command += ["--library-dir", library_dir]
    console.rule(f"[bold]arm: {config.name} seed={seed} library={library_dir or 'config default'}")
    completed = subprocess.run(command)
    after = {path for path in results_root.iterdir() if path.is_dir()} if results_root.exists() else set()
    new_dirs = sorted(after - before, key=lambda path: path.name)
    run_dir = next((path for path in reversed(new_dirs) if (path / "run_summary.json").exists()), None)
    return run_dir, completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed run driver + per-rung tier scorecard over run_summary.json records.")
    parser.add_argument("--config", type=str, default=None, help="Run config for the matrix arms.")
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds, e.g. 0,1,2.")
    parser.add_argument("--cold", action="store_true", help="Each arm gets its own fresh library under the matrix directory (the cold baseline).")
    parser.add_argument("--library-dir", type=str, default=None, help="Shared library dir for every arm (a warm arm); mutually exclusive with --cold.")
    parser.add_argument("--tag", type=str, default="", help="Suffix for the matrix output directory name.")
    parser.add_argument("--rungs", type=str, default="1-18", help="Rung range the scorecard reports (absent rungs show as T0).")
    parser.add_argument("--t2-floor", type=float, default=DEFAULT_T2_FLOOR, help="Metric floor for tier T2 (default: the wall-stone floor).")
    parser.add_argument("--scorecard", nargs="*", default=None, help="Skip running; aggregate these existing run directories instead.")
    args = parser.parse_args()

    rungs = parse_rungs(args.rungs)
    results_root = Path("results")

    if args.scorecard is not None:
        run_dirs = [Path(path) for path in args.scorecard]
        arms = []
    else:
        if not args.config:
            raise SystemExit("--config is required unless --scorecard is given")
        if args.cold and args.library_dir:
            raise SystemExit("--cold and --library-dir are mutually exclusive")
        config = Path(args.config)
        seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        matrix_dir = results_root / f"matrix_{timestamp}{'_' + args.tag if args.tag else ''}"
        matrix_dir.mkdir(parents=True, exist_ok=True)
        run_dirs, arms = [], []
        for seed in seeds:
            library_dir = str(matrix_dir / f"library_seed{seed}") if args.cold else args.library_dir
            run_dir, returncode = run_arm(config, seed, library_dir, results_root)
            arms.append({"config": str(config), "seed": seed, "library_dir": library_dir, "run_dir": str(run_dir) if run_dir else None, "returncode": returncode})
            if run_dir is None:
                console.print(f"[bold red]arm seed={seed} left no run_summary.json (returncode {returncode})")
            else:
                run_dirs.append(run_dir)

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        summary = load_summary(run_dir)
        if summary is None:
            console.print(f"[bold red]{run_dir} has no run_summary.json; skipped")
            continue
        rows.extend(flatten_rows(summary))

    scorecard = build_scorecard(rows, rungs, t2_floor=args.t2_floor)
    print_scorecard(scorecard)

    if args.scorecard is None and rows:
        matrix = {"arms": arms, "scorecard": scorecard}
        (matrix_dir / "matrix.json").write_text(json.dumps(matrix, indent=2))
        with open(matrix_dir / "matrix.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        console.print(f"matrix written to {matrix_dir}")


if __name__ == "__main__":
    main()
