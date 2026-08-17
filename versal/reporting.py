"""Deterministic, read-only run-summary reporting.

The reporter deliberately consumes only the durable run ledger, run-pinned config and task-pool
artifacts, and optional small library metadata. It never opens task checkpoints or entry payloads,
so it is safe to run against a live, crashed, resumed, or historical experiment directory.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from versal.utils.files import file_sha256

REPORT_SCHEMA_VERSION = 1
RUNG_FIELDS = (
    "rung",
    "task_count",
    "query_count",
    "query_coverage",
    "highest_held_out_accuracy",
    "mean_held_out_accuracy",
    "winning_task",
    "support_maximum",
    "support_mean",
    "accepted_outcomes",
    "failures",
    "admissions",
    "seconds",
    "summary",
)
ACCEPTED_OUTCOMES = frozenset(("evolved", "library_hit", "refined", "decomposed"))


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _tasks(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Read known ledgers while remaining forward-compatible with additive wrapper schemas."""

    for key in ("tasks", "task_records", "records"):
        value = summary.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    report = summary.get("report")
    if isinstance(report, dict):
        return _tasks(report)
    return []


def _held_out(row: dict[str, Any]) -> float | None:
    # New ledgers carry the literal sample-level rail explicitly; report_metric remains the
    # configurable legacy score (which may be task-exact rather than accuracy).
    if "query_accuracy" in row:
        return _finite(row.get("query_accuracy")) if row.get("query_status") == "evaluated" else None
    direct = _finite(row.get("report_metric"))
    if direct is not None:
        return direct
    task_metrics = row.get("task_metrics")
    if isinstance(task_metrics, dict):
        for key in ("query_accuracy", "query_covered_accuracy", "query_task_exact"):
            value = _finite(task_metrics.get(key))
            if value is not None:
                return value
    metrics = row.get("report_metrics")
    if isinstance(metrics, dict):
        for key in ("query_accuracy", "accuracy", "query_task_exact"):
            value = _finite(metrics.get(key))
            if value is not None:
                return value
    return None


def _support(row: dict[str, Any]) -> float | None:
    if "support_accuracy" in row:
        return _finite(row.get("support_accuracy")) if row.get("support_status") == "evaluated" else None
    value = _finite(row.get("metric"))
    if value is not None:
        return value
    metrics = row.get("champion_metrics")
    if isinstance(metrics, dict):
        for key in ("support_accuracy", "accuracy"):
            value = _finite(metrics.get(key))
            if value is not None:
                return value
    return None


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def _first_occurrences(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return independent task identities once, deriving repeat status for legacy ledgers."""

    seen: set[tuple[Any, Any]] = set()
    output: list[dict[str, Any]] = []
    for row in tasks:
        identity = (row.get("rung"), row.get("task"))
        repeated = bool(row.get("is_repeat")) if "is_repeat" in row else identity in seen
        if not repeated:
            output.append(row)
        seen.add(identity)
    return output


def _sentence(rung: int, task_count: int, query_values: list[float], support_values: list[float], failures: int) -> str:
    if not query_values:
        return f"Rung {rung} recorded {task_count} task(s), but no held-out query accuracy was available; {failures} task(s) failed."
    high = max(query_values)
    mean = sum(query_values) / len(query_values)
    support_mean = _mean(support_values)
    if support_mean is None:
        gap = "support coverage was unavailable"
    else:
        difference = support_mean - mean
        gap = f"the mean support-to-query gap was {difference:.4f}"
    return f"Rung {rung} covered {len(query_values)}/{task_count} held-out queries (mean {mean:.4f}, best {high:.4f}); {gap}, with {failures} failure(s)."


def _rung_rows(tasks: list[dict[str, Any]], configured_rungs: Iterable[Any]) -> list[dict[str, Any]]:
    by_rung: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in tasks:
        raw_rung = row.get("rung")
        if raw_rung is None:
            continue
        try:
            rung = int(raw_rung)
        except (TypeError, ValueError):
            continue
        by_rung[rung].append(row)
    rung_numbers = {int(rung) for rung in configured_rungs if str(rung).lstrip("-").isdigit()}
    rung_numbers.update(by_rung)
    output: list[dict[str, Any]] = []
    for rung in sorted(rung_numbers):
        rows = by_rung.get(rung, [])
        held = [(value, str(row.get("task", ""))) for row in rows if (value := _held_out(row)) is not None]
        support = [value for row in rows if (value := _support(row)) is not None]
        highest = max((item[0] for item in held), default=None)
        winner = min((task for value, task in held if value == highest), default="") if highest is not None else ""
        failures = sum(str(row.get("outcome", "")) == "failed" for row in rows)
        query_values = [value for value, _task in held]
        output.append(
            {
                "rung": rung,
                "task_count": len(rows),
                "query_count": len(held),
                "query_coverage": len(held) / len(rows) if rows else None,
                "highest_held_out_accuracy": highest,
                "mean_held_out_accuracy": _mean(query_values),
                "winning_task": winner or None,
                "support_maximum": max(support) if support else None,
                "support_mean": _mean(support),
                "accepted_outcomes": sum(str(row.get("outcome", "")) in ACCEPTED_OUTCOMES for row in rows),
                "failures": failures,
                "admissions": sum(len(row.get("new_library_keys", [])) for row in rows if isinstance(row.get("new_library_keys", []), list)),
                "seconds": sum(_finite(row.get("seconds")) or 0.0 for row in rows),
                "summary": _sentence(rung, len(rows), query_values, support, failures),
            }
        )
    return output


def _read_optional_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text()) if path.is_file() else None
    except (OSError, ValueError):
        return None


def build_run_report(run_dir: Path, library: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    if not isinstance(summary, dict):
        raise ValueError("run_summary.json must contain a JSON object")
    tasks = _tasks(summary)
    first_occurrences = _first_occurrences(tasks)
    held = [value for row in tasks if (value := _held_out(row)) is not None]
    support = [value for row in tasks if (value := _support(row)) is not None]
    first_held = [value for row in first_occurrences if (value := _held_out(row)) is not None]
    first_support = [value for row in first_occurrences if (value := _support(row)) is not None]
    outcome_counts = Counter(str(row.get("outcome", "unknown")) for row in tasks)
    strategy_counts = Counter(str(row.get("strategy")) for row in tasks if row.get("strategy") is not None)
    representation_counts = Counter(
        str(row.get("representation") or ("field" if row.get("strategy") == "field" else ("composition" if row.get("strategy") == "composition" else "flat")))
        for row in tasks
        if row.get("strategy") is not None
    )
    report_strategy_counts = Counter(str(row.get("report_strategy") or "legacy_unattributed") for row in tasks if _held_out(row) is not None)
    report_representation_counts = Counter(
        str(row.get("report_representation") or (row.get("report_strategy") if row.get("report_strategy") is not None else "legacy_unattributed"))
        for row in tasks
        if _held_out(row) is not None
    )
    stage_seconds: Counter[str] = Counter()
    resource_events: Counter[str] = Counter()
    strategy_metrics: dict[str, list[float]] = defaultdict(list)
    for row in tasks:
        if isinstance(row.get("stage_seconds"), dict):
            for name, value in row["stage_seconds"].items():
                if (number := _finite(value)) is not None:
                    stage_seconds[str(name)] += number
        if isinstance(row.get("resource_metrics"), dict):
            for name, value in row["resource_metrics"].items():
                if (number := _finite(value)) is not None and number:
                    resource_events[str(name)] += 1
        if isinstance(row.get("strategy_metrics"), dict):
            for name, value in row["strategy_metrics"].items():
                if (number := _finite(value)) is not None:
                    strategy_metrics[str(name)].append(number)
    gaps = [support_value - query for row in tasks if (support_value := _support(row)) is not None and (query := _held_out(row)) is not None]
    first_gaps = [support_value - query for row in first_occurrences if (support_value := _support(row)) is not None and (query := _held_out(row)) is not None]

    provenance_files: list[dict[str, Any]] = []
    for name in ("run_summary.json", "run_manifest.json", "config.toml", "config.effective.json", "task_pool.json"):
        path = run_dir / name
        if path.is_file():
            provenance_files.append({"path": name, "sha256": file_sha256(path), "size": path.stat().st_size})
    library_metadata: dict[str, Any] | None = None
    if library is not None:
        index_path = library.resolve() / "index.json"
        motifs_path = library.resolve() / "motifs.json"
        index = _read_optional_json(index_path)
        motifs = _read_optional_json(motifs_path)
        library_metadata = {
            "path": str(library.resolve()),
            "index_entries": len(index) if isinstance(index, list) else None,
            "index_sha256": file_sha256(index_path) if index_path.is_file() else None,
            "motifs_sha256": file_sha256(motifs_path) if motifs_path.is_file() else None,
            "motif_count": len(motifs.get("motifs", [])) if isinstance(motifs, dict) and isinstance(motifs.get("motifs"), list) else None,
            "field_entries": sum(row.get("representation") == "field" for row in index) if isinstance(index, list) else None,
            "field_exclusions": ["flat_grafting", "flat_macros", "live_flat_module_pool", "grammar_induction", "motif_claims"],
        }

    start_manifest = _read_optional_json(run_dir / "run_manifest.json")
    start_size = 0
    manifest_library = start_manifest.get("library_start") if isinstance(start_manifest, dict) else None
    if isinstance(manifest_library, dict) and manifest_library.get("entry_count") is not None:
        start_size = int(manifest_library["entry_count"])
    end_size = int(summary.get("library_size", 0) or 0)
    if not isinstance(manifest_library, dict) or manifest_library.get("entry_count") is None:
        # Legacy inference: exact count only. New runs use the immutable content identity above.
        if tasks:
            first_size = int(tasks[0].get("library_size", 0) or 0)
            first_admissions = len(tasks[0].get("new_library_keys", [])) if isinstance(tasks[0].get("new_library_keys", []), list) else 0
            start_size = max(0, first_size - first_admissions)
    task_pool = summary.get("task_pool") if isinstance(summary.get("task_pool"), dict) else {}
    tasks_to_run = int(summary.get("tasks_to_run", len(tasks)) or 0)
    unique_pool_references = int(task_pool.get("unique_references", task_pool.get("entries", 0)) or 0)
    revisit_slots = int(task_pool.get("revisit_slots", max(0, tasks_to_run - unique_pool_references)) or 0)
    repeated_attempts = len(tasks) - len(first_occurrences)
    counters = summary.get("counters") if isinstance(summary.get("counters"), dict) else {}
    routed_rows = [row for row in tasks if isinstance(row.get("strategy_metrics"), dict) and _finite(row["strategy_metrics"].get("router_score")) is not None]
    distillation_gaps = [value for row in routed_rows if (value := _finite((row.get("strategy_metrics") or {}).get("distillation_gap"))) is not None]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_schema_version": summary.get("schema_version"),
        "provenance": {
            "run_dir": str(run_dir),
            "files": provenance_files,
            "library": library_metadata,
            "dataset": summary.get("dataset_provenance"),
            "task_pool": task_pool,
            "start_manifest": start_manifest,
        },
        "run": {
            "status": summary.get("status", "unknown"),
            "seed": summary.get("seed"),
            "tasks_attempted": int(summary.get("tasks_attempted", len(tasks)) or 0),
            "tasks_to_run": tasks_to_run,
            "unique_tasks_attempted": len(first_occurrences),
            "repeated_attempts": repeated_attempts,
            "unique_pool_references": unique_pool_references,
            "revisit_slots": revisit_slots,
            "generations": int(summary.get("generations_run", summary.get("total_generations", 0)) or 0),
            "seconds": sum(_finite(row.get("seconds")) or 0.0 for row in tasks),
            "search_metric": summary.get("search_metric", "metric"),
            "report_metric": summary.get("report_metric", "report_metric"),
        },
        "quality": {
            "held_out_query_count": len(held),
            "held_out_query_coverage": len(held) / len(tasks) if tasks else None,
            "held_out_accuracy_mean": _mean(held),
            "held_out_accuracy_max": max(held) if held else None,
            "support_count": len(support),
            "support_accuracy_mean": _mean(support),
            "support_accuracy_max": max(support) if support else None,
            "support_query_gap_mean": _mean(gaps),
            "first_occurrence": {
                "task_count": len(first_occurrences),
                "held_out_query_count": len(first_held),
                "held_out_accuracy_mean": _mean(first_held),
                "held_out_accuracy_max": max(first_held) if first_held else None,
                "support_count": len(first_support),
                "support_accuracy_mean": _mean(first_support),
                "support_accuracy_max": max(first_support) if first_support else None,
                "support_query_gap_mean": _mean(first_gaps),
            },
        },
        "outcomes": dict(sorted(outcome_counts.items())),
        "strategies": {
            # Compatibility aliases retain the pre-attribution report surface; both mean selected.
            "task_usage": dict(sorted(strategy_counts.items())),
            "representations": dict(sorted(representation_counts.items())),
            "selected_task_usage": dict(sorted(strategy_counts.items())),
            "selected_representations": dict(sorted(representation_counts.items())),
            "held_out_task_usage": dict(sorted(report_strategy_counts.items())),
            "held_out_representations": dict(sorted(report_representation_counts.items())),
            "stage_seconds": dict(sorted(stage_seconds.items())),
            "metrics": {name: {"count": len(values), "mean": _mean(values), "max": max(values)} for name, values in sorted(strategy_metrics.items())},
        },
        "behavior": {
            "decomposition_count": sum(bool(row.get("decompose_op")) for row in tasks),
            "refinement_count": int(outcome_counts.get("refined", 0)),
            "library_hit_count": int(outcome_counts.get("library_hit", 0)),
            "deadline_count": sum(row.get("failure_stage") == "time_budget" for row in tasks),
            "cross_resolution_reuse_count": sum(float((row.get("task_metrics") or {}).get("cross_resolution_reuse", 0.0)) > 0 for row in tasks),
            "routed_attempt_count": len(routed_rows),
            "routed_distilled_count": int(counters.get("routed_solved", 0) or 0),
            "routed_undistillable_count": int(counters.get("routed_undistillable", 0) or 0),
            "routed_distillation_gap_mean": _mean(distillation_gaps),
        },
        "resources": {
            "events": dict(sorted(resource_events.items())),
            "counters": counters,
            "dataset": {
                "pool_load_seconds": _finite((summary.get("task_pool") or {}).get("load_seconds")) if isinstance(summary.get("task_pool"), dict) else None,
                "task_load_seconds": sum(_finite(row.get("task_load_seconds")) or 0.0 for row in tasks),
            },
        },
        "storage": {
            "library_start_entries": start_size,
            "library_peak_entries": max((int(row.get("library_size", 0) or 0) for row in tasks), default=end_size),
            "library_end_entries": end_size,
            "admissions": sum(len(row.get("new_library_keys", [])) for row in tasks if isinstance(row.get("new_library_keys", []), list)),
            "gc_removed": int(summary.get("gc_removed", 0) or 0),
        },
        "rungs": _rung_rows(tasks, summary.get("rungs", [])),
        "limitations": [
            "Missing held-out values are excluded from aggregates and rendered as N/A; valid zeroes are retained.",
            "The report is observational and cannot establish causality or matched-baseline superiority.",
            "Live end-state library metadata is index-level; the immutable starting identity hashes payload files without decoding them.",
            "Attempt-weighted aggregates include configured revisits; first-occurrence aggregates count each rung/task identity once.",
        ],
    }


def _display(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _csv_text(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=RUNG_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _display(row.get(key)) if key not in ("summary", "winning_task") else (row.get(key) or "N/A") for key in RUNG_FIELDS})
    return output.getvalue()


def _markdown(report: dict[str, Any]) -> str:
    run, quality = report["run"], report["quality"]
    lines = [
        "# Run report",
        "",
        "## Provenance",
        "",
        (
            f"This report was derived from `{report['provenance']['run_dir']}` using report schema {report['schema_version']}. "
            "It reads only the durable summary, pinned configs and task manifest, and optional library metadata."
        ),
        "",
    ]
    lines.extend(f"- `{item['path']}`: `{item['sha256']}` ({item['size']:,} bytes)" for item in report["provenance"]["files"])
    dataset = report["provenance"].get("dataset")
    if isinstance(dataset, dict):
        revision = dataset.get("revision") or "local"
        selection = dataset.get("selection_algorithm") or "unspecified"
        lines.append(f"- Dataset `{dataset.get('source', 'unknown')}` at revision `{revision}`; selection `{selection}`.")
    start_manifest = report["provenance"].get("start_manifest")
    if isinstance(start_manifest, dict):
        code = start_manifest.get("code") or {}
        library_start = start_manifest.get("library_start") or {}
        dirty = code.get("git_dirty")
        dirty_label = "unknown worktree state" if dirty is None else ("dirty worktree" if dirty else "clean worktree")
        lines.append(f"- Code commit `{code.get('git_commit') or 'unavailable'}`; {dirty_label}.")
        if library_start.get("entry_count") is not None:
            lines.append(f"- Starting library: {int(library_start['entry_count'])} entries; content hash `{library_start.get('content_sha256', 'unavailable')}`.")
        elif library_start.get("reason"):
            lines.append(f"- Starting library identity unavailable: {library_start['reason']}.")
    first = quality["first_occurrence"]
    lines.extend(
        [
            "",
            "## Executive summary",
            "",
            (
                f"Status `{run['status']}`; {run['tasks_attempted']}/{run['tasks_to_run']} tasks; {run['generations']:,} generations; "
                f"{_display(run['seconds'], 1)} recorded task-seconds. Held-out query accuracy covered {quality['held_out_query_count']} tasks "
                f"(mean {_display(quality['held_out_accuracy_mean'])}, maximum {_display(quality['held_out_accuracy_max'])}); support accuracy is reported "
                f"separately (mean {_display(quality['support_accuracy_mean'])}, maximum {_display(quality['support_accuracy_max'])}). "
                f"The pool contains {run['unique_pool_references']} unique reference(s); {run['repeated_attempts']} completed attempt(s) are revisits."
            ),
            (
                f"First-occurrence-only quality covers {first['held_out_query_count']}/{first['task_count']} query evaluations "
                f"(mean {_display(first['held_out_accuracy_mean'])}, maximum {_display(first['held_out_accuracy_max'])})."
            ),
            "",
            "## Per-rung results",
            "",
            "| Rung | Tasks | Query coverage | Held-out max | Held-out mean | Winning task | Support max | Support mean | Accepted | Failed | Admissions | Seconds |",
            "|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["rungs"]:
        coverage = "N/A" if row["query_coverage"] is None else f"{row['query_count']}/{row['task_count']} ({100 * row['query_coverage']:.1f}%)"
        cells = (
            row["rung"],
            row["task_count"],
            coverage,
            _display(row["highest_held_out_accuracy"]),
            _display(row["mean_held_out_accuracy"]),
            row["winning_task"] or "N/A",
            _display(row["support_maximum"]),
            _display(row["support_mean"]),
            row["accepted_outcomes"],
            row["failures"],
            row["admissions"],
            _display(row["seconds"], 1),
        )
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    lines.extend(["", "### One-sentence rung summaries", ""])
    lines.extend(f"- {row['summary']}" for row in report["rungs"])
    lines.extend(
        [
            "",
            "## Generalization gaps",
            "",
            (
                f"Across tasks with both measurements, mean support minus held-out query accuracy was {_display(quality['support_query_gap_mean'])}. "
                "Held-out query accuracy is the primary reported outcome; support accuracy measures search-time fitting and is not a substitute."
            ),
            "",
            "## Strategy usage",
            "",
            "Selected/admission paths:",
            "",
        ]
    )
    lines.extend(f"- `{name}`: {count} task record(s)" for name, count in report["strategies"]["selected_task_usage"].items())
    lines.extend(["", "Held-out evaluated paths:", ""])
    lines.extend(f"- `{name}`: {count} held-out evaluation(s)" for name, count in report["strategies"]["held_out_task_usage"].items())
    behavior = report["behavior"]
    lines.extend(
        [
            "",
            "## Decomposition, refinement, and library behavior",
            "",
            (
                f"The ledger records {behavior['decomposition_count']} decomposition-marked task(s), {behavior['refinement_count']} refinement outcome(s), "
                f"and {behavior['library_hit_count']} library-hit outcome(s). Library size moved from {report['storage']['library_start_entries']} to "
                f"{report['storage']['library_end_entries']} entries, peaked at {report['storage']['library_peak_entries']}, admitted "
                f"{report['storage']['admissions']} entries, and removed {report['storage']['gc_removed']} during GC."
            ),
            (
                f"Routing was exercised on {behavior['routed_attempt_count']} task(s): {behavior['routed_distilled_count']} distilled result(s), "
                f"{behavior['routed_undistillable_count']} undistillable result(s), and mean recorded distillation gap "
                f"{_display(behavior['routed_distillation_gap_mean'])}."
            ),
            "",
            "## Timing and resource events",
            "",
            f"Recorded task time totals {_display(run['seconds'], 1)} seconds, including {behavior['deadline_count']} deadline-marked task(s).",
            "",
        ]
    )
    lines.extend(f"- `{name}`: {_display(seconds, 1)} seconds" for name, seconds in report["strategies"]["stage_seconds"].items())
    dataset_timing = report["resources"].get("dataset", {})
    if dataset_timing.get("pool_load_seconds") is not None:
        lines.append(f"- `task_pool_discovery`: {_display(dataset_timing['pool_load_seconds'], 1)} seconds")
    if dataset_timing.get("task_load_seconds"):
        lines.append(f"- `task_materialization`: {_display(dataset_timing['task_load_seconds'], 1)} seconds")
    if report["resources"]["events"]:
        lines.extend(["", "Resource metric events:", ""])
        lines.extend(f"- `{name}`: {count}" for name, count in report["resources"]["events"].items())
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_run_report(run_dir: Path | str, library: Path | str | None = None) -> dict[str, Any]:
    directory = Path(run_dir)
    report = build_run_report(directory, Path(library) if library is not None else None)
    _atomic_write(directory / "rung_summary.csv", _csv_text(report["rungs"]).encode())
    _atomic_write(directory / "run_report.json", (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    _atomic_write(directory / "run_report.md", _markdown(report).encode())
    return report
