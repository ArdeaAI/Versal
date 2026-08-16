"""rung_doctor: answer "do these rungs even load, and what shapes are they?" WITHOUT a full run.

    uv run python -m versal.tools.rung_doctor --rungs 1-18 --n-tasks 2 --n-samples 50

Per rung it reports: load OK or the failure reason, the input/output signatures and widths of the
first task, whether the input carries a TIME axis (stepped substrate territory), and whether any
live library entry matches the io exactly or within a width tolerance. This is the diagnostic for
the silent-coverage failure mode (rung 5 never appeared in any phase-3 pool and nothing said so).
"""

import argparse
from typing import Any, Callable

from versal.evolution.multitask import build_pool_report
from versal.library import ModuleLibrary, task_io
from versal.utils.logging import Logger

console = Logger.get_console()


def parse_rungs(spec: str) -> list[int]:
    rungs: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            low, high = part.split("-", 1)
            rungs.extend(range(int(low), int(high) + 1))
        else:
            rungs.append(int(part))
    return rungs


def rung_report(
    rungs: list[int],
    *,
    source: str = "Ardea/Icarus-dataset",
    n_tasks: int = 2,
    n_samples: int = 50,
    support_fraction: float = 0.8,
    library: ModuleLibrary | None = None,
    width_tolerance: int = 4,
    dataset_factory: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """One dict per configured rung; pure so tests can drive it with a fake dataset factory."""
    rows: list[dict[str, Any]] = []
    for rung in rungs:
        report = build_pool_report(source, [rung], n_samples, support_fraction, n_tasks, True, 0, dataset_factory=dataset_factory)
        if report.skipped:
            skip = report.skipped[0]
            rows.append({"rung": rung, "status": f"FAIL:{skip.error_type}", "detail": skip.message})
            report.close()
            continue
        entry = report.entries[0]
        try:
            io = task_io(report.materialize(entry))
            temporal = "T" in io["inputs"][0]["signature"].split("|", 1)[-1].split(",")
            row: dict[str, Any] = {
                "rung": rung,
                "status": "OK",
                "tasks": len(report.entries),
                "example": entry.name,
                "input": f"{io['inputs'][0]['signature']} w={io['inputs'][0]['width']}",
                "output": f"{io['output']['signature']} w={io['output']['width']}",
                "temporal": temporal,
            }
            if library is not None:
                exact = library.query(input_signature=io["inputs"][0]["signature"], input_width=io["inputs"][0]["width"], output_width=io["output"]["width"])
                near = library.query(input_width=io["inputs"][0]["width"], output_width=io["output"]["width"], width_tolerance=width_tolerance)
                row["library"] = f"exact={len(exact)} near={len(near)}"
            rows.append(row)
        finally:
            report.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Icarus rung loadability and shapes without a run.")
    parser.add_argument("--source", default="Ardea/Icarus-dataset")
    parser.add_argument("--rungs", default="1-18", help='e.g. "1-5", "1,3,7", "1-18"')
    parser.add_argument("--n-tasks", type=int, default=2)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--library", default="library", help="library dir to match against (skipped if absent)")
    args = parser.parse_args()

    from pathlib import Path

    library = ModuleLibrary(args.library) if Path(args.library).exists() else None
    rows = rung_report(parse_rungs(args.rungs), source=args.source, n_tasks=args.n_tasks, n_samples=args.n_samples, library=library)

    from rich.table import Table

    table = Table(title=f"rung doctor: {args.source}")
    columns = ["rung", "status", "tasks", "example", "input", "output", "temporal", "library", "detail"]
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*(str(row.get(column, "")) for column in columns))
    console.print(table)


if __name__ == "__main__":
    main()
