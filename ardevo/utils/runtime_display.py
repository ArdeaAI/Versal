"""Human-facing Rich output for orchestrated runs.

JSON remains the exhaustive diagnostic record. This module renders the smaller operational story:
what the system tried, what each completed stage established, and the two literal accuracy rails.
"""

from collections import Counter
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from ardevo.utils.status import BOARD

STAGES: dict[str, tuple[str, str]] = {
    "lookup": ("Reuse", "test compatible learned solutions"),
    "refine": ("Refine", "improve a retrieved solution without regression"),
    "routed": ("Route experts", "combine frozen experts and distill an executable pathway"),
    "grammar": ("Apply grammar", "synthesize from independently recurring structures"),
    "direct": ("Evolve network", "grow and train a task-specific topology"),
    "composition": ("Compose modules", "wire reusable modules with trainable glue"),
    "decompose": ("Decompose", "split the task into valid subtasks and solve them recursively"),
    "decompose_first": ("Decompose first", "split an oversized task before attempting a flat network"),
    "query": ("Held-out query", "score the support-selected executable champion once"),
    "persist": ("Persist result", "retain a solution or useful stepping stone for later tasks"),
    "time_budget": ("Deadline", "stop work cleanly when the task allowance is exhausted"),
}

STATUS_REASONS = {
    "evaluated": "evaluated",
    "no_executable_champion": "no executable champion was available",
    "evaluation_unavailable": "evaluation did not produce a valid measurement",
    "query_split_unavailable": "the task has no usable held-out query split",
    "time_limit_before_evaluation": "the deadline arrived before held-out evaluation",
    "not_reached": "the stage was not reached",
    "task_crashed": "the task was interrupted before evaluation completed",
    "legacy_missing": "not recorded by this run schema",
}


def _duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(round(seconds), 60)
    return f"{minutes}m {remainder:02d}s"


def _accuracy(value: float | None, status: str, style: str) -> Text:
    if value is not None:
        return Text(f"{value:.4f}", style=style)
    reason = STATUS_REASONS.get(status, status.replace("_", " "))
    return Text(f"N/A — {reason}", style="bold yellow")


class RuntimeDisplay:
    """Rich renderer used by the trial and orchestrator."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._active_stage: str | None = None
        self._provisional_support: float | None = None

    def enable(self, live_status: bool = True) -> None:
        if live_status:
            BOARD.enable(self.console)

    def close(self) -> None:
        BOARD.close()

    @property
    def active_stage(self) -> str | None:
        return self._active_stage

    @property
    def provisional_support(self) -> float | None:
        return self._provisional_support

    def task_started(self, cursor: int, total: int, rung: int, name: str) -> None:
        self._active_stage = None
        self._provisional_support = None
        BOARD.task(cursor, total, rung, name)
        self.console.print(Rule(Text(f"Task {cursor}/{total} · rung {rung} · {name}", style="bold cyan"), style="cyan"))

    def stage_started(self, stage: str, *, detail: str = "") -> None:
        label, concept = STAGES.get(stage, (stage.replace("_", " ").title(), detail))
        self._active_stage = label
        BOARD.stage(label, detail or concept)

    def generation(self, strategy: str, generation: int, best_fitness: float, support_accuracy: float, mean_fitness: float, *, depth: int = 0) -> None:
        label = STAGES.get(strategy, (strategy, ""))[0]
        self._active_stage = label
        if depth == 0:
            self._provisional_support = support_accuracy if self._provisional_support is None else max(self._provisional_support, support_accuracy)
        BOARD.generation(label, generation, best_fitness, support_accuracy, mean_fitness)

    def stage_result(
        self,
        stage: str,
        outcome: str,
        detail: str,
        *,
        seconds: float | None = None,
        depth: int = 0,
        support_accuracy: float | None = None,
    ) -> None:
        label, concept = STAGES.get(stage, (stage.replace("_", " ").title(), ""))
        self._active_stage = label
        if support_accuracy is not None and depth == 0:
            self._provisional_support = support_accuracy if self._provisional_support is None else max(self._provisional_support, support_accuracy)
        style, symbol = {
            "accepted": ("green", "✓"),
            "hit": ("green", "✓"),
            "saved": ("green", "✓"),
            "continue": ("yellow", "→"),
            "miss": ("dim", "·"),
            "skipped": ("dim", "·"),
            "failed": ("red", "×"),
            "unavailable": ("yellow", "!"),
        }.get(outcome, ("cyan", "•"))
        row = Table.grid(padding=(0, 1))
        row.add_column(width=2 + depth * 2)
        row.add_column(min_width=17, style="bold")
        row.add_column(ratio=1)
        row.add_column(justify="right", style="dim")
        prefix = f"{'  ' * depth}{symbol}"
        score_label = "support" if depth == 0 else "subtask support"
        score = f" · {score_label} {support_accuracy:.4f}" if support_accuracy is not None else ""
        timing = _duration(seconds) if seconds is not None else ""
        row.add_row(Text(prefix, style=style), Text(label, style=style), Text(f"{detail or concept}{score}"), timing)
        self.console.print(row)

    def query_result(self, value: float | None, status: str, *, seconds: float | None = None, depth: int = 0) -> None:
        if value is not None:
            self.stage_result("query", "accepted", f"query accuracy {value:.4f}", seconds=seconds, depth=depth)
        else:
            self.stage_result("query", "unavailable", STATUS_REASONS.get(status, status.replace("_", " ")), seconds=seconds, depth=depth)

    def task_finished(
        self,
        cursor: int,
        total: int,
        rung: int,
        name: str,
        attempt: Any,
        *,
        solved: bool,
        task_seconds: float,
        new_library_keys: list[str],
        library_size: int,
    ) -> None:
        outcome = getattr(attempt, "outcome", "unknown")
        failure_stage = getattr(attempt, "failure_stage", None)
        if solved:
            reason = "accepted executable champion"
        elif failure_stage == "time_budget":
            reason = "task deadline reached"
        elif failure_stage == "parent_re_evolve":
            reason = "subtasks solved, but the recomposed parent stayed below threshold"
        elif isinstance(failure_stage, str) and failure_stage.startswith("subtask:"):
            reason = f"required subtask failed: {failure_stage.removeprefix('subtask:')}"
        elif getattr(attempt, "support_status", None) == "no_executable_champion":
            reason = "no strategy produced an executable champion"
        else:
            reason = "best executable champion remained below the acceptance threshold"

        grid = Table.grid(padding=(0, 2))
        grid.add_column(min_width=25)
        grid.add_column(ratio=1)
        outcome_style = "bold green" if solved else "bold red"
        grid.add_row("Outcome", Text(f"{outcome} — {reason}", style=outcome_style))
        grid.add_row(
            Text("BEST SUPPORT ACCURACY", style="bold bright_cyan"),
            _accuracy(getattr(attempt, "support_accuracy", None), getattr(attempt, "support_status", "legacy_missing"), "bold bright_cyan"),
        )
        grid.add_row(
            Text("HELD-OUT QUERY ACCURACY", style="bold bright_magenta"),
            _accuracy(getattr(attempt, "query_accuracy", None), getattr(attempt, "query_status", "legacy_missing"), "bold bright_magenta"),
        )
        diagnostic = getattr(attempt, "diagnostic_observation", None) or {}
        if diagnostic and getattr(attempt, "support_accuracy", None) is None:
            diagnostic_label = "subtask" if diagnostic.get("executable") else "router"
            grid.add_row(
                "Best diagnostic",
                f"{float(diagnostic['score']):.4f} {diagnostic.get('metric')} · {diagnostic_label} {diagnostic.get('task')} at depth {diagnostic.get('depth')}",
            )
        strategy = getattr(attempt, "strategy", None) or "none"
        if strategy == "time_budget":
            grid.add_row("Selected path", "none — no executable parent champion")
            grid.add_row("Stopped during", "task deadline")
        else:
            grid.add_row("Selected path", f"{STAGES.get(strategy, (strategy, ''))[0]} · {getattr(attempt, 'generations', 0)} generations")
        if solved and new_library_keys:
            persistence = f"saved {len(new_library_keys)} new entr{'y' if len(new_library_keys) == 1 else 'ies'}"
        elif not solved and getattr(attempt, "library_key", None):
            persistence = "saved a below-threshold stepping stone"
        elif solved:
            persistence = "solved for this run; no new library entry"
        else:
            persistence = "nothing retained"
        grid.add_row("Persistence", f"{persistence} · library size {library_size}")

        timing = Tree(Text(f"Timing · {_duration(task_seconds)} total", style="bold"), guide_style="dim")
        for stage, seconds in (getattr(attempt, "stage_seconds", None) or {}).items():
            label = STAGES.get(stage, (stage.replace("_", " ").title(), ""))[0]
            inclusive = " (includes recursive work)" if stage in {"decompose", "decompose_first"} else ""
            timing.add(f"{label}: {_duration(float(seconds))}{inclusive}")
        body = Group(grid, Text(), timing)
        border = "green" if solved else "red"
        self.console.print(Panel(body, title=f"Task {cursor}/{total} · rung {rung} · {name}", border_style=border, padding=(1, 2)))

    def task_interrupted(self, name: str, error: BaseException, *, support_accuracy: float | None, active_stage: str | None, elapsed: float) -> None:
        grid = Table.grid(padding=(0, 2))
        grid.add_row("Active stage", active_stage or "task setup")
        grid.add_row(Text("BEST SUPPORT ACCURACY", style="bold bright_cyan"), _accuracy(support_accuracy, "task_crashed", "bold bright_cyan"))
        grid.add_row(Text("HELD-OUT QUERY ACCURACY", style="bold bright_magenta"), _accuracy(None, "task_crashed", "bold bright_magenta"))
        grid.add_row("Interruption", f"{type(error).__name__}: {error}")
        grid.add_row("Elapsed", _duration(elapsed))
        self.console.print(Panel(grid, title=f"Interrupted · {name}", border_style="red"))

    def run_finished(self, task_records: list[dict[str, Any]], *, seconds: float, library_size: int) -> None:
        outcomes = Counter(record.get("outcome", "unknown") for record in task_records)
        table = Table.grid(padding=(0, 2))
        table.add_row("Tasks completed", str(len(task_records)))
        table.add_row("Outcomes", " · ".join(f"{name} {count}" for name, count in sorted(outcomes.items())) or "none")
        table.add_row("Elapsed", _duration(seconds))
        table.add_row("Library size", str(library_size))
        self.console.print(Panel(table, title="Run complete", border_style="green"))


class NullRuntimeDisplay:
    """Drop-in no-op for isolated unit construction."""

    active_stage = None
    provisional_support = None

    def __getattr__(self, _name: str):
        return lambda *args, **kwargs: None


NULL_DISPLAY = NullRuntimeDisplay()
