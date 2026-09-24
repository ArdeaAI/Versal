"""
Transient Rich status for the task currently running.

The footer answers only the liveness question: which task and conceptual stage are active, how
long the task has run, and the best literal support accuracy seen so far. Durable results are
printed separately and written to JSON.
"""

import time
from typing import Any

from rich.console import Console, Group
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Column
from rich.text import Text


def activity_text(message: str) -> Text:
    """
    Style the current activity consistently without interpreting user text as markup.
    """
    rendered = Text(message, style="magenta", no_wrap=True, overflow="ellipsis")
    rendered.highlight_regex(r"\b(?:solve|refine|exhaustive|\d+(?:\.\d+)?(?:ms|s|m)?)\b", "bold yellow")
    if message[:1] in {"✓", "→", "!", "×"}:
        rendered.stylize("bold yellow", 0, 1)
    return rendered


class _Footer:
    def __init__(self, board: "StatusBoard") -> None:
        self.board = board

    def __rich_console__(self, console: Console, options: Any):
        board = self.board
        if board.progress is None:
            yield Text("waiting for first task", style="dim")
            return
        lines: list[Any] = [board.progress]
        stage = Text("  best support ", style="magenta", no_wrap=True, overflow="ellipsis")
        score = f"{board.best_metric:.4f}" if board.best_metric is not None else "—"
        stage.append(score.ljust(6), style="bold yellow")
        stage.append("   ")
        stage.append(board.activity)
        stage.truncate(options.max_width, overflow="ellipsis")
        lines.append(stage)
        yield Group(*lines)


class StatusBoard:
    """
    State and Live region for the current task; all hooks are safe while disabled.
    """

    def __init__(self) -> None:
        self._live: Any = None
        self.progress: Progress | None = None
        self._progress_task: int | None = None
        self.task_line = ""
        self.stage_line = ""
        self.event_line = ""
        self.activity = Text()
        self._generation_detail = ""
        self.clock_started: float | None = None
        self.clock_budget: float | None = None  # retained for checkpoint-era callers; not displayed
        self.best_metric: float | None = None
        self.active_stage: str | None = None

    @property
    def enabled(self) -> bool:
        return self._live is not None

    def enable(self, console: Console) -> None:
        if self._live is not None or not console.is_terminal:
            return
        from rich.live import Live

        self._live = Live(_Footer(self), console=console, refresh_per_second=4, transient=True)
        self._live.start()

    def close(self) -> None:
        live, self._live = self._live, None
        if live is not None:
            try:
                live.stop()
            except Exception:
                pass
        self.progress = None
        self._progress_task = None
        self.activity = Text()
        self._generation_detail = ""

    def task(self, cursor: int, total: int, rung: int, name: str) -> None:
        if self._live is None:
            return
        self.task_line = f"task {cursor}/{total} · rung {rung} · {name}"
        self.stage_line = ""
        self.event_line = ""
        self.activity = Text()
        self._generation_detail = ""
        self.clock_started = None
        self.clock_budget = None
        self.best_metric = None
        self.active_stage = None
        self.progress = Progress(
            SpinnerColumn(style="bright_cyan"),
            TextColumn("[bold cyan]{task.description}", table_column=Column(no_wrap=True, overflow="ellipsis")),
            BarColumn(bar_width=None),
            TextColumn("{task.completed:.0f}/{task.total:.0f}"),
            TimeElapsedColumn(),
            auto_refresh=False,
            expand=True,
        )
        self._progress_task = self.progress.add_task(self.task_line, total=total, completed=max(0, cursor - 1))

    def clock(self, budget_seconds: float | None) -> None:
        # The configured deadline belongs in the config, not the console. Only elapsed time is shown.
        if self._live is None:
            return
        if self.clock_started is None:
            self.clock_started = time.perf_counter()
        self.clock_budget = None

    def message(self, name: str, rendered: Text, *, support_accuracy: float | None = None, depth: int = 0) -> None:
        """
        Replace the current activity and update only the parent task's support maximum.
        """
        if self._live is None:
            return
        self.active_stage = name
        self.activity = rendered.copy()
        self.stage_line = rendered.plain
        self.event_line = ""
        if support_accuracy is not None and depth == 0:
            self.best_metric = support_accuracy if self.best_metric is None else max(self.best_metric, support_accuracy)

    def stage(self, name: str, detail: str = "", *, phase: str | None = None, shared_generation: int | None = None) -> None:
        if self._live is None:
            return
        self._generation_detail = ((f"{phase} · " if phase else "") + f"shared generation {shared_generation}") if shared_generation is not None else ""
        context = self._generation_detail or detail
        self.message(name, activity_text(f"→ {name}" + (f" · {context}" if context else "")))

    def generation(self, strategy: str, generation: int, best_fitness: float, best_metric: float, mean_fitness: float, *, depth: int = 0) -> None:
        if self._live is None:
            return
        context = self._generation_detail or f"generation {generation}"
        score_label = "support" if depth == 0 else "subtask support"
        self.message(strategy, activity_text(f"→ {strategy} · {context} · {score_label} {best_metric:.4f}"), support_accuracy=best_metric, depth=depth)

    def event(self, text: str) -> None:
        if self._live is None:
            return
        self.message(self.active_stage or "", activity_text(text))
        self.event_line = text


BOARD = StatusBoard()
