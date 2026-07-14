"""Transient Rich status for the task currently running.

The footer answers only the liveness question: which task and conceptual stage are active, how
long the task has run, and the best literal support accuracy seen so far. Durable results are
printed separately and written to JSON.
"""

import time
from typing import Any

from rich.console import Console, Group
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text


class _Footer:
    def __init__(self, board: "StatusBoard") -> None:
        self.board = board

    def __rich_console__(self, console: Console, options: Any):
        board = self.board
        if board.progress is None:
            yield Text("waiting for first task", style="dim")
            return
        lines: list[Any] = [board.progress]
        if board.stage_line:
            stage = Text("  ")
            stage.append(board.stage_line, style="bold magenta")
            if board.best_metric is not None:
                stage.append(f"  best support {board.best_metric:.3f}", style="bold bright_cyan")
            lines.append(stage)
        if board.event_line:
            lines.append(Text(f"  {board.event_line}", style="dim"))
        yield Group(*lines)


class StatusBoard:
    """State and Live region for the current task; all hooks are safe while disabled."""

    def __init__(self) -> None:
        self._live: Any = None
        self.progress: Progress | None = None
        self._progress_task: int | None = None
        self.task_line = ""
        self.stage_line = ""
        self.event_line = ""
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

    def task(self, cursor: int, total: int, rung: int, name: str) -> None:
        if self._live is None:
            return
        self.task_line = f"task {cursor}/{total} · rung {rung} · {name}"
        self.stage_line = ""
        self.event_line = ""
        self.clock_started = None
        self.clock_budget = None
        self.best_metric = None
        self.active_stage = None
        self.progress = Progress(
            SpinnerColumn(style="bright_cyan"),
            TextColumn("[bold cyan]{task.description}"),
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

    def stage(self, name: str, detail: str = "") -> None:
        if self._live is None:
            return
        self.active_stage = name
        self.stage_line = f"{name}  {detail}".rstrip()

    def generation(self, strategy: str, generation: int, best_fitness: float, best_metric: float, mean_fitness: float) -> None:
        if self._live is None:
            return
        self.best_metric = best_metric if self.best_metric is None else max(self.best_metric, best_metric)
        self.active_stage = strategy
        self.stage_line = f"{strategy} · generation {generation} · current support {best_metric:.3f}"

    def event(self, text: str) -> None:
        if self._live is None:
            return
        self.event_line = text


BOARD = StatusBoard()
