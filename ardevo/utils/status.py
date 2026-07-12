"""A pinned live status footer for long runs: 1-3 rows at the bottom of the terminal showing what
the run is doing RIGHT NOW (task, budget clock, strategy generation, last event), while ordinary
console/log output keeps scrolling above it untouched.

The inner loop is silent by design (a single ARC-width generation can run for minutes with zero
output), so the footer is the liveness signal: its clock ticks on Rich's refresh thread even when
no new generation lands, which is exactly the "is it hung or just wide" question. Everything is a
no-op unless `enable()` ran, so tests, workers, piped output, and `[run] live_status = false` are
byte-identical to the pre-footer behavior. Hooks only ever run in the MAIN process (the assess pool
uses spawn and never imports this hot); the footer is deliberately generation-granular."""

import time
from typing import Any

from rich.console import Console

_BAR_WIDTH = 24


class _Footer:
    """The renderable: rebuilt from board state at every Live refresh, so the elapsed clock and
    budget bar advance between updates without any hook firing."""

    def __init__(self, board: "StatusBoard") -> None:
        self.board = board

    def __rich_console__(self, console: Console, options: Any):
        from rich.text import Text

        board = self.board
        clock = ""
        if board.clock_started is not None:
            elapsed = time.perf_counter() - board.clock_started
            if board.clock_budget:
                filled = min(int(_BAR_WIDTH * elapsed / board.clock_budget), _BAR_WIDTH)
                color = "green" if elapsed < 0.8 * board.clock_budget else "yellow"
                clock = f"  [{color}]{'█' * filled}{'─' * (_BAR_WIDTH - filled)}[/{color}] {elapsed:.0f}/{board.clock_budget:.0f}s"
            else:
                clock = f"  {elapsed:.0f}s"
        yield Text.from_markup(f"[bold cyan]{board.task_line}[/bold cyan]{clock}" if board.task_line else "[dim]waiting for first task[/dim]")
        if board.stage_line:
            yield Text.from_markup(f"  [magenta]{board.stage_line}[/magenta]")
        if board.event_line:
            yield Text.from_markup(f"  [dim]{board.event_line}[/dim]")


class StatusBoard:
    """The footer state + its Live region. One instance (`BOARD` below) serves the whole process."""

    def __init__(self) -> None:
        self._live: Any = None
        self.task_line = ""
        self.stage_line = ""
        self.event_line = ""
        self.clock_started: float | None = None
        self.clock_budget: float | None = None

    @property
    def enabled(self) -> bool:
        return self._live is not None

    def enable(self, console: Console) -> None:
        """Start the pinned region. Refuses quietly off-terminal (piped output, ClearML agents, CI):
        a Live region on a non-TTY degrades to garbage frames in the capture."""
        if self._live is not None or not console.is_terminal:
            return
        from rich.live import Live

        self._live = Live(_Footer(self), console=console, refresh_per_second=4, transient=True)
        self._live.start()

    def close(self) -> None:
        """Stop and clear the footer (transient: it vanishes from scrollback). Idempotent and safe
        mid-crash: the run's summaries must land even if the terminal is already wedged."""
        live, self._live = self._live, None
        if live is not None:
            try:
                live.stop()
            except Exception:
                pass

    # --- hooks (all no-ops when disabled) ---------------------------------------------------------

    def task(self, cursor: int, total: int, rung: int, name: str) -> None:
        if self._live is None:
            return
        self.task_line = f"task {cursor}/{total}  rung {rung}  {name}"
        self.stage_line = ""
        self.clock_started = None
        self.clock_budget = None

    def clock(self, budget_seconds: float | None) -> None:
        """Arm the per-attempt clock/bar; called at every solve() entry, so recursion re-arms it."""
        if self._live is None:
            return
        self.clock_started = time.perf_counter()
        self.clock_budget = budget_seconds if budget_seconds and budget_seconds > 0 else None

    def stage(self, name: str, detail: str = "") -> None:
        if self._live is None:
            return
        self.stage_line = f"{name}  {detail}".rstrip()

    def generation(self, strategy: str, generation: int, best_fitness: float, best_metric: float, mean_fitness: float) -> None:
        if self._live is None:
            return
        self.stage_line = f"{strategy}  gen {generation}  best fitness {best_fitness:.3f}  metric {best_metric:.3f}  mean {mean_fitness:.3f}"

    def event(self, text: str) -> None:
        if self._live is None:
            return
        self.event_line = f"{time.strftime('%H:%M:%S')}  {text}"


BOARD = StatusBoard()
