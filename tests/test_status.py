"""The live status footer: byte-identical no-op when disabled, pinned-region state when enabled."""

import io

import pytest
from rich.console import Console

from versal.utils.status import BOARD, StatusBoard, _Footer


def _terminal_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=True, width=120)


def test_disabled_board_hooks_are_noops() -> None:
    board = StatusBoard()
    board.task(1, 10, 3, "two_spirals")
    board.clock(600.0)
    board.stage("direct", "starting")
    board.generation("direct", 4, 0.5, 0.6, 0.4)
    board.event("solved")
    board.close()  # idempotent, never enabled
    assert not board.enabled
    assert board.task_line == "" and board.stage_line == "" and board.event_line == ""


def test_enable_refuses_non_terminal() -> None:
    board = StatusBoard()
    board.enable(Console(file=io.StringIO()))  # not a TTY: piped output, agents, CI
    assert not board.enabled


def test_enabled_board_tracks_state_and_renders() -> None:
    board = StatusBoard()
    console = _terminal_console()
    board.enable(console)
    try:
        assert board.enabled
        board.task(2, 18, 18, "arc.train.31aa019c")
        board.clock(600.0)
        board.stage("routed", "starting (budget 10 gens)")
        board.generation("direct", 3, 0.412, 0.63, 0.35)
        board.generation("direct", 4, 0.398, 0.51, 0.36)  # a worse generation must not lower the best
        assert "direct · generation 4" in board.stage_line and "support 0.5100" in board.stage_line
        board.event("stone shelved")
        probe = _terminal_console()  # render through a second console so the Live region is untouched
        with probe.capture() as capture:
            probe.print(_Footer(board))
        text = capture.get()
        assert "task 2/18" in text and "arc.train.31aa019c" in text
        assert "direct · generation 4" not in text  # the newest event replaces the prior activity
        assert "best support" in text and "0.6300" in text
        assert "stone shelved" in text
        assert "/600s" not in text  # configured deadlines stay in the config, not the runtime display
    finally:
        board.close()
    assert not board.enabled
    board.close()  # idempotent after close


def test_task_reset_clears_stage_and_clock() -> None:
    board = StatusBoard()
    board.enable(_terminal_console())
    try:
        board.clock(60.0)
        board.stage("direct")
        board.generation("direct", 1, 0.2, 0.9, 0.1)
        board.task(5, 10, 6, "mnist")
        assert board.stage_line == "" and board.clock_started is None and board.clock_budget is None
        assert board.best_metric is None  # the best-accuracy readout is per task
    finally:
        board.close()


def test_module_board_singleton_defaults_disabled() -> None:
    assert isinstance(BOARD, StatusBoard)
    assert not BOARD.enabled  # importing modules that hook it must never start a Live region


@pytest.mark.parametrize("width", [40, 80, 160])
def test_footer_keeps_support_and_activity_in_fixed_columns_without_wrapping(width: int) -> None:
    board = StatusBoard()
    board.enable(_terminal_console())
    try:
        board.task(1, 100, 1, "xor")
        snapshots = []
        for detail in ("short", "a very long validation explanation " * 8):
            board.stage("validation", detail)
            probe = Console(file=io.StringIO(), width=width, color_system=None)
            with probe.capture() as capture:
                probe.print(_Footer(board))
            snapshots.append(capture.get().splitlines()[-1])
        assert all(line.startswith("  best support —        → validation") for line in snapshots)
        assert all(len(line) <= width for line in snapshots)
        assert snapshots[1].endswith("…")
        board.generation("direct", 1, 0.0, 0.0, 0.0)
        probe = Console(file=io.StringIO(), width=width, color_system=None)
        with probe.capture() as capture:
            probe.print(_Footer(board))
        assert capture.get().splitlines()[-1].startswith("  best support 0.0000   → direct")
    finally:
        board.close()


def test_shared_generation_context_survives_inner_progress_and_uses_highlights() -> None:
    board = StatusBoard()
    board.enable(_terminal_console())
    try:
        board.task(1, 10, 1, "xor")
        board.stage("Evolve dense network", phase="refine", shared_generation=4)
        board.generation("Evolve dense network", 37, 1.0, 0.75, 0.8)
        assert board.stage_line == "→ Evolve dense network · refine · shared generation 4 · support 0.7500"
        assert board.activity.style == "magenta"
        emphasized = [board.activity.plain[span.start : span.end] for span in board.activity.spans if span.style == "bold yellow"]
        assert {"refine", "4", "0.7500"} <= set(emphasized)
        board.event("Escape pressed · stopping at the next safe boundary")
        assert board.activity.plain.startswith("Escape pressed")
        assert board.best_metric == 0.75
    finally:
        board.close()
