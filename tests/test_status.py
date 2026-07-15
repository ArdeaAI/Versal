"""The live status footer: byte-identical no-op when disabled, pinned-region state when enabled."""

import io

from rich.console import Console

from ardevo.utils.status import BOARD, StatusBoard, _Footer


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
        board.event("stone shelved")
        probe = _terminal_console()  # render through a second console so the Live region is untouched
        with probe.capture() as capture:
            probe.print(_Footer(board))
        text = capture.get()
        assert "task 2/18" in text and "arc.train.31aa019c" in text
        assert "direct · generation 4" in text and "current support 0.510" in text
        assert "best support 0.630" in text  # the running per-task maximum, not the current generation's
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
