"""The default console is concise, protocol-explicit, and visually separates quality rails."""

import io
import logging
import re
from types import SimpleNamespace

from rich.console import Console

from ardevo.utils.logging import Logger
from ardevo.utils.runtime_display import STAGES, RuntimeDisplay


def _render_display(width: int = 100) -> tuple[RuntimeDisplay, io.StringIO]:
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, color_system="standard", no_color=False, width=width)
    return RuntimeDisplay(console), stream


def _attempt(**overrides):
    values = {
        "outcome": "failed",
        "failure_stage": "time_budget",
        "support_accuracy": 0.0,
        "query_accuracy": None,
        "support_status": "evaluated",
        "query_status": "time_limit_before_evaluation",
        "strategy": "direct",
        "generations": 4,
        "library_key": None,
        "stage_seconds": {"routed": 1.25, "direct": 3.5},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_task_panel_preserves_zero_and_explains_missing_query() -> None:
    display, stream = _render_display()
    display.task_started(16, 18, 16, "deepsea.b2348")
    display.stage_result("lookup", "miss", "no compatible learned solution cleared the support gate", seconds=0.1)
    display.stage_result("direct", "continue", "4 generations", seconds=3.5, support_accuracy=0.0)
    display.task_finished(16, 18, 16, "deepsea.b2348", _attempt(), solved=False, task_seconds=4.8, new_library_keys=[], library_size=9)

    output = stream.getvalue()
    assert "Reuse" in output and "Evolve network" in output
    assert "BEST SUPPORT ACCURACY" in output and "0.0000" in output
    assert "HELD-OUT QUERY ACCURACY" in output and "N/A" in output
    assert "deadline arrived before held-out evaluation" in output
    assert "Timing" in output and "Route experts" in output
    assert "budget" not in output and "attempt: {" not in output and ".py:" not in output
    assert "\x1b[1;96m" in output  # bright cyan support rail
    assert "\x1b[1;95m" in output  # bright magenta query label


def test_valid_zero_query_is_not_rendered_as_missing_at_narrow_width() -> None:
    display, stream = _render_display(width=80)
    attempt = _attempt(outcome="evolved", failure_stage=None, query_accuracy=0.0, query_status="evaluated")
    display.task_finished(1, 1, 1, "xor", attempt, solved=True, task_seconds=0.2, new_library_keys=["m1"], library_size=1)
    output = stream.getvalue()
    assert output.count("0.0000") == 2
    assert "N/A" not in output
    assert "saved 1 new entry" in output


def test_recursive_diagnostic_never_becomes_parent_support() -> None:
    display, stream = _render_display()
    display.task_started(13, 18, 13, "darcy_flow.b3")
    display.stage_result("routed", "continue", "diagnostic only", depth=1, support_accuracy=0.618)
    attempt = _attempt(
        support_accuracy=None,
        support_status="no_executable_champion",
        diagnostic_observation={"score": 0.618, "metric": "support_accuracy", "task": "darcy_flow.b3.h0", "depth": 1, "strategy": "routed", "executable": True},
        strategy="time_budget",
        generations=0,
    )
    display.task_finished(13, 18, 13, "darcy_flow.b3", attempt, solved=False, task_seconds=640, new_library_keys=[], library_size=7)

    output = stream.getvalue()
    assert "subtask support 0.6180" in output
    assert "BEST SUPPORT ACCURACY" in output and "N/A" in output
    assert "Best diagnostic" in output and "darcy_flow.b3.h0" in output
    assert "none — no executable parent champion" in output


def test_task_panel_reports_lifecycle_changes_without_dumping_metrics() -> None:
    display, stream = _render_display(width=180)
    attempt = _attempt(strategy_metrics={"router_vertices_expired": 2.0, "router_vertices_revived": 1.0, "library_inactivity_retired": 3.0})
    display.task_finished(5, 18, 5, "mnist.b1", attempt, solved=False, task_seconds=2.0, new_library_keys=[], library_size=12)
    output = stream.getvalue()
    plain = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", output).split())
    assert "Overmind upkeep" in output
    assert "removed 2 dormant experts" in plain and "revived 1" in plain and "tombstoned 3 long-idle entries" in plain
    assert "router_vertices_expired" not in output


def test_graceful_shutdown_panel_explains_escape_and_missing_evaluation() -> None:
    display, stream = _render_display()
    attempt = _attempt(
        failure_stage="shutdown_requested",
        support_accuracy=None,
        support_status="not_reached",
        query_status="shutdown_before_evaluation",
        strategy="shutdown",
        generations=0,
    )
    display.task_finished(3, 18, 3, "two_spirals", attempt, solved=False, task_seconds=1.2, new_library_keys=[], library_size=2)
    display.run_finished([], seconds=1.2, library_size=2, status="stopped")

    output = stream.getvalue()
    assert "Escape requested a graceful stop" in output
    assert "run was stopped before held-out evaluation" in output
    assert "Run stopped gracefully" in output


def test_stage_catalog_explains_actions_instead_of_repeating_command_names() -> None:
    assert STAGES["routed"] == ("Route experts", "combine frozen experts and distill an executable pathway")
    assert STAGES["composition"] == ("Compose modules", "wire reusable modules with trainable glue")
    assert STAGES["query"] == ("Held-out query", "score the support-selected executable champion once")


def test_logging_defaults_clean_and_verbose_never_enables_raw_debug_records() -> None:
    try:
        logger = Logger.configure(verbose=False)
        assert logger.level == logging.WARNING
        logger = Logger.configure(verbose=True)
        assert logger.level == logging.INFO
        assert not logger.isEnabledFor(logging.DEBUG)
    finally:
        Logger.configure(verbose=False)
