"""[orchestrator] max_task_seconds: the per-attempt wall-clock budget.

Past the deadline the running stage finishes its current generation (the stall seam), later
ladder stages never start, and the attempt fails with its best champion while the wall ledger
still shelves it. Absent/0 is off and byte-identical: no counter key, and the stall factory
returns the bare StallDetector object.
"""

import time
import types
from pathlib import Path

from ardevo.dataset.icarus import Task
from ardevo.evolution.genome import Genome
from ardevo.orchestrator import StallDetector
from ardevo.strategy import EVOLVE_STRATEGY, StrategyResult
from tests.test_orchestrator import _fake_run_task, _orchestrator, _patch_run_task

_CALLS: list[str] = []


@EVOLVE_STRATEGY.register("tb_first")
def _build_tb_first(config: dict):
    def run(task, spec, runtime, *, budget: int, seed_comps=None) -> StrategyResult:
        _CALLS.append("first")
        return StrategyResult(strategy="tb_first", metric=0.5, generations_used=1)

    return run


@EVOLVE_STRATEGY.register("tb_second")
def _build_tb_second(config: dict):
    def run(task, spec, runtime, *, budget: int, seed_comps=None) -> StrategyResult:
        _CALLS.append("second")
        return StrategyResult(strategy="tb_second", metric=0.6, generations_used=1)

    return run


_STONE_GENOME: dict[str, Genome] = {}


@EVOLVE_STRATEGY.register("tb_stone")
def _build_tb_stone(config: dict):
    def run(task, spec, runtime, *, budget: int, seed_comps=None, seed_entries=None) -> StrategyResult:
        metrics = {"query_accuracy": 0.9, "query_loss": 0.1, "support_accuracy": 0.9, "weight_robustness": 0.5}
        return StrategyResult(strategy="tb_stone", metric=0.9, generations_used=1, champion_genome=_STONE_GENOME["genome"], champion_metrics=metrics)

    return run


def test_off_is_byte_identical(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    assert "time_budget_hits" not in orchestrator.counters
    assert isinstance(orchestrator._runtime().stall_factory(5), StallDetector)  # the bare detector, no wrapper


def test_budget_on_but_outside_solve_still_returns_bare_detector(tmp_path: Path) -> None:
    # The deadline exists only inside a solve(); factory calls made outside one (white-box tests,
    # future callers) must keep the identical object flow.
    orchestrator = _orchestrator(tmp_path, table={"max_task_seconds": 600})
    assert "time_budget_hits" in orchestrator.counters
    assert isinstance(orchestrator._runtime().stall_factory(5), StallDetector)


def test_expired_deadline_stops_ladder_after_first_strategy(tmp_path: Path, xor_task: Task) -> None:
    table = {"max_task_seconds": 1e-6, "max_depth": 0, "evolve": ["tb_first", "tb_second"], "evolve_budget": {"tb_first": 0.5, "tb_second": 0.5}, "decompose": []}
    orchestrator = _orchestrator(tmp_path, table=table)
    _CALLS.clear()
    assert orchestrator.solve(xor_task) is None
    assert _CALLS == ["first"]  # position 0 always runs; position 1 never starts
    attempt = orchestrator.attempts[-1]
    assert attempt.outcome == "failed" and attempt.failure_stage == "time_budget"
    assert attempt.metric == 0.5  # the best loser is still returned and recorded
    assert orchestrator.counters["time_budget_hits"] == 1


def test_expired_total_budget_is_recorded_without_starting_new_work(tmp_path: Path, xor_task: Task, monkeypatch) -> None:
    table = {"max_total_task_seconds": 3600, "max_depth": 0, "evolve": ["tb_first", "tb_second"], "decompose": []}
    orchestrator = _orchestrator(tmp_path, table=table)
    monkeypatch.setattr(type(orchestrator), "_total_deadline_exceeded", lambda _self: True)
    _CALLS.clear()
    assert orchestrator.solve(xor_task) is None
    assert _CALLS == []
    assert orchestrator.attempts[-1].strategy == "time_budget"
    assert orchestrator.counters["time_budget_hits"] == 1
    assert orchestrator.counters["total_time_budget_hits"] == 1


def test_total_timeout_finalizes_a_remembered_parent_champion(tmp_path: Path, xor_task: Task, solving_genome: Genome, monkeypatch) -> None:
    orchestrator = _orchestrator(
        tmp_path,
        table={"max_total_task_seconds": 1, "blind_query": True, "search_metric": "support_accuracy", "report_metric": "query_accuracy"},
    )
    result = StrategyResult(
        strategy="direct",
        metric=0.8,
        generations_used=3,
        champion_genome=solving_genome,
        champion_metrics={"support_accuracy": 0.8, "support_loss": 0.2},
    )
    orchestrator._best_parent_result = result
    finalized: list[bool] = []

    def attach(candidate, _spec):
        finalized.append(True)
        candidate.report_metrics = {"query_accuracy": 0.7, "query_loss": 0.3}
        return candidate

    monkeypatch.setattr(orchestrator, "_attach_report_metrics", attach)
    orchestrator._record_total_timeout(xor_task, 0)

    attempt = orchestrator.attempts[-1]
    assert finalized == [True]  # reporting is allowed after the search deadline
    assert attempt.strategy == "direct" and attempt.generations == 3
    assert attempt.support_accuracy == 0.8 and attempt.query_accuracy == 0.7
    assert attempt.failure_stage == "time_budget"


def test_recursive_solve_inherits_earlier_total_deadline(tmp_path: Path, xor_task: Task) -> None:
    orchestrator = _orchestrator(tmp_path, table={"max_task_seconds": 3600, "max_total_task_seconds": 3600})
    total_deadline = time.perf_counter() + 30
    orchestrator._total_task_deadline = total_deadline
    observed: list[float | None] = []

    def observe_deadline(_task: Task, depth: int = 0):
        observed.append(orchestrator._solve_deadline)
        return None

    setattr(orchestrator, "_solve_timed", observe_deadline)

    orchestrator.solve(xor_task, depth=1)

    assert observed == [total_deadline]
    assert orchestrator._total_task_deadline == total_deadline


def test_deadline_stop_wrapper_defers_to_the_detector_until_the_deadline(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path, table={"max_task_seconds": 3600})
    improving = [types.SimpleNamespace(fitness=float(step), metrics={"query_accuracy": 0.5, "query_loss": 0.1}) for step in range(4)]

    orchestrator._solve_deadline = time.perf_counter() + 3600
    stop = orchestrator._runtime().stall_factory(10)
    assert not isinstance(stop, StallDetector)  # wrapped: a deadline is active
    assert all(not stop(generation, best) for generation, best in enumerate(improving))  # detector rules while time remains

    orchestrator._solve_deadline = time.perf_counter() - 1.0
    expired = orchestrator._runtime().stall_factory(10)
    assert expired(0, improving[0])  # past the deadline the stop fires even while fitness improves
    orchestrator._solve_deadline = None


def test_timed_out_attempt_still_shelves_a_wall_stone(tmp_path: Path, xor_task: Task, solving_genome: Genome) -> None:
    table = {
        "max_task_seconds": 1e-6,
        "max_depth": 0,
        "evolve": ["tb_stone"],
        "evolve_budget": {"tb_stone": 1.0},
        "decompose": [],
        "wall": {"ledger": True, "min_metric": 0.45, "seed_top_k": 1},
    }
    orchestrator = _orchestrator(tmp_path, table=table)
    _STONE_GENOME["genome"] = solving_genome.clone()

    assert orchestrator.solve(xor_task) is None
    attempt = orchestrator.attempts[-1]
    assert attempt.outcome == "failed" and attempt.failure_stage == "time_budget"
    assert attempt.library_key is not None  # the champion was shelved despite the timeout
    assert orchestrator.counters["wall_stones_admitted"] == 1
    stone = orchestrator.library.load(attempt.library_key)
    assert stone.provenance.get("stepping_stone") is True


def test_decompose_skipped_past_deadline(tmp_path: Path, decomposable_task: Task) -> None:
    metrics = {"half_parity": 0.2, "half_parity.out0": 1.0, "half_parity.out1": 1.0}

    def staged(orchestrator):
        calls: list[str] = []

        def run_task(spec, state, **kwargs):
            if spec.output_ref == "half_parity" and "half_parity" in calls:
                return _fake_run_task({"half_parity": 1.0}, calls)(spec, state, **kwargs)
            return _fake_run_task(metrics, calls)(spec, state, **kwargs)

        _patch_run_task(orchestrator, run_task)

    unbudgeted = _orchestrator(tmp_path / "unbudgeted")
    staged(unbudgeted)
    assert unbudgeted.solve(decomposable_task) is not None
    assert unbudgeted.counters["decompositions"] == 1  # the control: this task DOES decompose given time

    budgeted = _orchestrator(tmp_path / "budgeted", table={"max_task_seconds": 1e-6})
    staged(budgeted)
    assert budgeted.solve(decomposable_task) is None
    assert budgeted.counters["decompositions"] == 0
    attempt = budgeted.attempts[-1]
    assert attempt.outcome == "failed" and attempt.failure_stage == "time_budget"
    assert budgeted.counters["time_budget_hits"] == 1
