"""Phase 7 Pillar B: stall-triggered operator-rate escalation.

When the direct search plateaus with budget left (or hits the half-budget mark), it escalates the
depth/recursion operator probabilities for the rest of the run instead of idling on the plateau, then
restores the original pipeline so the boost never leaks into the next task."""

from functools import partial
from pathlib import Path
from typing import Any

from ardevo.dataset.icarus import Task
from ardevo.evolution.mutation import MutationPipeline, Mutator, add_deep_node, add_rich_node
from ardevo.evolution.stall import build_stall_strategy
from ardevo.strategy import DirectStrategy
from tests.test_orchestrator import _orchestrator


def _pipeline() -> MutationPipeline:
    operators: list[tuple[str, Mutator]] = [
        ("add_rich_node", partial(add_rich_node, prob=0.12, fan_in=4)),
        ("add_deep_node", partial(add_deep_node, prob=0.12, fan_in=4, fan_out=2)),
    ]
    return MutationPipeline(operators, base_rates={"add_rich_node": 0.12, "add_deep_node": 0.12})


def _kw(operator: Mutator) -> dict[str, Any]:
    assert isinstance(operator, partial)
    return dict(operator.keywords)


def test_with_boosted_rates_raises_prob_and_keeps_other_params() -> None:
    boosted = _pipeline().with_boosted_rates({"add_deep_node": 0.25})
    by_name = dict(boosted.operators)
    assert _kw(by_name["add_deep_node"])["prob"] == 0.25
    assert _kw(by_name["add_deep_node"])["fan_in"] == 4 and _kw(by_name["add_deep_node"])["fan_out"] == 2  # preserved
    assert boosted.base_rates["add_deep_node"] == 0.25


def test_with_boosted_rates_leaves_unnamed_operators_untouched() -> None:
    original = _pipeline()
    boosted = original.with_boosted_rates({"add_deep_node": 0.25})
    by_name = dict(boosted.operators)
    assert _kw(by_name["add_rich_node"]) == {"prob": 0.12, "fan_in": 4}  # unchanged
    # The original pipeline is not mutated in place (a fresh pipeline is returned).
    assert _kw(dict(original.operators)["add_deep_node"])["prob"] == 0.12


def test_build_stall_strategy_resolves_none_and_adapt_rates() -> None:
    assert build_stall_strategy({}) is None
    assert build_stall_strategy({"orchestrator": {"stall_strategy": "none"}}) is None
    escalate = build_stall_strategy({"orchestrator": {"stall_strategy": "adapt_rates", "stall": {"add_deep_node": 0.3}}})
    assert escalate is not None
    boosted = escalate(_pipeline())
    assert _kw(dict(boosted.operators)["add_deep_node"])["prob"] == 0.3


def test_direct_with_stall_strategy_still_solves_and_does_not_leak(tmp_path: Path, xor_task: Task) -> None:
    # End-to-end: a forced early stall (stall_generations 1) escalates mid-run; the run still solves and
    # the pipeline is restored, so a second solve starts from the un-escalated rates.
    table = {
        "evolve": ["direct"],
        "accept_threshold": 0.2,
        "decompose": [],
        "budgets": {"depth0": 4},
        "stall_generations": 1,
        "stall_strategy": "adapt_rates",
        "stall": {"add_rich_node": 0.9, "add_deep_node": 0.9},
        "direct": {"pop_size": 12, "elitism": 2},
    }
    orchestrator = _orchestrator(tmp_path, table=table)
    strategy = dict(orchestrator.strategies)["direct"]
    assert isinstance(strategy, DirectStrategy)
    original_pipeline = strategy.evolver.mutation
    solution = orchestrator.solve(xor_task)
    assert solution is not None
    assert orchestrator.attempts[-1].outcome == "evolved" and orchestrator.attempts[-1].strategy == "direct"
    assert strategy.evolver.mutation is original_pipeline  # escalation restored after the task
