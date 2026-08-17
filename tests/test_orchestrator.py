"""Orchestrator: ladder order, stall detection, decomposition recursion, admission, anti-forgetting."""

import json
import math
import random
from pathlib import Path
from typing import cast

import pytest
import torch

from tests.test_hierarchical_loop import _config as _loop_config
from versal import checkpoint
from versal.dataset.icarus import Task
from versal.evolution.composition import AssemblyContext, ComposedNet, assemble, comp_from_dict, minimal_composition
from versal.evolution.genome import InnovationTracker, genome_from_dict, genome_to_dict
from versal.evolution.loop import AssessedComposition, state_from_dict, state_to_dict
from versal.evolution.registry import build_loop
from versal.library import COMPOSITION, MODULE, ModuleLibrary, task_io
from versal.orchestrator import Attempt, Orchestrator, RefinementRank, StallDetector, attempts_from_dicts, attempts_to_dicts, comp_task_spec, refinement_improves
from versal.strategy import StrategyResult
from versal.trials.orchestrated_trial import OrchestratedTrial


def _orchestrator(tmp_path: Path, **overrides) -> Orchestrator:
    config = _loop_config()
    config["orchestrator"] = {
        "tasks": 4,
        "accept_metric": "query_accuracy",
        "accept_threshold": 0.95,
        "floor": 0.05,
        "stall_generations": 2,
        "stall_epsilon": 0.005,
        "max_depth": 2,
        "quick_eval_top_k": 5,
        "decompose": ["output_slices", "input_subsets"],
        "output_slices_n_groups": 2,
        "input_subsets_n_subsets": 2,
        "budgets": {"depth0": 3, "depth1": 2, "depth2": 2},
        **overrides.pop("table", {}),
    }
    config.update(overrides.pop("config_extra", {}))
    loop = build_loop(config)
    library = ModuleLibrary(tmp_path / "lib")
    loop.attach_library(library)
    state = loop.fresh_state(random.Random(0))
    return Orchestrator(config, loop, library, state, **overrides)


def _patch_run_task(orchestrator: Orchestrator, fake) -> None:
    # Dynamic method shadowing for white-box ladder tests; setattr keeps the type checker out of it.
    setattr(orchestrator.loop, "run_task", fake)


def _fake_run_task(metric_by_ref: dict[str, float], calls: list[str]):
    def run_task(spec, state, *, budget, stop=None, seed_comps=None, on_generation=None):
        calls.append(spec.output_ref)
        metric = metric_by_ref.get(spec.output_ref, 0.0)
        if seed_comps:
            comp = seed_comps[0]  # pretend the wired skeleton won, so its refs flow into admission
        else:
            comp = minimal_composition(spec.input_specs, spec.output_ref, spec.output_width, state.comp_innovations, state.rng)
        metrics = {"query_accuracy": metric, "query_loss": 0.1, "support_accuracy": metric, "support_loss": 0.1}
        return AssessedComposition(comp=comp, metrics=metrics, fitness=metric, net=None)

    return run_task


def test_library_hit_short_circuits_evolution(tmp_path: Path, xor_task: Task, solving_genome) -> None:
    orchestrator = _orchestrator(tmp_path)
    orchestrator.library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={"accepted_metric": 1.0})
    calls: list[str] = []
    _patch_run_task(orchestrator, _fake_run_task({}, calls))  # would record any evolve attempt
    solution = orchestrator.solve(xor_task)
    assert solution is not None and solution.metric == 1.0
    assert calls == []  # not one generation spent
    assert orchestrator.attempts[-1].outcome == "library_hit"
    assert orchestrator.counters["library_hits"] == 1


def test_lookup_shutdown_after_support_skips_blind_report(monkeypatch, tmp_path: Path, xor_task: Task, solving_genome) -> None:
    orchestrator = _orchestrator(
        tmp_path,
        table={"blind_query": True, "accept_metric": "support_accuracy", "accept_threshold": 0.95},
    )
    orchestrator.library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={"accepted_metric": 1.0})
    shutdown = False
    calls = 0
    assessment = AssessedComposition(
        comp=minimal_composition([("BINARY|K", 2)], "xor", 1, InnovationTracker(_next_node_id=0), random.Random(0)),
        metrics={"support_accuracy": 1.0, "support_loss": 0.0},
        fitness=1.0,
        net=None,
    )

    def assess(*_args, **_kwargs):
        nonlocal calls, shutdown
        calls += 1
        if calls > 1:
            raise AssertionError("shutdown after support must skip held-out lookup evaluation")
        shutdown = True
        return assessment

    monkeypatch.setattr(orchestrator, "_quick_assessment", assess)
    orchestrator._shutdown_requested_callback = lambda: shutdown

    hit = orchestrator._lookup(xor_task, comp_task_spec(xor_task, include_query=False))

    assert hit is not None
    assert calls == 1
    assert hit.query_accuracy is None
    assert hit.query_status == "shutdown_before_evaluation"


def test_field_library_hit_matches_across_resolution_after_full_support_check(tmp_path: Path) -> None:
    from tests.test_field import _task
    from versal.dataset.icarus import Axis, Field, TaskKind, TaskMeta, ValueType
    from versal.evolution.init import minimal
    from versal.field import field_contract, field_feature_width, field_payload

    source = _task()
    contract = field_contract(source)
    assert contract is not None
    genome = minimal(field_feature_width(contract.input_channels), contract.output_channels, rng=random.Random(0))
    orchestrator = _orchestrator(tmp_path, table={"accept_metric": "support_accuracy", "accept_threshold": 0.0})
    orchestrator.library.add(entry_type=MODULE, payload=field_payload(genome, contract), io=task_io(source), provenance={"accepted_metric": 1.0})

    axes = (Axis.CHANNEL, Axis.HEIGHT, Axis.WIDTH)
    input_field = Field(torch.zeros(2, 6, 7), axes, ValueType.CONTINUOUS, None, None, None)
    output_field = Field(torch.zeros(1, 6, 7), axes, ValueType.CONTINUOUS, None, None, None)
    target = Task(TaskMeta(0, TaskKind.MAP, "foreign", fixed_split=True), [(input_field, output_field)], [])
    solution = orchestrator.solve(target)
    assert solution is not None and solution.entry_type == MODULE
    assert orchestrator.attempts[-1].outcome == "library_hit"
    assert orchestrator.attempts[-1].task_metrics["cross_resolution_reuse"] == 1.0


def test_field_admission_uses_isolated_library_and_persists_metadata(tmp_path: Path) -> None:
    from tests.test_field import _task

    table = {
        "accept_metric": "support_accuracy",
        "search_metric": "support_accuracy",
        "accept_threshold": 0.0,
        "max_depth": 0,
        "evolve": ["field"],
        "budgets": {"depth0": 1},
        "decompose": [],
        "field": {
            "pop_size": 2,
            "elitism": 1,
            "train_sites": 8,
            "audit_sites": 8,
            "verify_top_k": 1,
            "train": {"kind": "gradient", "steps": 1, "lr": 0.01, "writeback": True},
            "evaluate": {"kind": "standard"},
        },
    }
    orchestrator = _orchestrator(tmp_path, table=table)
    solution = orchestrator.solve(_task())
    assert solution is not None and solution.key is not None
    entry = orchestrator.library.load(solution.key)
    assert entry.entry_type == MODULE
    assert entry.payload["field_template"]["version"] == "local_multiscale_v1"


def test_shutdown_request_records_graceful_stop_without_deadline_accounting(tmp_path: Path, xor_task: Task) -> None:
    requested = True
    orchestrator = _orchestrator(tmp_path, shutdown_requested=lambda: requested)

    assert orchestrator.solve(xor_task) is None

    attempt = orchestrator.attempts[-1]
    assert attempt.strategy == "shutdown"
    assert attempt.failure_stage == "shutdown_requested"
    assert attempt.support_status == "not_reached"
    assert attempt.query_status == "shutdown_before_evaluation"
    assert "time_budget_hits" not in orchestrator.counters


def test_stall_triggers_decompose_recurse_and_level2_admission(tmp_path: Path, decomposable_task: Task) -> None:
    orchestrator = _orchestrator(tmp_path)
    calls: list[str] = []
    metrics = {"half_parity": 0.2, "half_parity.out0": 1.0, "half_parity.out1": 1.0}

    def staged_run_task(spec, state, **kwargs):
        if spec.output_ref == "half_parity" and "half_parity" in calls:
            return _fake_run_task({"half_parity": 1.0}, calls)(spec, state, **kwargs)  # the re-evolve after wiring
        return _fake_run_task(metrics, calls)(spec, state, **kwargs)

    _patch_run_task(orchestrator, staged_run_task)
    solution = orchestrator.solve(decomposable_task)
    assert solution is not None and solution.entry_type == COMPOSITION
    assert calls == ["half_parity", "half_parity.out0", "half_parity.out1", "half_parity"]
    outcomes = [attempt.outcome for attempt in orchestrator.attempts]
    assert outcomes == ["evolved", "evolved", "decomposed"]  # two depth-1 accepts, then the wired parent
    assert orchestrator.counters["decompositions"] == 1
    assert solution.key is not None
    entry = orchestrator.library.load(solution.key)
    assert entry.level == 3  # composition over level-2 sub-compositions
    assert len(orchestrator.library) == 3


def test_max_depth_prevents_decomposition(tmp_path: Path, decomposable_task: Task) -> None:
    orchestrator = _orchestrator(tmp_path, table={"max_depth": 0})
    calls: list[str] = []
    _patch_run_task(orchestrator, _fake_run_task({}, calls))
    solution = orchestrator.solve(decomposable_task)
    assert solution is None
    assert orchestrator.attempts[-1].outcome == "failed"
    assert orchestrator.counters["decompositions"] == 0
    assert calls == ["half_parity"]  # one evolve attempt, no recursion


def test_attempts_carry_champion_sample_metrics(tmp_path: Path, xor_task: Task) -> None:
    """The G0 diagnostic: hybrid-eval champions stamp their weight-sample metrics onto the Attempt
    (whitelisted keys only), and metrics without the weight-sample marker leave the field empty."""
    from versal.evolution.composition import minimal_composition as _minimal

    orchestrator = _orchestrator(tmp_path, table={"max_depth": 0})

    def run_task(spec, state, *, budget, stop=None, seed_comps=None, on_generation=None):
        comp = _minimal(spec.input_specs, spec.output_ref, spec.output_width, state.comp_innovations, state.rng)
        metrics = {
            "query_accuracy": 0.6,
            "query_loss": 0.4,
            "support_accuracy": 0.6,
            "support_loss": 0.4,
            "mean_sample_accuracy": 0.55,
            "max_sample_accuracy": 0.91,
            "min_sample_accuracy": 0.4,
            "best_sample_weight": -1.0,
            "weight_robustness": 0.38,
            "mean_sample_loss": 0.7,
        }
        return AssessedComposition(comp=comp, metrics=metrics, fitness=0.6, net=None)

    _patch_run_task(orchestrator, run_task)
    assert orchestrator.solve(xor_task) is None  # below the bar -> failed attempt
    attempt = orchestrator.attempts[-1]
    assert attempt.outcome == "failed"
    assert attempt.sample_metrics == {"mean_sample_accuracy": 0.55, "max_sample_accuracy": 0.91, "best_sample_weight": -1.0, "weight_robustness": 0.38}

    plain = _orchestrator(tmp_path / "plain", table={"max_depth": 0})
    _patch_run_task(plain, _fake_run_task({}, []))
    assert plain.solve(xor_task) is None
    assert plain.attempts[-1].sample_metrics == {}  # standard-eval metrics fabricate nothing


def test_held_out_report_cannot_replace_search_robustness(tmp_path: Path, xor_task: Task) -> None:
    orchestrator = _orchestrator(tmp_path)
    spec = comp_task_spec(xor_task)
    comp = minimal_composition(spec.input_specs, spec.output_ref, spec.output_width, orchestrator.state.comp_innovations, orchestrator.state.rng)
    search = AssessedComposition(comp=comp, metrics={"support_accuracy": 0.8, "weight_robustness": 0.25}, fitness=0.8, net=cast(ComposedNet, object()))
    result = StrategyResult("composition", metric=0.8, generations_used=1, champion_comp=search, champion_metrics=dict(search.metrics))
    reported = AssessedComposition(comp=comp, metrics={"query_accuracy": 1.0, "query_loss": 0.0, "weight_robustness": 0.99}, fitness=1.0, net=cast(ComposedNet, object()))
    previous_deadline = 123.0

    def previous_callback() -> bool:
        return True

    orchestrator.loop.evolver.deadline = previous_deadline
    orchestrator.loop.evolver.deadline_exceeded = previous_callback
    calls = 0

    def assess_report(*_args, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["train"] is False
        assert orchestrator.loop.evolver.deadline is None
        assert orchestrator.loop.evolver.deadline_exceeded is None
        return reported

    setattr(orchestrator.loop, "assess_composition", assess_report)

    attached = orchestrator._attach_report_metrics(result, xor_task, spec)
    attached_again = orchestrator._attach_report_metrics(result, xor_task, spec)

    assert attached_again is attached is result
    assert calls == 1
    assert attached.report_attempted is True
    assert attached.champion_metrics["weight_robustness"] == 0.25
    assert attached.report_metrics["weight_robustness"] == 0.99
    assert attached.champion_comp is search
    assert orchestrator.loop.evolver.deadline == previous_deadline
    assert orchestrator.loop.evolver.deadline_exceeded is previous_callback


def test_field_report_finalization_uses_report_only_candidate_once(monkeypatch, tmp_path: Path, xor_task: Task, solving_genome) -> None:
    from versal.strategy import FieldStrategy

    orchestrator = _orchestrator(tmp_path, table={"blind_query": True, "evolve": ["field"], "decompose": []})
    field_strategy = dict(orchestrator.strategies)["field"]
    assert isinstance(field_strategy, FieldStrategy)
    candidate = solving_genome.clone()
    template = {"version": "test"}
    result = StrategyResult(
        "field",
        metric=0.8,
        generations_used=1,
        report_candidate_genome=candidate,
        champion_metrics={"support_accuracy": 0.8},
        field_template=template,
    )
    previous_deadline = 123.0

    def previous_callback() -> bool:
        return True

    field_strategy.evolver.deadline = previous_deadline
    field_strategy.evolver.deadline_exceeded = previous_callback
    calls = 0

    def evaluate_report(self, genome, task, field_template):
        nonlocal calls
        calls += 1
        assert self.evolver.deadline is None
        assert self.evolver.deadline_exceeded is None
        assert genome is candidate and task is xor_task and field_template is template
        return {"query_accuracy": 0.7, "query_loss": 0.3}

    monkeypatch.setattr(FieldStrategy, "evaluate_report", evaluate_report)

    attached = orchestrator._attach_report_metrics(result, xor_task, comp_task_spec(xor_task))
    orchestrator._attach_report_metrics(result, xor_task, comp_task_spec(xor_task))

    assert attached.report_attempted is True
    assert attached.report_metrics == {"query_accuracy": 0.7, "query_loss": 0.3}
    assert calls == 1
    assert field_strategy.evolver.deadline == previous_deadline
    assert field_strategy.evolver.deadline_exceeded is previous_callback
    assert not orchestrator._accepts_result(attached)


def test_grammar_report_finalization_uses_its_nested_direct_strategy(monkeypatch, tmp_path: Path, xor_task: Task, solving_genome) -> None:
    from versal.strategy import DirectStrategy, GrammarStrategy

    orchestrator = _orchestrator(tmp_path, table={"blind_query": True, "evolve": ["grammar"], "decompose": []})
    grammar = dict(orchestrator.strategies)["grammar"]
    assert isinstance(grammar, GrammarStrategy)
    assert isinstance(grammar.direct, DirectStrategy)
    candidate = solving_genome.clone()
    result = StrategyResult(
        "grammar",
        metric=0.8,
        generations_used=1,
        report_candidate_genome=candidate,
        champion_metrics={"support_accuracy": 0.8},
    )
    evaluated: list = []

    def evaluate_report(self, genome, task):
        assert self is grammar.direct
        assert task is xor_task
        evaluated.append(genome)
        return {"query_accuracy": 0.7, "query_loss": 0.3}

    monkeypatch.setattr(DirectStrategy, "evaluate_report", evaluate_report)

    orchestrator._attach_report_metrics(result, xor_task, comp_task_spec(xor_task))
    orchestrator._attach_report_metrics(result, xor_task, comp_task_spec(xor_task))

    assert evaluated == [candidate]
    assert result.report_attempted is True
    assert result.report_metrics["query_accuracy"] == 0.7


def test_best_loser_prefers_highest_support_reportable_payload(tmp_path: Path, xor_task: Task, solving_genome) -> None:
    orchestrator = _orchestrator(
        tmp_path,
        table={
            "blind_query": True,
            "search_metric": "support_accuracy",
            "accept_metric": "support_accuracy",
            "accept_threshold": 0.95,
            "decompose": [],
        },
    )
    low = solving_genome.clone()
    high = solving_genome.clone()

    def diagnostic(*_args, **_kwargs):
        return StrategyResult("diagnostic", metric=0.99, generations_used=1, champion_metrics={"support_accuracy": 0.99})

    def low_reportable(*_args, **_kwargs):
        return StrategyResult(
            "low",
            metric=0.4,
            generations_used=1,
            report_candidate_genome=low,
            champion_metrics={"support_accuracy": 0.4},
        )

    def high_reportable(*_args, **_kwargs):
        return StrategyResult(
            "high",
            metric=0.8,
            generations_used=1,
            report_candidate_genome=high,
            champion_metrics={"support_accuracy": 0.8},
        )

    orchestrator.strategies = [("diagnostic", diagnostic), ("low", low_reportable), ("high", high_reportable)]
    orchestrator.evolve_shares = {name: 1.0 for name, _strategy in orchestrator.strategies}

    result = orchestrator._evolve(xor_task, comp_task_spec(xor_task, include_query=False), budget=3)

    assert result.strategy == "high"
    assert result.report_candidate_genome is high
    assert result.metric == 0.8


def test_non_blind_best_loser_preserves_highest_metric_routed_payload(tmp_path: Path, xor_task: Task, solving_genome) -> None:
    orchestrator = _orchestrator(tmp_path, table={"accept_threshold": 0.95, "decompose": []})

    def routed(*_args, **_kwargs):
        return StrategyResult(
            "routed",
            metric=0.8,
            generations_used=1,
            champion_routed=object(),
            champion_metrics={"support_accuracy": 0.8, "query_accuracy": 0.8},
        )

    def module(*_args, **_kwargs):
        return StrategyResult(
            "direct",
            metric=0.7,
            generations_used=1,
            champion_genome=solving_genome,
            champion_metrics={"support_accuracy": 0.7, "query_accuracy": 0.7},
        )

    orchestrator.strategies = [("routed", routed), ("direct", module)]
    orchestrator.evolve_shares = {name: 1.0 for name, _strategy in orchestrator.strategies}

    result = orchestrator._evolve(xor_task, comp_task_spec(xor_task), budget=2)

    assert result.strategy == "routed"
    assert result.metric == 0.8


def test_non_blind_parent_recovery_preserves_stronger_routed_payload(tmp_path: Path, solving_genome) -> None:
    orchestrator = _orchestrator(tmp_path, table={"accept_threshold": 0.95})
    routed = StrategyResult(
        "routed",
        metric=0.8,
        generations_used=1,
        champion_routed=object(),
        champion_metrics={"support_accuracy": 0.8, "query_accuracy": 0.8},
    )
    module = StrategyResult(
        "direct",
        metric=0.7,
        generations_used=1,
        champion_genome=solving_genome,
        champion_metrics={"support_accuracy": 0.7, "query_accuracy": 0.7},
    )

    orchestrator._remember_or_recover_parent_result(routed, depth=0)

    assert orchestrator._remember_or_recover_parent_result(module, depth=0) is routed
    assert orchestrator._best_parent_result is routed


def test_parent_recovery_retains_the_strongest_reportable_payload(tmp_path: Path, solving_genome) -> None:
    orchestrator = _orchestrator(tmp_path, table={"blind_query": True, "accept_metric": "support_accuracy", "accept_threshold": 0.95})
    first = StrategyResult(
        "direct",
        metric=0.8,
        generations_used=1,
        report_candidate_genome=solving_genome,
        champion_metrics={"support_accuracy": 0.8},
    )
    weaker = StrategyResult(
        "direct",
        metric=0.7,
        generations_used=1,
        report_candidate_genome=solving_genome.clone(),
        champion_metrics={"support_accuracy": 0.7},
    )

    assert orchestrator._remember_or_recover_parent_result(first, depth=0) is first
    assert orchestrator._remember_or_recover_parent_result(weaker, depth=0) is first


def test_parent_reporting_prefers_higher_support_router_over_admissible_payload(tmp_path: Path, solving_genome) -> None:
    orchestrator = _orchestrator(
        tmp_path,
        table={"blind_query": True, "search_metric": "support_accuracy", "accept_metric": "support_accuracy", "accept_threshold": 0.75},
    )
    router = StrategyResult(
        "routed",
        metric=0.1,
        generations_used=1,
        report_candidate_routed=object(),
        report_candidate_metrics={"support_accuracy": 0.9},
        champion_metrics={"routed_metric": 0.9},
    )
    reusable = StrategyResult(
        "direct",
        metric=0.8,
        generations_used=1,
        champion_genome=solving_genome,
        champion_metrics={"support_accuracy": 0.8},
    )

    orchestrator._consider_parent_report_result(router, depth=0)
    orchestrator._consider_parent_report_result(reusable, depth=0)

    assert orchestrator._report_result_for(reusable, depth=0) is router
    assert orchestrator._accepts_result(reusable)
    assert not orchestrator._accepts_result(router)


def test_accepted_parent_is_not_masked_by_a_stronger_report_only_incumbent(tmp_path: Path, solving_genome) -> None:
    orchestrator = _orchestrator(tmp_path, table={"accept_metric": "support_accuracy", "accept_threshold": 0.95})
    report_only = StrategyResult(
        "direct",
        metric=0.99,
        generations_used=1,
        report_candidate_genome=solving_genome,
        champion_metrics={"support_accuracy": 0.99},
    )
    accepted = StrategyResult(
        "direct",
        metric=0.95,
        generations_used=1,
        champion_genome=solving_genome.clone(),
        champion_metrics={"support_accuracy": 0.95},
    )

    orchestrator._remember_or_recover_parent_result(report_only, depth=0)

    assert orchestrator._remember_or_recover_parent_result(accepted, depth=0) is accepted
    assert orchestrator._best_parent_result is accepted


def test_failed_decomposition_records_the_strongest_retained_parent_report(tmp_path: Path, xor_task: Task, solving_genome) -> None:
    orchestrator = _orchestrator(
        tmp_path,
        table={
            "blind_query": True,
            "max_depth": 1,
            "accept_metric": "support_accuracy",
            "accept_threshold": 0.95,
            "decompose": ["output_slices"],
        },
    )
    original = StrategyResult(
        "direct",
        metric=0.4,
        generations_used=1,
        report_candidate_genome=solving_genome,
        champion_metrics={"support_accuracy": 0.4},
        report_metrics={"query_accuracy": 0.3, "query_loss": 0.7},
        report_attempted=True,
    )
    stronger_recomposition = StrategyResult(
        "direct",
        metric=0.8,
        generations_used=2,
        report_candidate_genome=solving_genome.clone(),
        champion_metrics={"support_accuracy": 0.8},
        report_metrics={"query_accuracy": 0.7, "query_loss": 0.3},
        report_attempted=True,
    )

    setattr(orchestrator, "_evolve", lambda *_args, **_kwargs: original)
    setattr(orchestrator, "_wants_decompose_first", lambda *_args, **_kwargs: False)

    def failed_recomposition(*_args, **_kwargs):
        orchestrator._best_parent_result = stronger_recomposition
        return None

    setattr(orchestrator, "_decompose_and_recurse", failed_recomposition)

    assert orchestrator.solve(xor_task) is None

    attempt = orchestrator.attempts[-1]
    assert attempt.metric == 0.8
    assert attempt.support_accuracy == 0.8
    assert attempt.query_accuracy == 0.7
    assert attempt.query_status == "evaluated"


def test_timed_out_recomposition_is_reported_but_not_accepted(tmp_path: Path, decomposable_task: Task, solving_genome) -> None:
    orchestrator = _orchestrator(
        tmp_path,
        table={
            "blind_query": True,
            "accept_metric": "support_accuracy",
            "accept_threshold": 0.95,
            "decompose": ["output_slices"],
        },
    )
    orchestrator._decomposition_leaf_count = 1
    orchestrator._solve_deadline = math.inf
    result = StrategyResult(
        "direct",
        metric=1.0,
        generations_used=1,
        champion_genome=solving_genome,
        champion_metrics={"support_accuracy": 1.0},
        report_metrics={"query_accuracy": 0.7, "query_loss": 0.3},
        report_attempted=True,
    )

    setattr(orchestrator, "_subtasks_promising", lambda *_args, **_kwargs: True)
    setattr(orchestrator, "solve", lambda *_args, **_kwargs: object())
    setattr(orchestrator, "_port_wired_skeleton", lambda *_args, **_kwargs: None)

    def timed_out_recomposition(*_args, **_kwargs) -> StrategyResult:
        orchestrator._solve_deadline = 0.0
        return result

    setattr(orchestrator, "_evolve", timed_out_recomposition)
    setattr(orchestrator, "_admit_result", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("timed-out support result must not be admitted")))

    assert orchestrator._decompose_and_recurse(decomposable_task, comp_task_spec(decomposable_task, include_query=False), 0, 2, 0.0) is None
    assert orchestrator.counters["accepts"] == 0


def test_recomposition_report_overrun_is_distinguished_from_support_timeout(tmp_path: Path, decomposable_task: Task, solving_genome) -> None:
    orchestrator = _orchestrator(
        tmp_path,
        table={
            "blind_query": True,
            "accept_metric": "support_accuracy",
            "accept_threshold": 0.95,
            "decompose": ["output_slices"],
        },
    )
    orchestrator._decomposition_leaf_count = 1
    orchestrator._solve_deadline = math.inf
    result = StrategyResult(
        "direct",
        metric=0.8,
        generations_used=1,
        report_candidate_genome=solving_genome,
        champion_metrics={"support_accuracy": 0.8},
    )

    setattr(orchestrator, "_subtasks_promising", lambda *_args, **_kwargs: True)
    setattr(orchestrator, "solve", lambda *_args, **_kwargs: object())
    setattr(orchestrator, "_port_wired_skeleton", lambda *_args, **_kwargs: None)
    setattr(orchestrator, "_evolve", lambda *_args, **_kwargs: result)

    def report_after_deadline(candidate: StrategyResult, *_args, **_kwargs) -> StrategyResult:
        candidate.report_attempted = True
        candidate.report_metrics = {"query_accuracy": 0.7, "query_loss": 0.3}
        orchestrator._solve_deadline = 0.0
        return candidate

    setattr(orchestrator, "_attach_report_metrics", report_after_deadline)

    assert orchestrator._decompose_and_recurse(decomposable_task, comp_task_spec(decomposable_task, include_query=False), 0, 2, 0.0) is None
    assert getattr(orchestrator, "_last_decompose_report_overrun", False) is True


def test_decompose_first_report_overrun_skips_fallback_without_time_budget_failure(tmp_path: Path, xor_task: Task, solving_genome) -> None:
    orchestrator = _orchestrator(
        tmp_path,
        table={"blind_query": True, "max_depth": 1, "decompose_first_above": 1, "accept_metric": "support_accuracy", "accept_threshold": 0.95},
    )
    retained = StrategyResult(
        "direct",
        metric=0.8,
        generations_used=1,
        report_candidate_genome=solving_genome,
        champion_metrics={"support_accuracy": 0.8},
        report_metrics={"query_accuracy": 0.7, "query_loss": 0.3},
        report_attempted=True,
    )
    setattr(orchestrator, "_wants_decompose_first", lambda *_args, **_kwargs: True)

    def report_overrun(*_args, **_kwargs):
        orchestrator._best_parent_result = retained
        setattr(orchestrator, "_last_decompose_report_overrun", True)
        orchestrator._solve_deadline = 0.0
        return None

    setattr(orchestrator, "_decompose_and_recurse", report_overrun)
    setattr(orchestrator, "_evolve", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("expired report must not start a fallback population")))

    assert orchestrator.solve(xor_task) is None

    attempt = orchestrator.attempts[-1]
    assert attempt.metric == 0.8
    assert attempt.query_accuracy == 0.7
    assert attempt.failure_stage != "time_budget"
    assert orchestrator.counters.get("time_budget_hits", 0) == 0


def test_missing_reporter_does_not_claim_an_evaluation_attempt(tmp_path: Path, xor_task: Task, solving_genome) -> None:
    orchestrator = _orchestrator(tmp_path, table={"blind_query": True})
    orchestrator.strategies = []
    result = StrategyResult(
        "direct",
        metric=0.8,
        generations_used=1,
        report_candidate_genome=solving_genome,
        champion_metrics={"support_accuracy": 0.8},
    )

    orchestrator._attach_report_metrics(result, xor_task, comp_task_spec(xor_task))

    assert result.report_attempted is False
    assert result.report_metrics == {}


def test_report_only_candidate_exposes_query_without_becoming_acceptable(tmp_path: Path, solving_genome) -> None:
    orchestrator = _orchestrator(
        tmp_path,
        table={
            "blind_query": True,
            "accept_metric": "support_accuracy",
            "search_metric": "support_accuracy",
            "report_metric": "query_accuracy",
        },
    )
    result = StrategyResult(
        "direct",
        metric=0.8,
        generations_used=1,
        report_candidate_genome=solving_genome,
        champion_metrics={"support_accuracy": 0.8},
        report_metrics={"query_accuracy": 0.7, "query_loss": 0.3},
        report_attempted=True,
    )

    attempt = orchestrator._attempt_from_result(result, task="xor", depth=0, outcome="failed")

    assert attempt.support_accuracy == 0.8
    assert attempt.support_status == "support_verification_incomplete"
    assert attempt.query_accuracy == 0.7
    assert attempt.query_status == "evaluated"
    assert not orchestrator._accepts_result(result)


def test_passing_metrics_without_an_executable_payload_are_never_accepted() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator.accept_metric = "support_task_appropriate"
    orchestrator.accept_threshold = 0.95
    result = StrategyResult(
        "routed",
        metric=0.1,
        generations_used=10,
        champion_metrics={"support_accuracy": 1.0, "support_task_exact": 1.0, "routed_undistillable": 1.0},
    )

    assert not result.has_admissible_champion
    assert not orchestrator._accepts_result(result)


def test_failed_subtask_fails_the_decomposition_gracefully(tmp_path: Path, decomposable_task: Task) -> None:
    orchestrator = _orchestrator(tmp_path)
    calls: list[str] = []
    _patch_run_task(orchestrator, _fake_run_task({"half_parity.out0": 1.0}, calls))  # out1 never clears
    solution = orchestrator.solve(decomposable_task)
    assert solution is None
    assert orchestrator.attempts[-1].outcome == "failed"
    # out1 itself recursed to depth 2 (output_slices cannot split a 1-wide output, so it went
    # straight to failed) before the parent gave up.
    assert any(attempt.task == "half_parity.out1" and attempt.outcome == "failed" for attempt in orchestrator.attempts)


def test_stall_detector_flatline_and_floor() -> None:
    from versal.evolution.composition import CompositionGenome

    def metric_of(item: AssessedComposition) -> float:
        return item.metrics.get("query_accuracy", 0.0)

    def assessed(accuracy: float, fitness: float) -> AssessedComposition:
        return AssessedComposition(comp=CompositionGenome(), metrics={"query_accuracy": accuracy}, fitness=fitness, net=None)

    flat = assessed(0.9, 0.5)
    detector = StallDetector(stall_generations=3, stall_epsilon=0.01, floor=0.2, budget=100, metric_of=metric_of)
    assert not detector(0, flat)  # the first observation counts as the baseline improvement
    assert not detector(1, flat) and not detector(2, flat)
    assert detector(3, flat)  # three consecutive generations without improvement
    improving = StallDetector(stall_generations=3, stall_epsilon=0.01, floor=0.2, budget=100, metric_of=metric_of)
    for generation in range(6):
        assert not improving(generation, assessed(0.9, 0.1 * generation))
    hopeless = assessed(0.1, 1.0)
    floor_detector = StallDetector(stall_generations=50, stall_epsilon=0.01, floor=0.2, budget=10, metric_of=metric_of)
    assert not floor_detector(0, hopeless)  # before half budget the floor is not checked
    assert floor_detector(5, hopeless)


def test_port_wired_skeleton_assembles_and_routes(tmp_path: Path, decomposable_task: Task) -> None:
    from versal.decompose import output_slices
    from versal.evolution.composition import comp_to_dict
    from versal.orchestrator import Solution

    orchestrator = _orchestrator(tmp_path)
    spec = comp_task_spec(decomposable_task)
    subtasks = output_slices(decomposable_task, rng=random.Random(0), n_groups=2)
    solutions = []
    for subtask in subtasks:
        comp = minimal_composition([("BINARY|K", 8)], subtask.task.meta.name, 1, InnovationTracker(_next_node_id=0), random.Random(1))
        key = orchestrator.library.add(entry_type=COMPOSITION, payload=comp_to_dict(comp), io=task_io(subtask.task), provenance={}, level=2)
        solutions.append((subtask, Solution(key=key, entry_type=COMPOSITION, metric=1.0)))
    skeleton = orchestrator._port_wired_skeleton(spec, solutions)
    assert skeleton is not None and len(skeleton.module_ids) == 2
    net = assemble(skeleton, AssemblyContext(bank_columns=dict(spec.bank_columns), library=orchestrator.library), spec.n_inputs)
    out = net(torch.zeros(3, 8))
    assert out.shape == (3, 2)


def test_port_wired_skeleton_declines_oversized_plan_before_glue_allocation(monkeypatch, tmp_path: Path, decomposable_task: Task) -> None:
    import versal.orchestrator as orchestrator_module
    from versal.decompose import output_slices
    from versal.evolution.composition import comp_to_dict
    from versal.orchestrator import Solution

    orchestrator = _orchestrator(tmp_path)
    orchestrator.loop.max_initial_glue_values = 100
    spec = comp_task_spec(decomposable_task)
    solutions = []
    for subtask in output_slices(decomposable_task, rng=random.Random(0), n_groups=2):
        comp = minimal_composition([("BINARY|K", 8)], subtask.task.meta.name, 1, InnovationTracker(_next_node_id=0), random.Random(1))
        key = orchestrator.library.add(entry_type=COMPOSITION, payload=comp_to_dict(comp), io=task_io(subtask.task), provenance={}, level=2)
        solutions.append((subtask, Solution(key=key, entry_type=COMPOSITION, metric=1.0)))

    def unexpected_allocation(*_args, **_kwargs):
        raise AssertionError("the skeleton guard must run before dense glue construction")

    monkeypatch.setattr(orchestrator_module, "_identity_glue", unexpected_allocation)
    monkeypatch.setattr(orchestrator_module, "_placement_glue", unexpected_allocation)
    monkeypatch.setattr(orchestrator_module, "_selection_glue", unexpected_allocation)

    assert orchestrator._port_wired_skeleton(spec, solutions) is None


def test_real_evolve_admit_then_revisit_is_a_hit(tmp_path: Path, xor_task: Task) -> None:
    """The anti-forgetting regression test, on the REAL loop: accept a (weak) solution, admit it,
    and the same task seen again is answered from the library without evolving."""
    orchestrator = _orchestrator(tmp_path, table={"accept_threshold": 0.2, "decompose": [], "budgets": {"depth0": 2}})
    first = orchestrator.solve(xor_task)
    assert first is not None
    assert orchestrator.attempts[-1].outcome == "evolved"
    assert len(orchestrator.library) >= 1
    second = orchestrator.solve(xor_task)
    assert second is not None
    assert orchestrator.attempts[-1].outcome == "library_hit"
    assert orchestrator.counters["library_hits"] == 1 and orchestrator.counters["accepts"] == 1


def test_admission_detaches_live_refs(tmp_path: Path, xor_task: Task) -> None:
    """Admitted compositions must reference only library entries (live species are run-local)."""
    orchestrator = _orchestrator(tmp_path, table={"accept_threshold": 0.0, "decompose": [], "budgets": {"depth0": 2}})
    solution = orchestrator.solve(xor_task)
    assert solution is not None and solution.key is not None
    entry = orchestrator.library.load(solution.key)
    comp = comp_from_dict(entry.payload)
    assert all(ref.startswith("library:") for ref in comp.refs())


def test_orchestrated_checkpoint_writes_artifacts_only_for_new_library_entries(tmp_path: Path, xor_task: Task) -> None:
    orchestrator = _orchestrator(tmp_path, table={"accept_threshold": 0.0, "decompose": [], "budgets": {"depth0": 1}})
    before = set(orchestrator.library.keys())
    solution = orchestrator.solve(xor_task)
    assert solution is not None
    new_keys = [key for key in orchestrator.library.keys() if key not in before]
    assert new_keys

    trial = object.__new__(OrchestratedTrial)
    trial.config = {"orchestrator": {}, "schedule": {}, "evolution": {}, "fitness": {}, "dataset": "synthetic"}
    trial.library = orchestrator.library
    trial.loop = orchestrator.loop
    trial.scheduler = type("Scheduler", (), {"state_dict": lambda self: {"last": 0}})()
    trial.run_dir = tmp_path / "run"
    trial.task = None
    trial.rungs = [1]
    trial.skipped_rungs = []

    trial._checkpoint(orchestrator, orchestrator.state, 1, new_keys, solution)
    directory = trial.run_dir / "task_0001"
    assert (directory / "stats.json").exists()
    assert (directory / "checkpoint.json").exists()
    assert (directory / "speciation.png").exists()
    assert (directory / "net.png").exists()

    before_hit = set(orchestrator.library.keys())
    hit = orchestrator.solve(xor_task)
    assert hit is not None
    assert orchestrator.attempts[-1].outcome == "library_hit"
    assert [key for key in orchestrator.library.keys() if key not in before_hit] == []


def test_real_end_to_end_decompose_with_direct_subsolves(tmp_path: Path, xor_pairs_task: Task) -> None:
    """The REAL ladder, no stubs: depth-0 evolution fails fast, output_slices splits the task, the
    DIRECT strategy solves both XOR halves, and the port-wired skeleton answers the parent."""
    table = {
        "evolve": ["composition", "direct"],
        "evolve_budget": {"composition": 0.1, "direct": 0.9},
        "accept_threshold": 0.95,
        "floor": 0.0,
        "stall_generations": 15,
        "max_depth": 1,
        "decompose": ["output_slices"],
        "output_slices_n_groups": 2,
        "budgets": {"depth0": 4, "depth1": 60},
        "direct": {"pop_size": 24, "elitism": 2, "train": {"kind": "gradient", "steps": 60, "lr": 0.05, "writeback": True}},
    }
    orchestrator = _orchestrator(tmp_path, table=table)
    solution = orchestrator.solve(xor_pairs_task)
    assert solution is not None, [a.to_dict() for a in orchestrator.attempts]
    parent = orchestrator.attempts[-1]
    assert parent.outcome == "decomposed" and parent.decompose_op == "output_slices"
    assert parent.metric >= 0.95
    sub_attempts = [a for a in orchestrator.attempts if a.depth == 1 and a.outcome == "evolved"]
    assert len(sub_attempts) == 2 and all(a.metric >= 0.95 for a in sub_attempts)
    # The FIRST half must be cracked by direct structure growth; the second half may legitimately
    # win via composition instead, reusing the just-admitted module (knowledge compounding within
    # a single decompose, which is the whole point).
    assert sub_attempts[0].strategy == "direct"
    for attempt in sub_attempts:
        assert attempt.library_key is not None
        entry = orchestrator.library.load(attempt.library_key)
        assert entry.io["inputs"][0]["width"] == 4 and entry.io["output"]["width"] == 1  # real task shape
    assert solution.key is not None
    assert orchestrator.library.load(solution.key).level >= 2  # composition over the admitted parts
    assert orchestrator.counters["decompositions"] == 1
    assert orchestrator.counters["decompose_subtask_failed"] == 0
    assert orchestrator.counters["decompose_parent_failed"] == 0


def test_decompose_forensics_record_where_it_died(tmp_path: Path, decomposable_task: Task) -> None:
    orchestrator = _orchestrator(tmp_path)
    calls: list[str] = []
    _patch_run_task(orchestrator, _fake_run_task({"half_parity.out0": 1.0}, calls))  # out1 never clears
    assert orchestrator.solve(decomposable_task) is None
    assert orchestrator.counters["decompose_subtask_failed"] >= 1
    failed = orchestrator.attempts[-1]
    assert failed.failure_stage == "subtask:half_parity.out1" and failed.decompose_op == "output_slices"

    orchestrator2 = _orchestrator(tmp_path / "second", table={"max_depth": 1})
    calls2: list[str] = []
    metrics = {"half_parity.out0": 1.0, "half_parity.out1": 1.0}  # parts solve, parent never does
    _patch_run_task(orchestrator2, _fake_run_task(metrics, calls2))
    assert orchestrator2.solve(decomposable_task) is None
    assert orchestrator2.counters["decompose_parent_failed"] == 1
    assert orchestrator2.attempts[-1].failure_stage == "parent_re_evolve"


def test_solvability_gate_off_by_default_passes_all(tmp_path: Path, decomposable_task: Task) -> None:
    from versal.decompose import output_slices

    orchestrator = _orchestrator(tmp_path)
    assert orchestrator.decompose_solvability_floor == 0.0  # legacy behavior: gate disabled
    subtasks = output_slices(decomposable_task, rng=random.Random(0), n_groups=2)
    assert orchestrator._subtasks_promising(subtasks) is True  # no probe run when the floor is 0


def test_solvability_gate_rejects_unfittable_subtasks(tmp_path: Path, decomposable_task: Task) -> None:
    from versal.decompose import output_slices
    from versal.strategy import StrategyResult

    orchestrator = _orchestrator(tmp_path, table={"decompose_solvability_floor": 0.6, "decompose_probe_generations": 2})
    subtasks = output_slices(decomposable_task, rng=random.Random(0), n_groups=2)

    # Probe reports support that cannot be fit past the floor -> the decomposition is rejected.
    setattr(
        orchestrator,
        "_evolve",
        lambda task, spec, budget, seed_comps=None: StrategyResult(strategy="direct", metric=0.0, generations_used=budget, champion_metrics={"support_accuracy": 0.3}),
    )
    assert orchestrator._subtasks_promising(subtasks) is False
    diagnostic = orchestrator._best_diagnostic_observation
    assert diagnostic is not None and diagnostic["score"] == 0.3
    assert diagnostic["executable"] is False

    # Probe reports a fittable support -> the decomposition is allowed through.
    setattr(
        orchestrator,
        "_evolve",
        lambda task, spec, budget, seed_comps=None: StrategyResult(strategy="direct", metric=0.0, generations_used=budget, champion_metrics={"support_accuracy": 0.95}),
    )
    assert orchestrator._subtasks_promising(subtasks) is True
    diagnostic = orchestrator._best_diagnostic_observation
    assert diagnostic is not None and diagnostic["score"] == 0.95


def test_decomposition_leaf_cap_reports_representation_limit(tmp_path: Path) -> None:
    from tests.test_field import _task

    orchestrator = _orchestrator(
        tmp_path,
        table={"decompose": ["spatial_patches"], "spatial_patches_n_patches": 2, "decompose_leaf_cap": 2, "decompose_solvability_floor": 0.0},
    )
    orchestrator._decomposition_leaf_count = 2
    task = _task()
    assert orchestrator._decompose_and_recurse(task, comp_task_spec(task), 0, 1, 0.0) is None
    assert orchestrator._failure_stage == "representation_limit"


def test_orchestrated_payload_round_trips(tmp_path: Path, decomposable_task: Task) -> None:
    orchestrator = _orchestrator(tmp_path, table={"accept_threshold": 0.0, "decompose": [], "budgets": {"depth0": 1}})
    orchestrator.solve(decomposable_task)
    payload = checkpoint.build_orchestrated_payload(
        task_cursor=1,
        rng=orchestrator.state.rng,
        scheduler=type("S", (), {"state_dict": lambda self: {"last": 0}})(),
        speciator=orchestrator.loop.evolver.speciate,
        loop_state=state_to_dict(orchestrator.state),
        attempts=attempts_to_dicts(orchestrator.attempts),
        counters=orchestrator.counters,
    )
    raw = json.dumps(payload)  # everything must be JSON-able for checkpoint.json
    restored = json.loads(raw)
    state = state_from_dict(restored["loop_state"], checkpoint.deserialize_rng(restored["rng"]))
    assert state.generation == orchestrator.state.generation
    assert len(state.modules) == len(orchestrator.state.modules)
    attempts = attempts_from_dicts(restored["attempts"])
    assert attempts_to_dicts(attempts) == attempts_to_dicts(orchestrator.attempts)


# --- learn-mode refinement of library hits ---------------------------------------------------------


def _refine_table(**overrides) -> dict:
    return {"refine": {"budget_k": 8, "decay": 0.5, "min_generations": 4, "metric_epsilon": 0.005, "robustness_epsilon": 0.01, **overrides}}


def _fake_direct(metric: float, robustness: float, genome, calls: list[dict], seed_metric: float | None = None):
    """A stand-in direct strategy: records what refinement asked for, returns a fixed champion."""

    def strategy(task, spec, runtime, *, budget: int, seed_comps=None, seed_entries=None) -> StrategyResult:
        calls.append({"budget": budget, "seed_entries": seed_entries, "threshold": runtime.accept_threshold})
        return StrategyResult(
            strategy="direct",
            metric=metric,
            generations_used=budget,
            champion_genome=genome,
            champion_metrics={"weight_robustness": robustness, "query_accuracy": metric},
            seed_metric=seed_metric,
        )

    return strategy


def test_refine_disabled_is_exactly_todays_hit(tmp_path: Path, xor_task: Task, solving_genome) -> None:
    """budget_k = 0 (live mode, the default) must be byte-identical to the plain hit path."""
    orchestrator = _orchestrator(tmp_path)
    key = orchestrator.library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={"accepted_metric": 1.0})
    calls: list[str] = []
    _patch_run_task(orchestrator, _fake_run_task({}, calls))
    solution = orchestrator.solve(xor_task)
    assert solution is not None and solution.metric == 1.0 and calls == []
    attempt = orchestrator.attempts[-1]
    assert attempt.outcome == "library_hit" and attempt.refine_generations == 0
    assert "refine_generations" not in attempt.to_dict()  # run summaries stay byte-identical
    assert not any(name.startswith("refine") for name in orchestrator.counters)
    assert "refine_attempts" not in orchestrator.library.load(key).stats  # no stats I/O on the hit path


@pytest.mark.parametrize(
    ("metric", "robustness", "complexity", "entry_type", "expected"),
    [
        (0.91, 0.0, 99, MODULE, True),  # metric win beats everything downstream
        (0.89, 0.9, 1, MODULE, False),  # metric loss loses regardless
        (0.90, 0.52, 99, MODULE, False),  # inside metric band, bloat loses despite robustness
        (0.90, 0.48, 1, MODULE, True),  # inside metric band, simplicity wins first
        (0.90, 0.50, 9, MODULE, True),  # smaller topology wins (minimize over time)
        (0.90, 0.50, 10, MODULE, False),  # equal everything is a non-event
        (0.905, 0.50, 10, MODULE, False),  # epsilon boundary: not strictly beyond the band
        (0.90, 0.50, 1, COMPOSITION, True),  # expanded cost makes entry types comparable
        (math.nan, 0.9, 1, MODULE, False),  # non-finite candidate never wins
        (0.99, math.inf, 1, MODULE, False),
    ],
)
def test_refine_comparator_lexicographic(metric: float, robustness: float, complexity: int, entry_type: str, expected: bool) -> None:
    incumbent = RefinementRank(metric=0.90, robustness=0.50, complexity=10, entry_type=MODULE)
    candidate = RefinementRank(metric=metric, robustness=robustness, complexity=complexity, entry_type=entry_type)
    assert refinement_improves(candidate, incumbent, metric_epsilon=0.005, robustness_epsilon=0.01) is expected


def test_refine_improvement_admits_and_retires_superseded(tmp_path: Path, xor_task: Task, solving_genome, linear_genome) -> None:
    """A smaller topology at equal metric/robustness is a strict win: shelved, the bulky incumbent
    tombstoned (it is dominated on the stored ranking fields), and the decay reset."""
    orchestrator = _orchestrator(tmp_path, table=_refine_table())
    old_key = orchestrator.library.add(
        entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={"accepted_metric": 1.0, "weight_robustness": 0.5}
    )
    calls: list[dict] = []
    orchestrator.strategies = [("direct", _fake_direct(1.0, 0.5, linear_genome, calls))]
    solution = orchestrator.solve(xor_task)
    assert solution is not None and solution.key is not None and solution.key != old_key
    assert calls[0]["budget"] == 8 and calls[0]["threshold"] == pytest.approx(1.0 + 0.005)
    assert calls[0]["seed_entries"] is not None and calls[0]["seed_entries"][0].key == old_key
    attempt = orchestrator.attempts[-1]
    assert attempt.outcome == "refined" and attempt.refine_generations == 8 and attempt.library_key == solution.key
    assert orchestrator.library.is_retired(old_key)
    assert orchestrator.counters["refine_improvements"] == 1 and orchestrator.counters["refine_attempts"] == 1
    stats = orchestrator.library.load(old_key).stats
    assert stats["refine_attempts"] == 1 and stats["refine_failures_since_gain"] == 0  # gain resets decay
    assert orchestrator.library.load(solution.key).entry_type == MODULE


def test_refine_no_gain_returns_original_and_decays(tmp_path: Path, xor_task: Task, solving_genome, linear_genome) -> None:
    orchestrator = _orchestrator(tmp_path, table=_refine_table())
    key = orchestrator.library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={"accepted_metric": 1.0, "weight_robustness": 0.5})
    calls: list[dict] = []
    orchestrator.strategies = [("direct", _fake_direct(0.5, 0.0, linear_genome, calls))]
    first = orchestrator.solve(xor_task)
    assert first is not None and first.key == key and first.metric == 1.0  # never a regression
    attempt = orchestrator.attempts[-1]
    assert attempt.outcome == "library_hit" and attempt.refine_generations == 8  # the bounded extra compute
    assert orchestrator.counters["refine_no_gain"] == 1
    second = orchestrator.solve(xor_task)
    assert second is not None and second.key == key
    assert [call["budget"] for call in calls] == [8, 4]  # effective K decayed by 0.5 per failure
    third = orchestrator.solve(xor_task)  # failures = 2 -> effective 2 < min_generations 4 -> skip
    assert third is not None and len(calls) == 2
    assert orchestrator.counters["refine_skipped_decayed"] == 1
    assert orchestrator.counters["library_hits"] == 3


def test_refine_below_accept_threshold_never_admits(tmp_path: Path, xor_task: Task, solving_genome, linear_genome) -> None:
    """A robustness win inside the metric-tie band can still sit below the accept bar; it must not shelve."""
    orchestrator = _orchestrator(tmp_path, table={**_refine_table(), "accept_threshold": 1.0})
    key = orchestrator.library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={"accepted_metric": 1.0, "weight_robustness": 0.5})
    calls: list[dict] = []
    orchestrator.strategies = [("direct", _fake_direct(0.996, 0.9, linear_genome, calls))]  # tie on metric, better robustness
    solution = orchestrator.solve(xor_task)
    assert solution is not None and solution.key == key
    assert orchestrator.attempts[-1].outcome == "library_hit"
    assert orchestrator.counters["refine_no_gain"] == 1 and orchestrator.counters["refine_improvements"] == 0
    assert len(orchestrator.library) == 1  # nothing new shelved


@pytest.mark.parametrize("stop_reason", ["timeout", "shutdown"])
def test_refine_stopped_support_candidate_never_admits(
    stop_reason: str,
    tmp_path: Path,
    xor_task: Task,
    solving_genome,
    linear_genome,
) -> None:
    orchestrator = _orchestrator(tmp_path, table=_refine_table())
    old_key = orchestrator.library.add(
        entry_type=MODULE,
        payload=genome_to_dict(solving_genome),
        io=task_io(xor_task),
        provenance={"accepted_metric": 1.0, "weight_robustness": 0.5},
    )
    shutdown = False
    orchestrator._shutdown_requested_callback = lambda: shutdown

    def stopped_strategy(*_args, **_kwargs) -> StrategyResult:
        nonlocal shutdown
        if stop_reason == "timeout":
            orchestrator._solve_deadline = 0.0
        else:
            shutdown = True
        return StrategyResult(
            strategy="direct",
            metric=1.1,
            generations_used=1,
            champion_genome=linear_genome,
            champion_metrics={"query_accuracy": 1.1, "weight_robustness": 0.9},
        )

    orchestrator.strategies = [("direct", stopped_strategy)]
    setattr(orchestrator, "_admit_result", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("stopped refinement must not admit")))

    solution = orchestrator.solve(xor_task)

    assert solution is not None and solution.key == old_key
    assert orchestrator.attempts[-1].outcome == "library_hit"
    assert orchestrator.counters["refine_improvements"] == 0


def test_refine_depth_guard_skips_subsolve_hits(tmp_path: Path, xor_task: Task, solving_genome) -> None:
    orchestrator = _orchestrator(tmp_path, table=_refine_table())
    orchestrator.library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={"accepted_metric": 1.0})

    def exploding(task, spec, runtime, *, budget, seed_comps=None, seed_entries=None):
        raise AssertionError("refinement must not fire past refine depth_max")

    orchestrator.strategies = [("direct", exploding)]
    solution = orchestrator.solve(xor_task, 1)  # a hit inside a decompose recursion
    assert solution is not None
    assert orchestrator.attempts[-1].outcome == "library_hit"
    assert orchestrator.counters["refine_attempts"] == 0


def test_refine_composition_hit_seeds_run_task(tmp_path: Path, xor_task: Task) -> None:
    from versal.evolution.composition import comp_to_dict

    orchestrator = _orchestrator(tmp_path, table={**_refine_table(), "accept_threshold": 0.0})
    comp = minimal_composition([("BINARY|K", 2)], "xor", 1, InnovationTracker(_next_node_id=0), random.Random(1))
    key = orchestrator.library.add(entry_type=COMPOSITION, payload=comp_to_dict(comp), io=task_io(xor_task), provenance={"accepted_metric": 0.5}, level=2)
    received: dict = {}

    def fake_run_task(spec, state, *, budget, stop=None, seed_comps=None, on_generation=None):
        received["seed_comps"] = seed_comps
        received["budget"] = budget
        out = seed_comps[0] if seed_comps else minimal_composition(spec.input_specs, spec.output_ref, spec.output_width, state.comp_innovations, state.rng)
        return AssessedComposition(comp=out, metrics={"query_accuracy": 0.0, "query_loss": 0.1, "support_accuracy": 0.0, "support_loss": 0.1}, fitness=0.0, net=None)

    _patch_run_task(orchestrator, fake_run_task)
    solution = orchestrator.solve(xor_task)
    assert solution is not None and solution.key == key  # no gain: the original hit comes back
    assert received["seed_comps"] is not None and comp_to_dict(received["seed_comps"][0]) == comp_to_dict(comp)
    assert received["budget"] == 8


def test_refine_admission_rejected_returns_unshelved_and_counts_failure(tmp_path: Path, xor_task: Task, solving_genome, linear_genome) -> None:
    orchestrator = _orchestrator(tmp_path, table=_refine_table(), config_extra={"library": {"admission": "default", "min_metric": 2.0}})
    key = orchestrator.library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={"accepted_metric": 1.0, "weight_robustness": 0.5})
    calls: list[dict] = []
    orchestrator.strategies = [("direct", _fake_direct(1.0, 0.5, linear_genome, calls))]
    solution = orchestrator.solve(xor_task)
    assert solution is not None and solution.key is None and solution.metric == 1.0  # solved, not shelved
    assert orchestrator.attempts[-1].outcome == "refined"
    assert not orchestrator.library.is_retired(key)  # nothing replaced it on the shelf
    assert orchestrator.counters["admission_rejected"] == 1 and orchestrator.counters["refine_improvements"] == 1
    stats = orchestrator.library.load(key).stats
    assert stats["refine_failures_since_gain"] == 1  # unshelvable gains must not re-spend full K


def test_refined_attempt_round_trips_checkpoint() -> None:
    refined = Attempt(task="t", depth=0, outcome="refined", metric=0.99, generations=6, library_key="m1_abc", strategy="direct", refine_generations=6)
    hit_after_failed_refine = Attempt(task="t", depth=0, outcome="library_hit", metric=1.0, generations=0, refine_generations=4)
    restored = attempts_from_dicts(json.loads(json.dumps(attempts_to_dicts([refined, hit_after_failed_refine]))))
    assert restored[0].refine_generations == 6 and restored[0].outcome == "refined"
    assert restored[1].refine_generations == 4
    legacy = {"task": "t", "depth": 0, "outcome": "library_hit", "metric": 1.0, "generations": 0}
    assert Attempt.from_dict(legacy).refine_generations == 0  # pre-feature checkpoints resume cleanly
    plain = Attempt(task="t", depth=0, outcome="library_hit", metric=1.0, generations=0)
    assert "refine_generations" not in plain.to_dict()  # live-mode summaries stay byte-identical


def test_refine_skipped_when_matching_strategy_not_configured(tmp_path: Path, xor_task: Task, solving_genome) -> None:
    orchestrator = _orchestrator(tmp_path, table=_refine_table())  # default evolve = ["composition"]: no direct
    key = orchestrator.library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={"accepted_metric": 1.0})
    solution = orchestrator.solve(xor_task)
    assert solution is not None and solution.key == key
    assert orchestrator.attempts[-1].outcome == "library_hit"
    assert orchestrator.counters["refine_skipped_no_strategy"] == 1 and orchestrator.counters["refine_attempts"] == 0


def test_refine_retrained_clone_never_admits(tmp_path: Path, xor_task: Task, solving_genome) -> None:
    """The clone-factory guard: entry keys hash weights, so a topology-identical champion with
    retrained weights always gets a fresh key; exact topology identity must catch it even after ids
    are restamped
    (the 2026-07-03 incident admitted 11 such clones and tombstoned their parents)."""
    orchestrator = _orchestrator(tmp_path, table=_refine_table(deduplicate_topologies=True))
    old_key = orchestrator.library.add(
        entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={"accepted_metric": 1.0, "weight_robustness": 0.5}
    )
    reweighted = genome_to_dict(solving_genome)
    remap = {node["id"]: node["id"] + 100 for node in reweighted["nodes"]}
    for node in reweighted["nodes"]:
        node["id"] = remap[node["id"]]
    for connection in reweighted["connections"]:
        connection["in"] = remap[connection["in"]]
        connection["out"] = remap[connection["out"]]
        connection["weight"] += 0.5
        connection["innovation"] += 1000
    calls: list[dict] = []
    # Metric tie at 1.0, robustness 0.9 vs stored 0.5: the comparator ALONE would admit this.
    orchestrator.strategies = [("direct", _fake_direct(1.0, 0.9, genome_from_dict(reweighted), calls))]
    solution = orchestrator.solve(xor_task)
    assert solution is not None and solution.key == old_key  # the original hit, not a clone
    assert orchestrator.attempts[-1].outcome == "library_hit"
    assert len(orchestrator.library) == 1 and not orchestrator.library.is_retired(old_key)
    assert orchestrator.counters["refine_no_gain"] == 1 and orchestrator.counters["refine_improvements"] == 0
    assert orchestrator.library.load(old_key).stats["refine_failures_since_gain"] == 1  # decay bites
    assert "topology_duplicates_skipped" in orchestrator.attempts[-1].strategy_metrics  # survives into JSON/report aggregation


def test_refine_seed_metric_is_the_incumbent_baseline(tmp_path: Path, xor_task: Task, solving_genome, linear_genome) -> None:
    """A candidate must beat the incumbent GIVEN THE SAME TRAINING: when the strategy reports the
    seed's own trained standing, that (not the untrained quick metric) is the bar to clear."""
    table = _refine_table()
    for seed_metric, should_admit in ((1.03, False), (None, True)):
        base = tmp_path / ("with_seed" if seed_metric else "without_seed")
        orchestrator = _orchestrator(base, table=table)
        old_key = orchestrator.library.add(
            entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={"accepted_metric": 1.0, "weight_robustness": 0.5}
        )
        calls: list[dict] = []
        # Candidate at 1.02 beats the quick metric (1.0 + 0.005) but NOT the trained seed (1.03 - 0.005).
        orchestrator.strategies = [("direct", _fake_direct(1.02, 0.5, linear_genome, calls, seed_metric=seed_metric))]
        solution = orchestrator.solve(xor_task)
        assert solution is not None
        if should_admit:
            assert solution.key is not None and solution.key != old_key
            assert orchestrator.attempts[-1].outcome == "refined"
        else:
            assert solution.key == old_key
            assert orchestrator.attempts[-1].outcome == "library_hit"
            assert orchestrator.counters["refine_no_gain"] == 1


def test_retire_guard_requires_strict_margin(tmp_path: Path, xor_task: Task, solving_genome) -> None:
    """Weak dominance alone must not tombstone: an incumbent with degenerate stored robustness 0.0
    (every temporal module) would otherwise die to any same-metric clone."""
    orchestrator = _orchestrator(tmp_path, table=_refine_table())
    library = orchestrator.library
    incumbent = RefinementRank(metric=1.0, robustness=0.0, complexity=10, entry_type=MODULE)
    plants = iter(range(1, 10))

    def planted() -> str:
        payload = genome_to_dict(solving_genome)
        payload["connections"][0]["weight"] += next(plants)  # tombstones are permanent; each plant needs a fresh key
        key = library.add(entry_type=MODULE, payload=payload, io=task_io(xor_task), provenance={"accepted_metric": 1.0, "weight_robustness": 0.0})
        assert not library.is_retired(key)
        return key

    key = planted()
    tie = RefinementRank(metric=1.0, robustness=0.005, complexity=10, entry_type=MODULE)  # inside both epsilons, same size
    orchestrator._retire_if_dominated(key, tie, incumbent)
    assert not library.is_retired(key)

    robustness_win = RefinementRank(metric=1.0, robustness=0.5, complexity=10, entry_type=MODULE)
    orchestrator._retire_if_dominated(key, robustness_win, incumbent)
    assert library.is_retired(key)

    key = planted()
    simpler_at_parity = RefinementRank(metric=1.0, robustness=0.005, complexity=4, entry_type=MODULE)
    orchestrator._retire_if_dominated(key, simpler_at_parity, incumbent)
    assert library.is_retired(key)

    key = planted()
    simpler_other_type = RefinementRank(metric=1.0, robustness=0.005, complexity=4, entry_type=COMPOSITION)
    orchestrator._retire_if_dominated(key, simpler_other_type, incumbent)
    assert library.is_retired(key)  # expanded complexity makes cross-type comparison meaningful


def test_refine_capability_gain_recharges_lineage_cooldown(tmp_path: Path, xor_task: Task, solving_genome, linear_genome) -> None:
    """A metric/robustness win is new capability: the replacement entry starts a fresh cooldown."""
    orchestrator = _orchestrator(tmp_path, table=_refine_table())
    old_key = orchestrator.library.add(
        entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={"accepted_metric": 1.0, "weight_robustness": 0.4}
    )
    orchestrator.strategies = [("direct", _fake_direct(1.0, 0.6, linear_genome, []))]  # robustness tier win
    solution = orchestrator.solve(xor_task)
    assert solution is not None and solution.key is not None and solution.key != old_key
    entry = orchestrator.library.load(solution.key)
    assert entry.provenance["refined_from"] == old_key  # lineage is traceable
    assert entry.stats["refine_attempts"] == 1 and entry.stats["refine_failures_since_gain"] == 0


def test_refine_compression_gain_spends_lineage_cooldown(tmp_path: Path, xor_task: Task, solving_genome, linear_genome) -> None:
    """A complexity-only win is polish, not capability: the replacement inherits the chain's decay
    plus one failure, so one capability epoch funds only a few compression passes (24->12->6->skip),
    never an endless per-variant treadmill of near-identical entries."""
    orchestrator = _orchestrator(tmp_path, table=_refine_table())
    old_key = orchestrator.library.add(
        entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={"accepted_metric": 1.0, "weight_robustness": 0.5}
    )
    assert genome_from_dict(genome_to_dict(linear_genome)).complexity() < solving_genome.complexity()
    orchestrator.strategies = [("direct", _fake_direct(1.0, 0.5, linear_genome, []))]  # metric/robustness ties, smaller
    solution = orchestrator.solve(xor_task)
    assert solution is not None and solution.key is not None and solution.key != old_key
    head = orchestrator.library.load(solution.key)
    assert head.provenance["refined_from"] == old_key
    assert head.stats["refine_attempts"] == 1 and head.stats["refine_failures_since_gain"] == 1  # spent, not recharged
    assert orchestrator._effective_refine_budget(head) == 4  # 8 * 0.5^1: one more pass at most
    orchestrator.library.seed_refine_stats(solution.key, attempts=2, failures=2)  # after a second compression
    assert orchestrator._effective_refine_budget(orchestrator.library.load(solution.key)) == 2  # < min_generations: skips
    improved, generations = orchestrator._refine_hit(
        __import__("versal.orchestrator", fromlist=["Solution"]).Solution(key=solution.key, entry_type=MODULE, metric=1.0), xor_task, comp_task_spec(xor_task), 0
    )
    assert improved is None and generations == 0
    assert orchestrator.counters["refine_skipped_decayed"] == 1


def test_refine_always_spends_reduced_budget_on_every_hit(tmp_path: Path, xor_task: Task, solving_genome, linear_genome) -> None:
    orchestrator = _orchestrator(tmp_path, table=_refine_table(mode="always"))
    key = orchestrator.library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={"accepted_metric": 1.0})
    calls: list[dict] = []
    orchestrator.strategies = [("direct", _fake_direct(0.5, 0.0, linear_genome, calls))]

    for _ in range(6):
        solution = orchestrator.solve(xor_task)
        assert solution is not None and solution.key == key  # known-good fallback never regresses

    assert [call["budget"] for call in calls] == [8] * 6
    assert orchestrator.counters["refine_attempts"] == 6
    assert orchestrator.counters["refine_no_gain"] == 6
    assert orchestrator.counters["refine_skipped_decayed"] == 0


def _wall_table(**overrides) -> dict:
    return {"wall": {"ledger": True, "min_metric": 0.4, "seed_top_k": 1, **overrides}, "max_depth": 0, "decompose": []}


def test_wall_ledger_shelves_seeds_and_replaces(tmp_path: Path, xor_task: Task, solving_genome, linear_genome) -> None:
    """Failure leaves a trace: the best champion shelves as a below-bar dependency stone, the next
    attempt on the signature warm-starts from it, and the stone only upgrades on a strict win."""
    orchestrator = _orchestrator(tmp_path, table=_wall_table())
    calls: list[dict] = []
    orchestrator.strategies = [("direct", _fake_direct(0.6, 0.2, linear_genome, calls))]

    assert orchestrator.solve(xor_task) is None  # below the 0.95 accept bar
    stone_key = orchestrator.attempts[-1].library_key
    assert stone_key is not None
    stone = orchestrator.library.load(stone_key)
    assert stone.provenance["stepping_stone"] is True and stone.provenance["accepted_metric"] == 0.6
    stone_summary = orchestrator.library.summary(stone_key)
    assert stone_summary is not None and stone_summary["dependency"] is True  # out of caps, out of signature_group
    assert orchestrator.counters["wall_stones_admitted"] == 1
    assert calls[0]["seed_entries"] is None  # nothing to seed from on the first assault

    assert orchestrator.solve(xor_task) is None  # lookup misses (a stone never clears quick-eval)
    assert calls[1]["seed_entries"] is not None and calls[1]["seed_entries"][0].key == stone_key
    assert orchestrator.counters["wall_seeded_attempts"] == 1
    # The identical champion (same fingerprint) must NOT mint a second stone.
    assert orchestrator.counters["wall_stones_admitted"] == 1 and orchestrator.counters["wall_stones_improved"] == 0

    better = genome_to_dict(solving_genome)
    orchestrator.strategies = [("direct", _fake_direct(0.7, 0.2, genome_from_dict(better), calls))]  # strict metric win, new topology
    assert orchestrator.solve(xor_task) is None
    new_stone_key = orchestrator.attempts[-1].library_key
    assert new_stone_key is not None and new_stone_key != stone_key
    assert orchestrator.library.is_retired(stone_key)  # one stone per lineage
    assert orchestrator.library.load(new_stone_key).provenance["refined_from"] == stone_key
    assert orchestrator.counters["wall_stones_improved"] == 1


def test_wall_ledger_below_min_metric_shelves_nothing(tmp_path: Path, xor_task: Task, linear_genome) -> None:
    orchestrator = _orchestrator(tmp_path, table=_wall_table(min_metric=0.65))
    orchestrator.strategies = [("direct", _fake_direct(0.6, 0.2, linear_genome, []))]
    assert orchestrator.solve(xor_task) is None
    assert orchestrator.attempts[-1].library_key is None
    assert len(orchestrator.library) == 0 and orchestrator.counters["wall_stones_admitted"] == 0


def test_wall_ledger_off_is_byte_identical(tmp_path: Path, xor_task: Task, linear_genome) -> None:
    orchestrator = _orchestrator(tmp_path, table={"max_depth": 0, "decompose": []})
    orchestrator.strategies = [("direct", _fake_direct(0.6, 0.2, linear_genome, []))]
    assert orchestrator.solve(xor_task) is None
    assert not any(name.startswith("wall") for name in orchestrator.counters)
    assert len(orchestrator.library) == 0
