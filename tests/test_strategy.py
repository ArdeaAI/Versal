"""Evolve strategies: ordering and budget carry, B4 champion verification, direct mode, temporal."""

import random
from dataclasses import replace as gene_replace
from pathlib import Path

from ardevo.dataset.icarus import Task
from ardevo.evolution.genome import InnovationTracker, genome_to_dict
from ardevo.evolution.loop import AssessedComposition, CompTaskSpec, HierarchicalLoop, HierarchicalState
from ardevo.evolution.registry import build_loop
from ardevo.library import MODULE, task_io
from ardevo.strategy import EVOLVE_STRATEGY, CompositionStrategy, GrammarStrategy, StrategyResult, StrategyRuntime
from tests.test_hierarchical_loop import _config as _loop_config
from tests.test_hierarchical_loop import _live_comp, _spec
from tests.test_orchestrator import _orchestrator


def _runtime_for(loop: HierarchicalLoop, state: HierarchicalState, threshold: float) -> StrategyRuntime:
    from ardevo.library import ModuleLibrary

    library = loop.library if loop.library is not None else ModuleLibrary("/tmp/claude/unused_strategy_test_lib")
    return StrategyRuntime(
        loop=loop,
        library=library,
        state=state,
        accept_threshold=threshold,
        metric_of=lambda item: float(item.metrics.get("support_accuracy", 0.0)),
        stall_factory=lambda budget: lambda generation, best: False,
    )


def test_verification_reassembles_against_current_state(xor_task: Task) -> None:
    """B4 regression: the admitted champion must carry CURRENT module weights, never stale ones."""
    loop = build_loop(_loop_config())
    state = loop.fresh_state(random.Random(0))
    state.comp_innovations = InnovationTracker(_next_node_id=100)
    spec = _spec(xor_task)
    species_id = sorted(state.species_champions)[0]
    best = loop.assess_composition(_live_comp(species_id, 2, 1), spec, state, train=False)
    assert best.net is not None
    stale = dict(best.net.inner_modules[f"live:{species_id}"].export_weights())

    # The world moves on after run_task returns: the champion genome gets new weights.
    moved = state.species_champions[species_id].clone()
    moved.connections = [gene_replace(conn, weight=conn.weight + 1.0) for conn in moved.connections]
    state.species_champions[species_id] = moved

    fresh = CompositionStrategy()._verify(best, spec, _runtime_for(loop, state, threshold=2.0))
    assert fresh.net is not None
    fresh_weights = fresh.net.inner_modules[f"live:{species_id}"].export_weights()
    assert fresh_weights != stale
    expected = {(conn.in_id, conn.out_id, conn.recurrent): conn.weight for conn in moved.connections if conn.enabled}
    assert all(abs(fresh_weights[key] - expected[key]) < 1e-6 for key in fresh_weights)


def test_evolve_runs_strategies_in_order_with_budget_carry(tmp_path: Path, xor_task: Task) -> None:
    calls: list[tuple[str, int]] = []

    @EVOLVE_STRATEGY.register("stub_low")
    def _build_low(config: dict):
        def run(task, spec: CompTaskSpec, runtime, *, budget: int, seed_comps=None) -> StrategyResult:
            calls.append(("low", budget))
            return StrategyResult(strategy="stub_low", metric=0.1, generations_used=3)

        return run

    @EVOLVE_STRATEGY.register("stub_high")
    def _build_high(config: dict):
        def run(task, spec: CompTaskSpec, runtime, *, budget: int, seed_comps=None) -> StrategyResult:
            calls.append(("high", budget))
            from ardevo.evolution.composition import minimal_composition

            comp = minimal_composition(spec.input_specs, spec.output_ref, spec.output_width, runtime.state.comp_innovations, runtime.state.rng)
            champion = AssessedComposition(comp=comp, metrics={"query_accuracy": 1.0, "query_loss": 0.1, "support_accuracy": 1.0, "support_loss": 0.1}, fitness=1.0, net=None)
            return StrategyResult(strategy="stub_high", metric=1.0, generations_used=2, champion_comp=champion, champion_metrics=dict(champion.metrics))

        return run

    @EVOLVE_STRATEGY.register("stub_never")
    def _build_never(config: dict):
        def run(task, spec, runtime, *, budget, seed_comps=None) -> StrategyResult:
            raise AssertionError("first-success-wins must prevent this strategy from running")

        return run

    table = {"evolve": ["stub_low", "stub_high", "stub_never"], "evolve_budget": {"stub_low": 1.0, "stub_high": 1.0, "stub_never": 1.0}, "budgets": {"depth0": 9}, "decompose": []}
    orchestrator = _orchestrator(tmp_path, table=table)
    solution = orchestrator.solve(xor_task)
    assert solution is not None
    # Shares give stub_low round(9/3)=3; it used 3 of 9, leaving 6; stub_high gets min(6, 3)=3.
    assert calls == [("low", 3), ("high", 3)]
    attempt = orchestrator.attempts[-1]
    assert attempt.outcome == "evolved" and attempt.strategy == "stub_high" and attempt.generations == 2


def test_metric_only_routed_miss_escalates_to_an_admissible_strategy(tmp_path: Path, xor_task: Task) -> None:
    calls: list[str] = []

    @EVOLVE_STRATEGY.register("stub_undistillable_routed")
    def _build_undistillable(config: dict):
        def run(task, spec, runtime, *, budget, seed_comps=None) -> StrategyResult:
            calls.append("routed")
            return StrategyResult(
                strategy="routed",
                metric=0.2,
                generations_used=1,
                champion_metrics={"support_accuracy": 1.0, "query_accuracy": 1.0, "routed_undistillable": 1.0},
            )

        return run

    @EVOLVE_STRATEGY.register("stub_after_routed")
    def _build_after_routed(config: dict):
        def run(task, spec: CompTaskSpec, runtime, *, budget, seed_comps=None) -> StrategyResult:
            calls.append("composition")
            from ardevo.evolution.composition import minimal_composition

            comp = minimal_composition(spec.input_specs, spec.output_ref, spec.output_width, runtime.state.comp_innovations, runtime.state.rng)
            metrics = {"support_accuracy": 1.0, "support_loss": 0.0, "query_accuracy": 1.0, "query_loss": 0.0}
            champion = AssessedComposition(comp=comp, metrics=metrics, fitness=1.0, net=None)
            return StrategyResult(strategy="composition", metric=1.0, generations_used=1, champion_comp=champion, champion_metrics=metrics)

        return run

    table = {
        "evolve": ["stub_undistillable_routed", "stub_after_routed"],
        "evolve_budget": {"stub_undistillable_routed": 1.0, "stub_after_routed": 1.0},
        "budgets": {"depth0": 4},
        "decompose": [],
    }
    orchestrator = _orchestrator(tmp_path, table=table)

    solution = orchestrator.solve(xor_task)

    assert calls == ["routed", "composition"]
    assert solution is not None and solution.entry_type == "composition"
    assert orchestrator.attempts[-1].strategy == "composition"


def test_direct_strategy_solves_xor_admits_real_io_and_revisit_hits(tmp_path: Path, xor_task: Task) -> None:
    table = {"evolve": ["direct"], "accept_threshold": 0.2, "decompose": [], "budgets": {"depth0": 3}, "direct": {"pop_size": 12, "elitism": 2}}
    orchestrator = _orchestrator(tmp_path, table=table)
    first = orchestrator.solve(xor_task)
    assert first is not None
    attempt = orchestrator.attempts[-1]
    assert attempt.outcome == "evolved" and attempt.strategy == "direct"
    # The size readout rides every direct result: champion scalars plus final-population stats.
    sizes = attempt.size_metrics
    assert sizes["champion_nodes"] > 0.0 and sizes["champion_connections"] > 0.0
    assert sizes["pop_max_nodes"] >= sizes["pop_median_nodes"] > 0.0
    assert sizes["pop_max_connections"] >= sizes["pop_median_connections"]
    assert first.key is not None
    entry = orchestrator.library.load(first.key)
    assert entry.entry_type == MODULE
    assert entry.io == task_io(xor_task)  # REAL task-shaped io, not the ANY 4->2 snapshot shape
    assert entry.provenance["strategy"] == "direct"

    second = orchestrator.solve(xor_task)
    assert second is not None
    assert orchestrator.attempts[-1].outcome == "library_hit"
    assert orchestrator.counters["library_hits"] == 1


def test_direct_strategy_picks_temporal_adapter(temporal_task: Task, xor_task: Task) -> None:
    from ardevo.evolution.evolver import TaskAdapter
    from ardevo.temporal import TemporalTaskAdapter

    config = {
        "evolution": {"init": {"kind": "minimal"}, "mutation": {"operators": []}, "train": {"kind": "none"}},
        "fitness": {"components": []},
    }
    strategy = EVOLVE_STRATEGY.get("direct")(config)
    temporal = strategy._adapter(temporal_task)
    assert isinstance(temporal, TemporalTaskAdapter) and temporal.mode == "last"
    flat = strategy._adapter(xor_task)
    assert isinstance(flat, TaskAdapter)


def test_stamp_input_coordinates_unravels_row_major(linear_genome) -> None:
    import pytest

    from ardevo.evolution.init import minimal, stamp_input_coordinates

    genome = minimal(6, 1, rng=random.Random(0))
    stamped = stamp_input_coordinates(genome, (2, 3))
    coordinates = [stamped.nodes[node_id].coordinate for node_id in stamped.input_ids]
    assert coordinates == [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0)]
    assert stamped.nodes[stamped.bias_ids[0]].coordinate is None  # only inputs carry geometry
    with pytest.raises(ValueError, match="cells"):
        stamp_input_coordinates(linear_genome, (3, 3))


def test_seed_state_seeded_front_injects_grafted_genome(tmp_path: Path, xor_task: Task, xor_adapter, solving_genome) -> None:
    from ardevo.evolution.registry import build_evolver
    from ardevo.library import ModuleLibrary, graft

    library = ModuleLibrary(tmp_path / "lib")
    key = library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={})
    entry = library.load(key)

    def run_once():
        evolver = build_evolver(_loop_config())
        return evolver.seed_state(xor_adapter, random.Random(0), seeded_front=lambda tracker: [graft(entry, tracker)])

    state = run_once()
    seeded = state.population[0]
    assert len(seeded.genome.nodes) == len(solving_genome.nodes)
    assert len(seeded.genome.connections) == len(solving_genome.connections)
    assert seeded.metrics["support_accuracy"] == 1.0  # the graft flows through assess_many like everyone
    again = run_once()
    assert genome_to_dict(again.population[0].genome) == genome_to_dict(seeded.genome)  # deterministic


def test_seed_state_without_seeded_front_is_unchanged(xor_adapter) -> None:
    from ardevo.evolution.registry import build_evolver

    baseline = build_evolver(_loop_config()).seed_state(xor_adapter, random.Random(0))
    explicit_none = build_evolver(_loop_config()).seed_state(xor_adapter, random.Random(0), seeded_front=None)
    assert [genome_to_dict(item.genome) for item in baseline.population] == [genome_to_dict(item.genome) for item in explicit_none.population]


def test_direct_strategy_seed_entries_clears_bar_in_one_generation(tmp_path: Path, xor_task: Task, solving_genome) -> None:
    from ardevo.orchestrator import comp_task_spec

    orchestrator = _orchestrator(tmp_path, table={"evolve": ["direct", "composition"], "direct": {"pop_size": 8, "elitism": 2}})
    key = orchestrator.library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={"accepted_metric": 1.0})
    entry = orchestrator.library.load(key)
    strategy = dict(orchestrator.strategies)["direct"]
    runtime = orchestrator._refine_runtime(0.99)
    result = strategy(xor_task, comp_task_spec(xor_task), runtime, budget=2, seed_entries=[entry])
    assert result.metric >= 0.99  # the grafted incumbent already clears the bar; the early exit fires
    assert result.champion_genome is not None
    # Refine fairness: the seed's own trained standing is stamped as the incumbent baseline.
    assert result.seed_metric is not None and result.seed_metric >= 0.99


def test_direct_strategy_without_seed_entries_stamps_no_seed_metric(tmp_path: Path, xor_task: Task) -> None:
    from ardevo.orchestrator import comp_task_spec

    orchestrator = _orchestrator(tmp_path, table={"evolve": ["direct", "composition"], "direct": {"pop_size": 4, "elitism": 1}})
    strategy = dict(orchestrator.strategies)["direct"]
    result = strategy(xor_task, comp_task_spec(xor_task), orchestrator._runtime(), budget=1)
    assert result.seed_metric is None


def test_grammar_strategy_seeds_compatible_program_into_direct(monkeypatch, tmp_path: Path, xor_task: Task, solving_genome) -> None:
    from types import SimpleNamespace

    from ardevo import grammar as grammar_module
    from ardevo.library import ModuleLibrary

    loop = build_loop(_loop_config())
    loop.attach_library(ModuleLibrary(tmp_path / "lib"))
    state = loop.fresh_state(random.Random(0))
    runtime = _runtime_for(loop, state, threshold=0.9)
    captured: list = []

    def direct(task, spec, runtime, *, budget, seed_genomes=None, **_kwargs):
        captured.extend(seed_genomes or [])
        return StrategyResult("direct", 1.0, 1, champion_genome=solving_genome, champion_metrics={"support_accuracy": 1.0})

    strategy = GrammarStrategy(direct=direct)
    strategy._grammar = SimpleNamespace(productions=[object()])
    strategy._library_keys = tuple(runtime.library.keys())
    monkeypatch.setattr(strategy, "_programs", lambda _runtime: [object()])
    monkeypatch.setattr(grammar_module, "compile_program", lambda *_args, **_kwargs: solving_genome)

    result = strategy(xor_task, _spec(xor_task), runtime, budget=4)

    assert result.strategy == "grammar" and result.generations_used == 1
    assert captured == [solving_genome]


def test_grammar_strategy_is_zero_cost_before_independent_motifs_exist(tmp_path: Path, xor_task: Task) -> None:
    from ardevo.library import ModuleLibrary

    loop = build_loop(_loop_config())
    loop.attach_library(ModuleLibrary(tmp_path / "empty"))
    state = loop.fresh_state(random.Random(0))
    strategy = GrammarStrategy(direct=lambda *_args, **_kwargs: StrategyResult("direct", 0.0, 0))

    result = strategy(xor_task, _spec(xor_task), _runtime_for(loop, state, threshold=0.9), budget=4)

    assert result.generations_used == 0 and result.metric == 0.0
    assert result.champion_metrics["grammar_productions"] == 0.0


def test_lookup_quick_evaluates_temporal_modules(tmp_path: Path, temporal_task: Task) -> None:
    """A direct-admitted temporal module must be reusable at lookup time through the stepped decode."""
    from tests.test_recurrence import _running_parity_genome

    orchestrator = _orchestrator(tmp_path)
    key = orchestrator.library.add(
        entry_type=MODULE,
        payload=genome_to_dict(_running_parity_genome()),
        io=task_io(temporal_task),
        provenance={"accepted_metric": 1.0, "weight_robustness": 0.5},
    )
    solution = orchestrator.solve(temporal_task)
    assert solution is not None and solution.key == key
    attempt = orchestrator.attempts[-1]
    assert attempt.outcome == "library_hit" and attempt.metric == 1.0
