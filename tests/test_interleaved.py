"""
Policy, exact-task identity, and persistent population regressions.
"""

import copy
import json
import math
import random
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tests.test_orchestrator import _orchestrator
from versal.dataset.icarus import Axis, Field, Task, TaskKind, TaskMeta, ValueType
from versal.evolution.genome import NodeGene, NodeKind
from versal.interleaved import TaskSearchState
from versal.library import MODULE, ModuleLibrary
from versal.orchestrator import comp_task_spec
from versal.strategy_common import StrategyResult
from versal.strategy_direct import DirectStrategy
from versal.strategy_sessions import SESSION_STRATEGY, DirectSession, StrategySession
from versal.temporal import TemporalTaskAdapter
from versal.topology import task_content_fingerprint


def _policy(**extra):
    return {
        "search_policy": "interleaved",
        "blind_query": True,
        "search_metric": "support_accuracy",
        "accept_metric": "support_accuracy",
        "report_metric": "query_accuracy",
        "cross_validation": {"enabled": True},
        "evolve": ["direct", "composition"],
        "evolve_budget": {"direct": 1.0, "composition": 1.0},
        "refine": {"budget_k": 4, "mode": "always", "depth_max": 0},
        **extra,
    }


def test_first_solution_keeps_competing_and_pins_smaller_champion(tmp_path, monkeypatch, xor_task, solving_genome):
    seen = []
    larger = solving_genome.clone()
    larger.nodes[100] = NodeGene(100, NodeKind.HIDDEN, "tanh")

    class Session(StrategySession):
        def _advance(self, runtime):
            seen.append((self.role, self.generations, self.refining))
            self.evaluations += 1
            genome = larger if self.role == "direct" else solving_genome
            return StrategyResult(self.role, 1.0, 1, champion_genome=genome.clone(), champion_metrics={"support_accuracy": 1.0})

    monkeypatch.setitem(SESSION_STRATEGY._items, "direct", Session)
    monkeypatch.setitem(SESSION_STRATEGY._items, "composition", Session)
    orchestrator = _orchestrator(tmp_path, table=_policy(), config_extra={"library": {"admission": "archive", "per_niche_cap": 0, "max_per_signature": 0}})
    first = orchestrator.solve(xor_task)
    assert orchestrator.search is not None
    assert first is not None and first.key is not None
    assert orchestrator.attempts[-1].refine_generations == 4
    assert orchestrator.attempts[-1].generations == 5
    assert seen[0] == ("direct", 0, False)
    assert any(name == "composition" and refining for name, _generation, refining in seen)
    assert orchestrator.library.load(first.key).payload["nodes"][-1]["id"] != 100
    assert orchestrator.search.incumbent_key(xor_task) == first.key

    snapshot = orchestrator.search.state_dict()
    # No caller retains a population object between tasks: the disk snapshot supplies it.
    second = orchestrator.solve(xor_task)
    assert second is not None and second.key == first.key
    assert orchestrator.attempts[-1].size_metrics["champion_expanded_complexity"] == solving_genome.complexity()
    assert len(seen) == 9
    assert max(generation for _name, generation, _refining in seen) >= 3
    assert sum(item["generations"] for item in orchestrator.attempts[-1].strategy_work.values()) == 4
    assert snapshot != orchestrator.search.state_dict()


def test_dormant_strategies_spend_no_credits_and_can_wake(tmp_path, monkeypatch, xor_task, solving_genome):
    class Session(StrategySession):
        def ready(self, runtime):
            return self.role == "direct" or bool(runtime.library.keys())

        def _advance(self, runtime):
            return StrategyResult(self.role, 1.0, 1, champion_genome=solving_genome.clone(), champion_metrics={"support_accuracy": 1.0})

    for name in ("direct", "composition"):
        monkeypatch.setitem(SESSION_STRATEGY._items, name, Session)
    orchestrator = _orchestrator(tmp_path, table=_policy())
    orchestrator.solve(xor_task)
    work = orchestrator.attempts[-1].strategy_work
    assert work["direct"]["generations"] >= 1
    assert work["composition"]["generations"] >= 1
    assert sum(row["generations"] for row in work.values()) == 5


def test_direct_population_round_trip_preserves_rng_species_and_novelty(tmp_path, xor_task):
    orchestrator = _orchestrator(tmp_path, table=_policy(evolve=["direct"]))
    strategy = dict(orchestrator.strategies)["direct"]
    runtime = orchestrator._runtime()
    spec = comp_task_spec(xor_task, include_query=False)
    continuous = DirectSession(strategy, xor_task, spec, seed=7)
    continuous.advance(runtime)
    saved = copy.deepcopy(continuous.state_dict())
    continuous.advance(runtime)
    expected = continuous.state_dict()
    restored = DirectSession(strategy, xor_task, spec, seed=7, saved=saved)
    restored.advance(runtime)
    assert restored.state_dict() == expected


def _multichannel_temporal_task(*, steps: int = 8, time_first: bool = False, sequence_output: bool = False) -> Task:
    """
    Four features per step, including the channel/time layout used by pole tasks.
    """
    samples = torch.rand(8, 4, steps, generator=torch.Generator().manual_seed(17))
    pairs: list[tuple[Field, Field]] = []
    for sample in samples:
        signal = sample.mean(dim=0)
        source = Field(sample.T if time_first else sample, (Axis.TIME, Axis.CHANNEL) if time_first else (Axis.CHANNEL, Axis.TIME), ValueType.CONTINUOUS, None, (0.0, 1.0), None)
        target = Field(signal if sequence_output else signal[-1:], (Axis.TIME,) if sequence_output else (Axis.CHANNEL,), ValueType.CONTINUOUS, None, (0.0, 1.0), None)
        pairs.append((source, target))
    return Task(TaskMeta(4, TaskKind.MAP, "multichannel_sequence", fixed_split=True), support=pairs[:6], query=pairs[6:])


@pytest.mark.parametrize("steps", [1, 8])
@pytest.mark.parametrize("time_first", [False, True])
@pytest.mark.parametrize("sequence_output", [False, True])
def test_multichannel_temporal_session_advances_and_resumes(tmp_path: Path, steps: int, time_first: bool, sequence_output: bool) -> None:
    task = _multichannel_temporal_task(steps=steps, time_first=time_first, sequence_output=sequence_output)
    orchestrator = _orchestrator(
        tmp_path,
        table=_policy(evolve=["direct"], direct={"pop_size": 4, "train": {"kind": "gradient", "steps": 2, "lr": 0.05, "writeback": True}}),
    )
    strategy = dict(orchestrator.strategies)["direct"]
    assert isinstance(strategy, DirectStrategy)
    original_init = strategy.evolver.init_op
    runtime = orchestrator._runtime()
    spec = comp_task_spec(task, include_query=False)
    session = DirectSession(strategy, task, spec, seed=7)
    session.advance(runtime)

    assert strategy.evolver.init_op is original_init
    assert isinstance(session.adapter, TemporalTaskAdapter)
    assert session.adapter.n_inputs == 4
    assert session.adapter.mode == ("all" if sequence_output else "last")
    assert session.adapter.encoded.support_input[0].shape == (6, steps, 4)
    assert session.adapter.encoded.query_input is session.adapter.encoded.query_target is None
    assert session.state is not None
    for member in session.state.population:
        assert len(member.genome.input_ids) == 4
        assert all(member.genome.nodes[node_id].coordinate is None for node_id in member.genome.input_ids)
    saved = copy.deepcopy(session.state_dict())

    result = session.advance(runtime)
    assert result.champion_genome is not None
    assert math.isfinite(result.champion_metrics["support_loss"])
    module = session.adapter.decode(result.champion_genome)
    assert module(session.adapter.encoded.support_input[0]).shape == (6, steps if sequence_output else 1)
    restored = DirectSession(strategy, task, spec, seed=7, saved=saved)
    restored.advance(runtime)
    assert restored.state_dict() == session.state_dict()
    assert strategy.evolver.init_op is original_init


def test_shared_direct_strategy_restores_initializer_between_spatial_and_temporal_tasks(tmp_path: Path) -> None:
    temporal = _multichannel_temporal_task()

    def spatial_pairs(pairs: list[tuple[Field, Field]]) -> list[tuple[Field, Field]]:
        return [(replace(source, data=source.data.unsqueeze(0), axes=(Axis.CHANNEL, Axis.HEIGHT, Axis.WIDTH)), target) for source, target in pairs]

    spatial = replace(temporal, meta=replace(temporal.meta, rung=15, name="spectrogram"), support=spatial_pairs(temporal.support), query=spatial_pairs(temporal.query))
    orchestrator = _orchestrator(
        tmp_path,
        table=_policy(evolve=["direct"], direct={"pop_size": 4, "train": {"kind": "gradient", "steps": 2, "lr": 0.05, "writeback": True}}),
    )
    strategy = dict(orchestrator.strategies)["direct"]
    assert isinstance(strategy, DirectStrategy)
    original_init = strategy.evolver.init_op
    snapshots = []
    for task in (spatial, temporal, spatial):
        session = DirectSession(strategy, task, comp_task_spec(task, include_query=False), seed=7)
        session.advance(orchestrator._runtime())
        assert strategy.evolver.init_op is original_init
        assert session.state is not None
        for member in session.state.population:
            coordinates = [member.genome.nodes[node_id].coordinate for node_id in member.genome.input_ids]
            assert coordinates == ([(0.0, float(channel), float(step)) for channel in range(4) for step in range(8)] if task is spatial else [None] * 4)
        snapshots.append(session.state_dict())
    assert snapshots[0] == snapshots[2]


def test_compact_native_parent_survives_population_selection(tmp_path, xor_task, solving_genome):
    from versal.evolution.genome import genome_to_dict
    from versal.strategy_direct import DirectStrategy

    orchestrator = _orchestrator(tmp_path, table=_policy(evolve=["direct"]))
    strategy = dict(orchestrator.strategies)["direct"]
    assert isinstance(strategy, DirectStrategy)
    runtime = orchestrator._runtime()
    spec = comp_task_spec(xor_task, include_query=False)
    session = DirectSession(strategy, xor_task, spec, seed=7)
    session.refining = True
    session.advance(runtime)
    assert session.state is not None
    compact = strategy.evolver.evaluate_only(solving_genome, session.adapter)
    session.state.population = [compact] * strategy.evolver.pop_size
    session._verified(runtime)
    larger = solving_genome.clone()
    larger.nodes[100] = NodeGene(100, NodeKind.HIDDEN, "tanh")
    # Selection can favor margin or novelty and discard the smaller perfect parent.
    session.state.population = [strategy.evolver.evaluate_only(larger, session.adapter)] * strategy.evolver.pop_size
    result = session._verified(runtime)
    assert result.champion_genome == solving_genome
    assert len(session.state.population) == strategy.evolver.pop_size
    assert any(item.genome == solving_genome for item in session.state.population)
    restored = DirectSession(strategy, xor_task, spec, seed=7, saved=session.state_dict())
    assert restored.protected is not None
    assert genome_to_dict(restored.protected.genome) == genome_to_dict(solving_genome)


def test_composition_freezes_scored_inner_weights_and_preserves_parent(tmp_path, xor_task, solving_genome):
    from versal.evolution.genome import genome_to_dict
    from versal.library import task_io
    from versal.strategy_sessions import CompositionSession, freeze_composition

    orchestrator = _orchestrator(tmp_path, table=_policy(evolve=["composition"]))
    strategy = dict(orchestrator.strategies)["composition"]
    runtime = orchestrator._runtime()
    spec = comp_task_spec(xor_task, include_query=False)
    session = CompositionSession(strategy, xor_task, spec, seed=7)
    key = orchestrator.library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={})
    session.offer(key)
    comp = session._seeds(runtime)[0]
    node_id = comp.module_ids[0]
    comp.nodes[node_id] = replace(comp.nodes[node_id], ref="live:0", trainable=True)
    runtime.state.species_champions[0] = solving_genome.clone()
    assessed = runtime.loop.assess_composition(comp, spec, runtime.state, train=False)
    assert assessed.net is not None
    with torch.no_grad():
        for parameter in assessed.net.inner_modules["live:0"].parameters():
            parameter.add_(0.1)
        expected = assessed.net(spec.encoded.support_input[0]).clone()
    frozen = freeze_composition(assessed, runtime, xor_task)
    shared = runtime.state.species_champions[0]
    shared.connections = [replace(connection, weight=connection.weight + 100.0) for connection in shared.connections]
    verified = runtime.loop.assess_composition(frozen.comp, spec, runtime.state, train=False)
    assert verified.net is not None
    with torch.no_grad():
        assert torch.equal(verified.net(spec.encoded.support_input[0]), expected)
    assert all(not node.ref.startswith("live:") for node in frozen.comp.nodes.values())
    session.protected = verified
    session.refining = True
    session.population = [assessed] * runtime.loop.comp_pop_size
    session._pin(runtime)
    assert len(session.population) == runtime.loop.comp_pop_size
    assert any(item.comp == verified.comp for item in session.population)
    restored = CompositionSession(strategy, xor_task, spec, seed=7, saved=session.state_dict())
    assert restored.protected is not None and restored.protected.comp == verified.comp


def test_field_population_continues_identically_after_restore(tmp_path):
    from tests.test_field import _task
    from versal.strategy_sessions import FieldSession

    task = _task()
    orchestrator = _orchestrator(
        tmp_path,
        table=_policy(evolve=["field"], field={"pop_size": 4, "train_sites": 8, "audit_sites": 8, "verify_top_k": 2, "train": {"kind": "gradient", "steps": 1}}),
    )
    strategy = dict(orchestrator.strategies)["field"]
    runtime = orchestrator._runtime()
    spec = comp_task_spec(task, include_query=False)
    session = FieldSession(strategy, task, spec, seed=7)
    assert session.ready(runtime)
    session.advance(runtime)
    saved = copy.deepcopy(session.state_dict())
    session.advance(runtime)
    restored = FieldSession(strategy, task, spec, seed=7, saved=saved)
    restored.advance(runtime)
    assert restored.state_dict() == session.state_dict()


def test_task_identity_uses_masks_and_contract_but_not_query(xor_task):
    original = task_content_fingerprint(xor_task)
    assert task_content_fingerprint(replace(xor_task, query=[])) == original
    assert task_content_fingerprint(replace(xor_task, meta=replace(xor_task.meta, name="renamed", rung=999))) == original
    source, target = xor_task.support[0]
    changed_mask = replace(source, mask=torch.zeros_like(source.data, dtype=torch.bool))
    assert task_content_fingerprint(replace(xor_task, support=[(changed_mask, target), *xor_task.support[1:]])) != original
    changed_range = replace(source, value_range=(0.0, 1.0))
    assert task_content_fingerprint(replace(xor_task, support=[(changed_range, target), *xor_task.support[1:]])) != original


def test_support_spec_never_reads_query_examples(xor_task):
    class InaccessibleQuery(list):
        def __iter__(self):
            raise AssertionError("query examples were read during support preparation")

    task = replace(xor_task, query=InaccessibleQuery(xor_task.query))
    spec = comp_task_spec(task, include_query=False)
    assert spec.encoded.query_input is None and spec.encoded.query_target is None


def test_search_snapshot_keeps_retired_population_dependencies_loadable(tmp_path, xor_task, solving_genome):
    from versal.evolution.genome import genome_to_dict
    from versal.library import task_io

    orchestrator = _orchestrator(tmp_path, table=_policy())
    key = orchestrator.library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={})
    state = TaskSearchState(sessions={"direct": {"pending": [key]}})
    assert orchestrator.search is not None
    orchestrator.search._save(orchestrator.search.identity(xor_task), state)
    orchestrator.library.retire(key)
    assert orchestrator.library.collect_garbage() == []
    assert ModuleLibrary(orchestrator.library.root).load(key).key == key


def test_smaller_candidate_cannot_trade_away_perfect_accuracy(tmp_path, solving_genome):
    orchestrator = _orchestrator(tmp_path, table=_policy())
    genome = solving_genome.clone()
    smaller = genome.clone()
    smaller.connections.pop()
    incumbent = StrategyResult("direct", 1.0, 1, champion_genome=genome, champion_metrics={"support_accuracy": 1.0})
    challenger = StrategyResult("direct", 0.999, 1, champion_genome=smaller, champion_metrics={"support_accuracy": 0.999})
    assert orchestrator.search is not None
    assert not orchestrator.search.improves(challenger, incumbent)


def test_composition_shrink_operators_preserve_parent_and_acyclicity():
    from versal.evolution.composition import (
        CompMutationContext,
        RefSpec,
        add_module_between,
        bypass_module,
        comp_topological_order,
        minimal_composition,
        remove_comp_edge,
        remove_module_node,
    )
    from versal.evolution.genome import InnovationTracker

    tracker = InnovationTracker(0)
    rng = random.Random(0)
    comp = minimal_composition([("x", 2)], "y", 1, tracker, rng)
    add_module_between(comp, RefSpec("live:0", 2, 1), comp.input_ids[0], comp.output_ids[0], tracker, rng)
    before = comp.clone()
    context = CompMutationContext(tracker, [])
    for operator in (remove_comp_edge, remove_module_node, bypass_module):
        child = operator(comp, context, rng=random.Random(0), prob=1.0)
        assert comp == before
        assert child.complexity() < comp.complexity()
        assert len(comp_topological_order(child)) == len(child.nodes)


def test_accepted_distillation_reports_its_own_support_and_query(tmp_path, xor_task):
    from versal.evolution.composition import CompositionGenome
    from versal.evolution.loop import AssessedComposition

    orchestrator = _orchestrator(tmp_path, table=_policy(accept_threshold=0.75))
    candidate = StrategyResult(
        "routed",
        0.8,
        1,
        champion_comp=AssessedComposition(CompositionGenome(), {"support_accuracy": 0.8}, 0.8, None),
        champion_metrics={"support_accuracy": 0.8},
        report_candidate_routed=object(),
        report_candidate_metrics={"support_accuracy": 1.0},
        report_metrics={"query_accuracy": 0.6, "query_loss": 0.4},
        validation_status="exhaustive",
    )
    assert orchestrator._quality_of_result(candidate)[:2] == (0.8, 0.6)


def test_router_snapshot_does_not_change_when_live_router_trains(tmp_path, monkeypatch, xor_task, solving_genome):
    from tests.test_routing import _seed_library, _task_view
    from versal.routing import RoutedTaskView, RouterService, frozen_router

    library = _seed_library(tmp_path, xor_task, solving_genome)
    service = RouterService(library, d_model=8, top_k=1, max_steps=1, persist_dir=tmp_path / "router", lazy_residency=True)
    service.sync()
    view, x, _width = _task_view(service.net, xor_task)
    with torch.no_grad():
        expected = view(x).clone()
    service.save()
    snapshot = frozen_router(service.net)
    accelerator_seeds = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", accelerator_seeds.append)
    before = torch.get_rng_state().clone()
    assert snapshot.shard_loader is not None
    snapshot.shard_loader("input", view.input_key)
    assert torch.equal(torch.get_rng_state(), before)
    assert accelerator_seeds == []
    with torch.no_grad():
        for parameter in service.net.parameters():
            parameter.add_(10.0)
    service.save()
    frozen = RoutedTaskView(snapshot, input_key=view.input_key, head_key=view.head_key, support_input=x)
    with torch.no_grad():
        assert torch.equal(frozen(x), expected)
        assert not torch.equal(view(x), expected)


def test_replay_round_trip_excludes_query(xor_task):
    from versal.strategy_sessions import replay_state, restore_replay

    spec = comp_task_spec(xor_task)
    encoded = spec.encoded
    data = replay_state([(encoded, "input", "head", encoded.support_input[0])])
    restored, input_key, head_key, support = restore_replay(data)[0]
    assert restored.query_input is restored.query_target is None
    assert torch.equal(restored.support_input[0], encoded.support_input[0])
    assert restored.support_target[2] == encoded.support_target[2]
    assert torch.equal(support, encoded.support_input[0])
    assert (input_key, head_key) == ("input", "head")


def test_task_boundary_resume_matches_uninterrupted_search(tmp_path):
    from versal.tools.xor_repro import run_seed
    from versal.utils.config import Config

    config = Config("configs/xor_repro.toml").current
    config["evolution"]["pop_size"] = 4
    config["evolution"]["modules"]["pop_size"] = 4
    config["evolution"]["composition"]["pop_size"] = 4
    config["evolution"]["train"]["steps"] = 2
    table = config["orchestrator"]
    table["max_depth"] = 0
    table["budgets"] = {"depth0": 8}
    table["refine"]["budget_k"] = 4
    table["direct"]["pop_size"] = 4
    table["direct"]["train"]["steps"] = 4
    table["routed"].update(d_model=8, max_steps=1, top_k=1, train_steps=4, generation_cost=4, replay_tasks=2)
    for directory, encounters, resume in [(tmp_path / "continuous", 4, False), (tmp_path / "resumed", 2, False), (tmp_path / "resumed", 4, True)]:
        run_seed(config, directory, seed=0, encounters=encounters, resume=resume)
    continuous = json.loads((tmp_path / "continuous" / "trajectory.json").read_text())
    resumed = json.loads((tmp_path / "resumed" / "trajectory.json").read_text())
    for rows in (continuous, resumed):
        for row in rows:
            for work in row["work"].values():
                work.pop("seconds")
                work.pop("preparation_seconds", None)
    assert continuous == resumed
    first = json.loads((tmp_path / "continuous" / "checkpoint.json").read_text())
    second = json.loads((tmp_path / "resumed" / "checkpoint.json").read_text())
    for key in ("rng", "torch_rng", "loop", "speciation", "search"):
        assert first[key] == second[key], key


def test_reproduction_refuses_missing_checkpoint_and_empty_experiment(tmp_path):
    from versal.tools.xor_repro import run_seed

    with pytest.raises(FileNotFoundError, match="no completed task boundary"):
        run_seed({}, tmp_path, seed=0, resume=True)
    with pytest.raises(ValueError, match="positive"):
        run_seed({}, tmp_path, seed=0, encounters=0)


def test_deduplication_allows_new_weights_and_remembers_actual_duplicates(tmp_path, solving_genome):
    from versal.evolution.genome import genome_to_dict
    from versal.interleaved import ExecutableTabuSession, SnapshotTabuStore

    records = {}
    library = ModuleLibrary(tmp_path / "library")
    session = ExecutableTabuSession(SnapshotTabuStore(records), "task", library)
    original = genome_to_dict(solving_genome)
    assert session.reserve(MODULE, original)
    rates_only = copy.deepcopy(original)
    rates_only["operator_rates"] = {"remove_connection": 0.9}
    assert not session.reserve(MODULE, rates_only)
    trained = solving_genome.clone()
    trained.connections[0] = replace(trained.connections[0], weight=trained.connections[0].weight + 0.5)
    assert session.reserve(MODULE, genome_to_dict(trained))
    session.commit()
    restored = ExecutableTabuSession(SnapshotTabuStore(json.loads(json.dumps(records))), "task", library)
    assert not restored.reserve(MODULE, original)
    assert not restored.reserve(MODULE, genome_to_dict(trained))


def test_composition_deduplication_tracks_changes_to_live_weights(tmp_path, solving_genome):
    from versal.evolution.composition import CompNodeGene, CompNodeKind, CompositionGenome, comp_to_dict
    from versal.interleaved import ExecutableTabuSession, SnapshotTabuStore
    from versal.library import COMPOSITION

    live = {0: solving_genome.clone()}
    session = ExecutableTabuSession(SnapshotTabuStore({}), "task", ModuleLibrary(tmp_path / "library"), live_genomes=lambda: live)
    comp = CompositionGenome({0: CompNodeGene(0, CompNodeKind.MODULE, "live:0", 2, 1)}, [])
    payload = comp_to_dict(comp)
    assert session.reserve(COMPOSITION, payload)
    assert not session.reserve(COMPOSITION, payload)
    live[0].connections[0] = replace(live[0].connections[0], weight=0.0)
    assert session.reserve(COMPOSITION, payload)


def test_composition_innovations_survive_a_fresh_shared_loop(tmp_path, xor_task):
    from versal.strategy_sessions import CompositionSession

    orchestrator = _orchestrator(tmp_path, table=_policy(evolve=["composition"]))
    runtime = orchestrator._runtime()
    strategy = dict(orchestrator.strategies)["composition"]
    spec = comp_task_spec(xor_task, include_query=False)
    session = CompositionSession(strategy, xor_task, spec, seed=7)
    shared_before = runtime.state.comp_innovations.to_dict()
    session.advance(runtime)
    saved = session.state_dict()
    assert runtime.state.comp_innovations.to_dict() == shared_before
    assert saved["innovations"]["next_node_id"] > shared_before["next_node_id"]
    restored = CompositionSession(strategy, xor_task, spec, seed=7, saved=saved)
    assert restored.innovations is not None
    assert restored.innovations.new_node_id() == saved["innovations"]["next_node_id"]
