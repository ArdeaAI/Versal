"""Hierarchical loop: state seeding, end-to-end task runs, attribution, writeback, serialization."""

import random
from pathlib import Path

import pytest

from ardevo.dataset.icarus import Level0Encoder, Task, encode_task
from ardevo.evolution.composition import CompEdgeGene, CompNodeGene, CompNodeKind, CompositionGenome
from ardevo.evolution.evolver import Evolver
from ardevo.evolution.genome import ConnectionGene, Genome, NodeGene, NodeKind, genome_to_dict, topological_order
from ardevo.evolution.loop import AssessedComposition, CompTaskSpec, HierarchicalLoop, state_from_dict, state_to_dict
from ardevo.evolution.registry import build_loop
from ardevo.library import MODULE, ModuleLibrary, task_io


def _config() -> dict:
    return {
        "seed": 0,
        "evolution": {
            "loop": "hierarchical",
            "pop_size": 8,
            "elitism": 1,
            "init": {"kind": "minimal", "weight_scale": 1.0},
            "selection": {"kind": "tournament", "tournament_size": 3},
            "crossover": {"kind": "neat", "rate": 0.2},
            "mutation": {
                "operators": ["add_rich_node", "add_connection", "perturb_weights"],
                "add_rich_node_prob": 0.3,
                "add_rich_node_fan_in": 3,
                "add_connection_prob": 0.2,
                "perturb_weights_prob": 0.5,
                "perturb_weights_sigma": 0.4,
            },
            "train": {"kind": "gradient", "steps": 10, "lr": 0.05, "writeback": True},
            "speciation": {"kind": "neat", "threshold": 1.5, "target_species": 3},
            "composition": {
                "pop_size": 6,
                "elitism": 2,
                "selection": {"kind": "tournament", "tournament_size": 2},
                "crossover": {"kind": "comp_neat", "rate": 0.3},
                "mutation": {
                    "operators": ["add_module_node", "switch_ref", "add_comp_edge", "toggle_comp_edge", "perturb_glue"],
                    "add_module_node_prob": 0.4,
                    "switch_ref_prob": 0.1,
                    "add_comp_edge_prob": 0.2,
                    "toggle_comp_edge_prob": 0.05,
                    "perturb_glue_prob": 0.6,
                    "perturb_glue_sigma": 0.3,
                },
            },
            "modules": {"pop_size": 8, "elitism": 1, "in_ports": 4, "out_ports": 1, "advance_every": 2, "writeback": "champion", "attribution": "max", "decay": 0.9},
        },
        "fitness": {"components": ["support_accuracy", "hidden_penalty"], "w_support_accuracy": 1.0, "w_hidden_penalty": 0.01},
    }


def _spec(task: Task) -> CompTaskSpec:
    io = task_io(task)
    width = io["inputs"][0]["width"]
    signature = io["inputs"][0]["signature"]
    encoder = Level0Encoder(max_flat_dim=width)
    return CompTaskSpec(
        encoded=encode_task(task, encoder),
        encoder=encoder,
        n_inputs=width,
        input_specs=[(signature, width)],
        bank_columns={signature: list(range(width))},
        output_ref=task.meta.name,
        output_width=io["output"]["width"],
    )


def test_build_loop_resolves_kinds() -> None:
    default = build_loop({"evolution": {}, "fitness": {"components": []}})
    assert isinstance(default, HierarchicalLoop) and isinstance(default.evolver, Evolver)
    hierarchical = build_loop(_config())
    assert isinstance(hierarchical, HierarchicalLoop)
    for gone in ("flat", "nonsense"):
        with pytest.raises(KeyError):
            build_loop({"evolution": {"loop": gone}})


def test_fresh_state_seeds_modules_and_species() -> None:
    loop = build_loop(_config())
    state = loop.fresh_state(random.Random(0))
    assert len(state.modules) == 8
    assert state.species_champions  # at least one species exists and is referenceable
    catalog = loop.ref_catalog(state)
    assert all(spec.ref.startswith("live:") for spec in catalog)
    assert all(spec.in_width == 4 and spec.out_width == 1 for spec in catalog)


def test_run_task_end_to_end(decomposable_task: Task) -> None:
    loop = build_loop(_config())
    state = loop.fresh_state(random.Random(0))
    spec = _spec(decomposable_task)
    history: list[float] = []
    best = loop.run_task(spec, state, budget=3, on_generation=lambda gen, champ, mean: history.append(champ.fitness))
    assert best.net is not None and best.fitness > -1e8
    assert "support_accuracy" in best.metrics
    assert state.generation == 3 and len(history) == 3
    assert len(state.module_species_history) >= 2  # seed speciation + at least one module advance


def _cyclic_module_genome() -> Genome:
    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity"),
        1: NodeGene(1, NodeKind.INPUT, "identity"),
        2: NodeGene(2, NodeKind.INPUT, "identity"),
        3: NodeGene(3, NodeKind.INPUT, "identity"),
        4: NodeGene(4, NodeKind.BIAS, "identity"),
        5: NodeGene(5, NodeKind.OUTPUT, "identity"),
        6: NodeGene(6, NodeKind.HIDDEN, "tanh"),
        7: NodeGene(7, NodeKind.HIDDEN, "tanh"),
    }
    return Genome(
        nodes=nodes,
        connections=[
            ConnectionGene(0, 6, 1.0, True, 0),
            ConnectionGene(6, 7, 1.0, True, 1),
            ConnectionGene(7, 6, 1.0, True, 2),
            ConnectionGene(7, 5, 1.0, True, 3),
        ],
    )


def test_advance_modules_repairs_cyclic_offspring() -> None:
    loop = build_loop(_config())
    state = loop.fresh_state(random.Random(0))
    cyclic = _cyclic_module_genome()
    setattr(loop.evolver, "crossover_op", lambda parent_a, parent_b, *, rng: cyclic)
    setattr(loop.evolver, "mutation", lambda genome, ctx, *, rng: genome)

    loop.advance_modules(state)

    for module in state.modules:
        topological_order(module.genome)


def _live_comp(species_id: int, in_width: int, out_width: int) -> CompositionGenome:
    nodes = {
        0: CompNodeGene(0, CompNodeKind.INPUT, "BINARY|K", 0, in_width),
        1: CompNodeGene(1, CompNodeKind.MODULE, f"live:{species_id}", 4, 1),
        2: CompNodeGene(2, CompNodeKind.OUTPUT, "head", out_width, 0),
    }
    edges = [
        CompEdgeGene(0, 1, True, 0, tuple(0.5 for _ in range(in_width * 4))),
        CompEdgeGene(1, 2, True, 1, tuple(1.0 for _ in range(out_width))),
    ]
    return CompositionGenome(nodes=nodes, edges=edges)


def test_attribution_credits_champion_spares_referenced_decays_unreferenced() -> None:
    """The champion of a referenced species takes the attributed value; its non-champion members are
    SPARED (live stepping stones in an active species); only UNREFERENCED species decay."""
    loop = build_loop(_config())
    state = loop.fresh_state(random.Random(0))
    species_id = sorted(state.species_champions)[0]
    for module in state.modules:
        module.fitness = 0.5
    item = AssessedComposition(comp=_live_comp(species_id, 8, 2), metrics={}, fitness=0.7, net=None)
    loop._attribute([item], state)
    champion_index = state.species_champion_index[species_id]
    assert state.modules[champion_index].fitness == 0.7
    referenced_others = [index for index in state.species_members[species_id] if index != champion_index]
    assert all(abs(state.modules[index].fitness - 0.5) < 1e-9 for index in referenced_others)  # spared
    unreferenced = [index for sid, members in state.species_members.items() if sid != species_id for index in members]
    assert all(abs(state.modules[index].fitness - 0.45) < 1e-9 for index in unreferenced)  # 0.5 * decay


def test_decay_never_rewards_negative_fitness() -> None:
    """B5 regression: unreferenced modules with NEGATIVE scores must not drift toward neutral."""
    loop = build_loop(_config())
    state = loop.fresh_state(random.Random(0))
    species_id = sorted(state.species_champions)[0]
    for module in state.modules:
        module.fitness = -0.5
    item = AssessedComposition(comp=_live_comp(species_id, 8, 2), metrics={}, fitness=0.7, net=None)
    loop._attribute([item], state)
    champion_index = state.species_champion_index[species_id]
    assert state.modules[champion_index].fitness == 0.7
    others = [index for members in state.species_members.values() for index in members if index != champion_index]
    assert all(state.modules[index].fitness == -0.5 for index in others)  # NOT -0.475


def test_elites_with_dead_refs_are_repaired_not_floored(xor_task: Task) -> None:
    """B3 regression: an elite whose live species died must be re-pointed, not silently floored."""
    from ardevo.evolution.genome import InnovationTracker

    loop = build_loop(_config())
    state = loop.fresh_state(random.Random(0))
    state.comp_innovations = InnovationTracker(_next_node_id=100)  # hand-built comp uses ids 0-2
    spec = _spec(xor_task)
    dead_elite = loop._assess(_live_comp(9999, 2, 1), spec, state, train=False)
    assert dead_elite.fitness <= -1e8  # without repair, the dead ref floors it
    dead_elite = AssessedComposition(comp=_live_comp(9999, 2, 1), metrics={}, fitness=5.0, net=None)  # pretend it was the champion
    survivors = loop._reproduce_comps([dead_elite], spec, state)
    elite = survivors[0]
    assert elite.fitness > -1e8
    assert all(int(ref.removeprefix("live:")) in state.species_champions for ref in elite.comp.refs() if ref.startswith("live:"))


def test_attribution_bumps_library_stats(tmp_path: Path, solving_genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    io = {"inputs": [{"signature": "BINARY|K", "width": 2}], "output": {"signature": "BINARY|K", "width": 1}}
    key = library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=io, provenance={})
    loop = build_loop(_config())
    loop.attach_library(library)
    state = loop.fresh_state(random.Random(0))
    comp = _live_comp(sorted(state.species_champions)[0], 8, 2)
    comp.nodes[1] = CompNodeGene(1, CompNodeKind.MODULE, f"library:{key}", 4, 1)
    loop._attribute([AssessedComposition(comp=comp, metrics={}, fitness=0.6, net=None)], state)
    assert library.load(key).stats["use_count"] == 1


def test_module_writeback_flows_from_best_composition(xor_task: Task) -> None:
    import torch

    loop = build_loop(_config())
    state = loop.fresh_state(random.Random(0))
    species_id = sorted(state.species_champions)[0]
    spec = _spec(xor_task)
    item = loop._assess(_live_comp(species_id, 2, 1), spec, state, train=False)
    assert item.net is not None
    inner = item.net.inner_modules[f"live:{species_id}"]
    with torch.no_grad():
        for parameter in inner.parameters():
            parameter += 0.125
    before = {(c.in_id, c.out_id): c.weight for c in state.species_champions[species_id].connections if c.enabled}
    loop._module_writeback([item], state)
    after = {(c.in_id, c.out_id): c.weight for c in state.species_champions[species_id].connections if c.enabled}
    assert any(abs(after[k] - before[k] - 0.125) < 1e-6 for k in before)


def test_repair_refs_repoints_dead_species() -> None:
    loop = build_loop(_config())
    state = loop.fresh_state(random.Random(0))
    comp = _live_comp(9999, 8, 2)
    repaired = loop._repair_refs(comp, state)
    ref = repaired.nodes[1].ref
    assert int(ref.removeprefix("live:")) in state.species_champions
    assert state.repaired_refs == 1


def test_absorb_new_entries_grafts_over_worst_non_champions(tmp_path: Path, solving_genome) -> None:
    from ardevo.evolution.init import minimal
    from ardevo.library import MODULE, ModuleLibrary

    library = ModuleLibrary(tmp_path / "lib")
    module_genome = minimal(4, 1, rng=random.Random(3))
    io = {"inputs": [{"signature": "ANY", "width": 4}], "output": {"signature": "ANY", "width": 1}}
    key = library.add(entry_type=MODULE, payload=genome_to_dict(module_genome), io=io, provenance={"accepted_metric": 1.0, "weight_robustness": 0.9})

    config = _config()
    config["evolution"]["modules"]["absorb_top_k"] = 2
    loop = build_loop(config)
    loop.attach_library(library)
    state = loop.fresh_state(random.Random(0))
    for index, module in enumerate(state.modules):
        module.fitness = float(index)  # index 0 is the worst
    champions = set(state.species_champion_index.values())
    worst_replaceable = min((i for i in range(len(state.modules)) if i not in champions), key=lambda i: state.modules[i].fitness)

    absorbed = loop.absorb_new_entries(state)
    assert absorbed == 1 and state.absorbed_keys == [key]
    grafted = state.modules[worst_replaceable].genome
    assert sorted(c.weight for c in grafted.connections) == sorted(c.weight for c in module_genome.connections)
    # Idempotent: already-absorbed entries are never grafted twice.
    assert loop.absorb_new_entries(state) == 0


def test_mutation_context_sees_live_library_entries(tmp_path: Path, solving_genome) -> None:
    """The by-path cache snapshot goes stale; the live handle must see mid-run admissions."""
    from ardevo.evolution.mutation import add_library_module
    from ardevo.library import MODULE, ModuleLibrary

    config = _config()
    loop = build_loop(config)
    library = ModuleLibrary(tmp_path / "lib")
    loop.attach_library(library)
    state = loop.fresh_state(random.Random(0))
    ctx = loop._module_context(state)
    assert ctx.library is library

    host = state.modules[0].genome
    before = add_library_module(host, ctx, rng=random.Random(0), prob=1.0, path=str(tmp_path / "lib"))
    assert genome_to_dict(before) == genome_to_dict(host)  # empty library: no-op
    io = {"inputs": [{"signature": "BINARY|K", "width": 2}], "output": {"signature": "BINARY|K", "width": 1}}
    library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=io, provenance={})  # admitted MID-RUN
    after = add_library_module(host, ctx, rng=random.Random(0), prob=1.0, path=str(tmp_path / "lib"))
    assert len(after.nodes) > len(host.nodes)  # the live handle saw the new entry


def test_state_serialization_round_trip(decomposable_task: Task) -> None:
    loop = build_loop(_config())
    state = loop.fresh_state(random.Random(0))
    loop.run_task(_spec(decomposable_task), state, budget=2)
    data = state_to_dict(state)
    restored = state_from_dict(data, random.Random(1))
    assert restored.generation == state.generation
    assert [genome_to_dict(m.genome) for m in restored.modules] == [genome_to_dict(m.genome) for m in state.modules]
    assert restored.species_members == state.species_members
    assert {k: genome_to_dict(v) for k, v in restored.species_champions.items()} == {k: genome_to_dict(v) for k, v in state.species_champions.items()}
