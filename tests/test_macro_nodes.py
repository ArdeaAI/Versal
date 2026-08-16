"""Macro nodes: whole library networks as single frozen units inside flat genomes."""

import random
from pathlib import Path

import pytest
import torch

from versal.evolution.genome import (
    ConnectionGene,
    Genome,
    InnovationTracker,
    MacroGene,
    NodeGene,
    NodeKind,
    genome_from_dict,
    genome_to_dict,
    would_create_cycle,
)
from versal.evolution.mutation import MutationContext, add_macro_node
from versal.library import MODULE, ModuleLibrary, graft, macro_resolver, module_level
from versal.substrate import decode, decode_recurrent, set_macro_resolver

_IO = {"inputs": [{"signature": "BINARY|K", "width": 2}], "output": {"signature": "BINARY|K", "width": 1}}


@pytest.fixture(autouse=True)
def _reset_default_resolver():
    yield
    set_macro_resolver(None)


def _library_with_xor(tmp_path: Path, solving_genome: Genome) -> tuple[ModuleLibrary, str]:
    library = ModuleLibrary(tmp_path / "lib")
    key = library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=_IO, provenance={"accepted_metric": 1.0, "weight_robustness": 0.8})
    return library, key


def _macro_host(key: str, readout_weight: float = 1.0) -> Genome:
    """2 inputs + bias -> macro(xor solver) -> output: host output equals the inner XOR logit."""
    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity"),
        1: NodeGene(1, NodeKind.INPUT, "identity"),
        2: NodeGene(2, NodeKind.BIAS, "identity"),
        3: NodeGene(3, NodeKind.OUTPUT, "identity"),
        4: NodeGene(4, NodeKind.HIDDEN, "identity"),  # the macro's output stub
    }
    connections = [ConnectionGene(4, 3, readout_weight, True, 0)]
    macros = [MacroGene(ref=f"library:{key}", input_node_ids=(0, 1), output_node_ids=(4,), innovation=100)]
    return Genome(nodes=nodes, connections=connections, macros=macros)


def test_macro_forward_equals_inner_network(tmp_path: Path, solving_genome: Genome) -> None:
    library, key = _library_with_xor(tmp_path, solving_genome)
    host = decode(_macro_host(key), 2, 1, macro_resolver=macro_resolver(library))
    inner = decode(solving_genome, 2, 1)
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    assert torch.allclose(host(x), inner(x))


def test_missing_resolver_raises(tmp_path: Path, solving_genome: Genome) -> None:
    _library, key = _library_with_xor(tmp_path, solving_genome)
    with pytest.raises(ValueError, match="macro resolver"):
        decode(_macro_host(key), 2, 1)


def test_default_resolver_seam(tmp_path: Path, solving_genome: Genome) -> None:
    library, key = _library_with_xor(tmp_path, solving_genome)
    set_macro_resolver(macro_resolver(library))
    module = decode(_macro_host(key), 2, 1)  # no explicit resolver: the default applies
    assert module(torch.tensor([[1.0, 0.0]])).shape == (1, 1)


def test_nested_macros_resolve_recursively(tmp_path: Path, solving_genome: Genome) -> None:
    library, key = _library_with_xor(tmp_path, solving_genome)
    middle = _macro_host(key)
    middle_key = library.add(entry_type=MODULE, payload=genome_to_dict(middle), io=_IO, provenance={}, level=module_level(middle, library))
    outer = decode(_macro_host(middle_key), 2, 1, macro_resolver=macro_resolver(library))
    inner = decode(solving_genome, 2, 1)
    x = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    assert torch.allclose(outer(x), inner(x))  # identity stubs and unit readouts all the way down
    assert library.load(middle_key).level == 2
    assert module_level(_macro_host(middle_key), library) == 3


def test_macro_inner_weights_are_frozen_through_training(tmp_path: Path, solving_genome: Genome, xor_adapter) -> None:
    from versal.evolution.train import gradient

    library, key = _library_with_xor(tmp_path, solving_genome)
    genome = _macro_host(key, readout_weight=0.5)
    module = decode(genome, 2, 1, macro_resolver=macro_resolver(library))
    inner_before = [parameter.detach().clone() for parameter in module._macro_inner.parameters()]
    host_before = module.export_weights()
    gradient(genome, module, xor_adapter.encoded, rng=random.Random(0), steps=20, lr=0.1, writeback=False)
    for parameter, before in zip(module._macro_inner.parameters(), inner_before):
        assert torch.equal(parameter.detach(), before)  # frozen inner untouched by Adam
    assert module.export_weights() != host_before  # the readout edge trained
    assert module.core() == (None, None)  # macro hosts are not batchable


def test_weight_samples_never_fill_macro_inners(tmp_path: Path, solving_genome: Genome, xor_adapter) -> None:
    from versal.evolution.evaluate import weight_samples

    library, key = _library_with_xor(tmp_path, solving_genome)
    genome = _macro_host(key)
    module = decode(genome, 2, 1, macro_resolver=macro_resolver(library))
    inner_before = [parameter.detach().clone() for parameter in module._macro_inner.parameters()]
    metrics = weight_samples(genome, module, xor_adapter)
    assert "weight_robustness" in metrics
    for parameter, before in zip(module._macro_inner.parameters(), inner_before):
        assert torch.equal(parameter.detach(), before)


def test_macro_depth_guard(tmp_path: Path, solving_genome: Genome) -> None:
    library, key = _library_with_xor(tmp_path, solving_genome)
    for _ in range(5):
        host = _macro_host(key)
        key = library.add(entry_type=MODULE, payload=genome_to_dict(host), io=_IO, provenance={}, level=module_level(host, library))
    with pytest.raises(ValueError, match="nesting"):
        decode(_macro_host(key), 2, 1, macro_resolver=macro_resolver(library))


def test_recurrent_host_with_macro_runs(tmp_path: Path, solving_genome: Genome) -> None:
    library, key = _library_with_xor(tmp_path, solving_genome)
    genome = _macro_host(key)
    genome.connections.append(ConnectionGene(4, 4, 0.5, True, 7, recurrent=True))
    module = decode_recurrent(genome, 2, 1, mode="last", macro_resolver=macro_resolver(library))
    out = module(torch.zeros(3, 4, 2))
    assert out.shape == (3, 1)


def test_serialization_round_trip_and_legacy(tmp_path: Path, solving_genome: Genome) -> None:
    _library, key = _library_with_xor(tmp_path, solving_genome)
    genome = _macro_host(key)
    rebuilt = genome_from_dict(genome_to_dict(genome))
    assert rebuilt.macros == genome.macros
    legacy = genome_to_dict(solving_genome)
    legacy.pop("macros")
    assert genome_from_dict(legacy).macros == []


def test_macro_implied_edges_guard_cycles(tmp_path: Path, solving_genome: Genome) -> None:
    _library, key = _library_with_xor(tmp_path, solving_genome)
    genome = _macro_host(key)
    genome.nodes[5] = NodeGene(5, NodeKind.HIDDEN, "tanh")
    genome.macros = [MacroGene(ref=f"library:{key}", input_node_ids=(0, 5), output_node_ids=(4,), innovation=100)]
    # 4 -> 5 would close a cycle THROUGH the macro (5 feeds the macro, the macro feeds 4).
    assert would_create_cycle(genome, 4, 5)
    assert not would_create_cycle(genome, 5, 3)


def test_add_macro_node_places_and_wires(tmp_path: Path, solving_genome: Genome, linear_genome: Genome) -> None:
    library, key = _library_with_xor(tmp_path, solving_genome)
    ctx = MutationContext(innovations=InnovationTracker.from_genomes([linear_genome]), activations=["tanh"], default_activation="tanh", library=library)
    child = add_macro_node(linear_genome, ctx, rng=random.Random(0), prob=1.0)
    assert len(child.macros) == 1
    macro = child.macros[0]
    assert macro.ref == f"library:{key}" and len(macro.input_node_ids) == 2 and len(macro.output_node_ids) == 1
    stub = macro.output_node_ids[0]
    assert child.nodes[stub].kind is NodeKind.HIDDEN and child.nodes[stub].activation == "identity"
    assert any(conn.in_id == stub and conn.out_id in child.output_ids for conn in child.connections)
    module = decode(child, 2, 1, macro_resolver=macro_resolver(library))
    assert module(torch.tensor([[1.0, 0.0]])).shape == (1, 1)
    # max_outputs filter: nothing fits, no-op.
    assert add_macro_node(linear_genome, ctx, rng=random.Random(0), prob=1.0, max_outputs=0).macros == []


def test_add_macro_node_samples_ctx_library_not_disk(tmp_path: Path, linear_genome: Genome) -> None:
    """Regression for the empty-run crash: with an EMPTY ctx.library, add_macro_node must add no
    macro (it samples ctx.library, never falling back to the on-disk "library" dir). Otherwise the
    direct population grafts a macro ref the decode-time resolver cannot satisfy -> a hard KeyError."""
    empty = ModuleLibrary(tmp_path / "empty_lib")
    ctx = MutationContext(innovations=InnovationTracker.from_genomes([linear_genome]), activations=["tanh"], default_activation="tanh", library=empty)
    child = add_macro_node(linear_genome, ctx, rng=random.Random(0), prob=1.0)
    assert child.macros == []


def test_add_macro_node_uses_genome_node_count_not_io_width(tmp_path: Path) -> None:
    """Regression: a TEMPORAL module is admitted with a FLATTENED io width that differs from its
    genome's per-step input/output NODE count. add_macro_node must size the placement from the node
    count (what decode validates), not the io width, or decode raises a shape mismatch mid-run."""
    inner = Genome(
        nodes={
            0: NodeGene(0, NodeKind.INPUT, "identity"),
            1: NodeGene(1, NodeKind.INPUT, "identity"),
            2: NodeGene(2, NodeKind.INPUT, "identity"),
            3: NodeGene(3, NodeKind.OUTPUT, "identity"),
            4: NodeGene(4, NodeKind.OUTPUT, "identity"),
        },
        connections=[ConnectionGene(0, 3, 1.0, True, 0), ConnectionGene(1, 3, 1.0, True, 1), ConnectionGene(2, 4, 1.0, True, 2)],
    )
    library = ModuleLibrary(tmp_path / "lib")
    flattened_io = {"inputs": [{"signature": "CONTINUOUS|C,T", "width": 5}], "output": {"signature": "CONTINUOUS|C", "width": 7}}  # 5/7 != genome 3/2
    key = library.add(entry_type=MODULE, payload=genome_to_dict(inner), io=flattened_io, provenance={})

    host = Genome(
        nodes={
            **{i: NodeGene(i, NodeKind.INPUT, "identity") for i in range(5)},
            5: NodeGene(5, NodeKind.BIAS, "identity"),
            6: NodeGene(6, NodeKind.OUTPUT, "identity"),
        },
        connections=[ConnectionGene(0, 6, 0.1, True, 0)],
    )
    ctx = MutationContext(innovations=InnovationTracker(_next_node_id=100), activations=["identity"], default_activation="identity", library=library)
    child = add_macro_node(host, ctx, rng=random.Random(0), prob=1.0)
    assert len(child.macros) == 1
    macro = child.macros[0]
    assert macro.ref == f"library:{key}"
    assert len(macro.input_node_ids) == 3 and len(macro.output_node_ids) == 2  # genome node counts, NOT io 5/7
    module = decode(child, 5, 1, macro_resolver=macro_resolver(library))  # no shape-mismatch ValueError
    assert module(torch.zeros((2, 5))).shape == (2, 1)


def test_mutators_never_target_macro_stubs(tmp_path: Path, solving_genome: Genome, linear_genome: Genome) -> None:
    from versal.evolution.mutation import add_connection, add_deep_node, add_recurrent_connection, mutate_activation, mutate_aggregation

    library, key = _library_with_xor(tmp_path, solving_genome)
    ctx = MutationContext(innovations=InnovationTracker.from_genomes([linear_genome]), activations=["tanh", "relu"], default_activation="tanh", library=library)
    genome = add_macro_node(linear_genome, ctx, rng=random.Random(0), prob=1.0)
    stubs = genome.macro_output_ids
    rng = random.Random(1)
    for _ in range(200):
        for operator in (add_connection, add_deep_node, add_recurrent_connection, mutate_activation, mutate_aggregation):
            genome = operator(genome, ctx, rng=rng, prob=1.0)
    assert all(conn.out_id not in stubs for conn in genome.connections)  # stubs are sources only
    for stub in stubs:
        assert genome.nodes[stub].activation == "identity" and genome.nodes[stub].aggregation == "sum"


def test_crossover_inherits_macros_as_units(tmp_path: Path, solving_genome: Genome, linear_genome: Genome) -> None:
    from versal.evolution.crossover import neat

    library, _key = _library_with_xor(tmp_path, solving_genome)
    ctx = MutationContext(innovations=InnovationTracker.from_genomes([linear_genome]), activations=["tanh"], default_activation="tanh", library=library)
    parent_a = add_macro_node(linear_genome, ctx, rng=random.Random(0), prob=1.0)
    child = neat(parent_a, linear_genome, rng=random.Random(0))
    assert child.macros == parent_a.macros
    child_of_plain = neat(linear_genome, parent_a, rng=random.Random(0))
    assert child_of_plain.macros == []  # disjoint macros from the less-fit parent drop


def test_speciation_sees_macro_markers(tmp_path: Path, solving_genome: Genome, linear_genome: Genome) -> None:
    from versal.evolution.speciation import compatibility_distance

    library, _key = _library_with_xor(tmp_path, solving_genome)
    ctx = MutationContext(innovations=InnovationTracker.from_genomes([linear_genome]), activations=["tanh"], default_activation="tanh", library=library)
    with_macro = add_macro_node(linear_genome, ctx, rng=random.Random(0), prob=1.0)
    no_macros = with_macro.clone()
    no_macros.macros = []
    coeffs = {"c_excess": 1.0, "c_disjoint": 1.0, "c_weight": 0.5}
    assert compatibility_distance(with_macro, no_macros, **coeffs, c_macro=1.0) > compatibility_distance(with_macro, no_macros, **coeffs, c_macro=0.0)


def test_graft_remaps_macros_with_fresh_markers(tmp_path: Path, solving_genome: Genome, linear_genome: Genome) -> None:
    library, _key = _library_with_xor(tmp_path, solving_genome)
    ctx = MutationContext(innovations=InnovationTracker.from_genomes([linear_genome]), activations=["tanh"], default_activation="tanh", library=library)
    host = add_macro_node(linear_genome, ctx, rng=random.Random(0), prob=1.0)
    host_key = library.add(entry_type=MODULE, payload=genome_to_dict(host), io=_IO, provenance={}, level=module_level(host, library))
    tracker = InnovationTracker(_next_node_id=500, _next_innovation=500)
    grafted = graft(library.load(host_key), tracker)
    assert len(grafted.macros) == 1
    macro = grafted.macros[0]
    assert all(node_id >= 500 for node_id in (*macro.input_node_ids, *macro.output_node_ids))
    assert macro.innovation >= 500  # fresh marker through the receiving tracker
    assert macro.ref == host.macros[0].ref  # library identity survives


def _macro_chain(library: ModuleLibrary, solving_genome: Genome, links: int) -> list[str]:
    """Admit a chain of module entries where entry N carries a macro ref to entry N-1."""
    keys: list[str] = []
    for link in range(links):
        payload = genome_to_dict(solving_genome)
        payload["connections"][0]["weight"] += link + 1  # unique key per link
        if keys:
            # A decode-valid placement: the inner (solving_genome) has 2 inputs / 1 output, so the
            # macro reads both host inputs and lands on one hidden node as its output stub.
            input_ids = [node["id"] for node in payload["nodes"] if node["kind"] == "input"]
            stub = next(node["id"] for node in payload["nodes"] if node["kind"] == "hidden")
            payload["macros"] = [{"ref": f"library:{keys[-1]}", "inputs": input_ids, "outputs": [stub], "innovation": 50 + link, "trainable": False}]
        keys.append(library.add(entry_type=MODULE, payload=payload, io=_IO, provenance={"accepted_metric": 1.0}))
    return keys


def test_macro_subtree_depth_walks_the_chain(tmp_path: Path, solving_genome: Genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    keys = _macro_chain(library, solving_genome, links=5)
    assert [library.macro_subtree_depth(key) for key in keys] == [0, 1, 2, 3, 4]
    assert library.macro_subtree_depth("m1_missing") == 999  # never a safe macro target


def test_configurable_macro_depth_counts_reference_boundaries_from_a_free_root(tmp_path: Path, solving_genome: Genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    keys = _macro_chain(library, solving_genome, links=5)  # top entry has four refs below it
    host = _macro_host(keys[-1])  # the host-to-top ref is the fifth boundary
    resolver = macro_resolver(library)

    with pytest.raises(ValueError, match="max_inline_depth=4"):
        decode(host, 2, 1, macro_resolver=resolver, max_inline_depth=4)
    assert decode(host, 2, 1, macro_resolver=resolver, max_inline_depth=5)(torch.zeros(1, 2)).shape == (1, 1)


def test_adapter_carries_depth_limit_through_main_and_worker_decode(tmp_path: Path, solving_genome: Genome, xor_adapter) -> None:
    from dataclasses import replace

    from tests.test_hierarchical_loop import _config as _loop_config
    from versal.evolution.evolver import _FLOOR_FITNESS, _assess_in_worker
    from versal.evolution.registry import build_evolver

    library = ModuleLibrary(tmp_path / "lib")
    keys = _macro_chain(library, solving_genome, links=5)
    host = _macro_host(keys[-1])
    set_macro_resolver(macro_resolver(library))
    evolver = build_evolver(_loop_config())
    shallow = replace(xor_adapter, max_inline_depth=4)
    deep = replace(xor_adapter, max_inline_depth=5)

    assert evolver.evaluate_only(host, shallow).fitness == _FLOOR_FITNESS
    assert evolver.evaluate_only(host, deep).module is not None
    _genome, shallow_metrics, shallow_fitness = _assess_in_worker(
        host,
        adapter=shallow,
        train_op=evolver.train_op,
        evaluate_op=evolver.evaluate_op,
        fitness=evolver.fitness,
    )
    _genome, deep_metrics, deep_fitness = _assess_in_worker(
        host,
        adapter=deep,
        train_op=evolver.train_op,
        evaluate_op=evolver.evaluate_op,
        fitness=evolver.fitness,
    )
    assert shallow_fitness == _FLOOR_FITNESS and shallow_metrics["decode_failed"] == 1.0
    assert deep_fitness > _FLOOR_FITNESS and "decode_failed" not in deep_metrics


def test_macro_cycle_detection_is_independent_of_depth_limit() -> None:
    loop = _macro_host("loop")
    with pytest.raises(ValueError, match="cycle"):
        decode(loop, 2, 1, macro_resolver=lambda _key: loop, max_inline_depth=100)


def test_add_macro_node_never_nests_past_the_decode_cap(tmp_path: Path, solving_genome: Genome, linear_genome: Genome) -> None:
    """The wall-ledger lesson: seed-then-embed cycles deepen the macro chain one level per attempt
    until decode dies at _MAX_MACRO_DEPTH. The mutation must refuse targets that would get there."""
    library = ModuleLibrary(tmp_path / "lib")
    keys = _macro_chain(library, solving_genome, links=5)  # depths 0..4; only depths <= 3 are embeddable
    ctx = MutationContext(innovations=InnovationTracker.from_genomes([linear_genome]), activations=["tanh"], default_activation="tanh", library=library)
    for seed in range(40):
        child = add_macro_node(linear_genome, ctx, rng=random.Random(seed), prob=1.0)
        assert child.macros and child.macros[0].ref != f"library:{keys[4]}"  # the depth-4 entry is never chosen
        decode(child, 2, 1, macro_resolver=macro_resolver(library))  # every proposed child DECODES


def test_add_macro_node_uses_configured_depth_limit(monkeypatch, tmp_path: Path, solving_genome: Genome, linear_genome: Genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    keys = _macro_chain(library, solving_genome, links=5)
    top = library.load(keys[-1])
    assert library.reference_subtree_depth(top.key) == 4
    monkeypatch.setattr(library, "query", lambda **_kwargs: [top])

    shallow = MutationContext(
        innovations=InnovationTracker.from_genomes([linear_genome]),
        activations=["tanh"],
        default_activation="tanh",
        library=library,
        max_inline_depth=4,
    )
    deep = MutationContext(
        innovations=InnovationTracker.from_genomes([linear_genome]),
        activations=["tanh"],
        default_activation="tanh",
        library=library,
        max_inline_depth=5,
    )
    assert add_macro_node(linear_genome, shallow, rng=random.Random(0), prob=1.0).macros == []
    child = add_macro_node(linear_genome, deep, rng=random.Random(0), prob=1.0)
    assert [macro.ref for macro in child.macros] == [f"library:{top.key}"]


def test_undecodable_genome_floors_instead_of_killing_the_run(tmp_path: Path, solving_genome: Genome, xor_adapter) -> None:
    """A corpse in the population is buried by assessment, never fatal: serial, pooled-worker, and
    champion-verification paths all floor (the two_spirals pool.map crash regression test)."""
    from tests.test_hierarchical_loop import _config as _loop_config
    from versal.evolution.evolver import _FLOOR_FITNESS, EvolverState, _assess_in_worker
    from versal.evolution.registry import build_evolver

    dead = solving_genome.clone()
    dead.macros.append(MacroGene(ref="library:m1_gone", input_node_ids=(0,), output_node_ids=(3,), innovation=99, trainable=False))

    evolver = build_evolver(_loop_config())
    state = EvolverState(population=[], innovations=InnovationTracker.from_genomes([dead]), rng=random.Random(0))
    floored = evolver.assess(dead, xor_adapter, state)
    assert floored.fitness == _FLOOR_FITNESS and floored.module is None and floored.metrics["decode_failed"] == 1.0
    assert evolver.evaluate_only(dead, xor_adapter).fitness == _FLOOR_FITNESS

    genome, metrics, fitness = _assess_in_worker(dead, adapter=xor_adapter, train_op=evolver.train_op, evaluate_op=evolver.evaluate_op, fitness=evolver.fitness)
    assert fitness == _FLOOR_FITNESS and metrics["decode_failed"] == 1.0

    mixed = evolver.assess_many([dead, solving_genome], xor_adapter, state)  # a corpse among the living
    assert mixed[0].fitness == _FLOOR_FITNESS and mixed[1].fitness > _FLOOR_FITNESS
