"""Macro nodes: whole library networks as single frozen units inside flat genomes."""

import random
from pathlib import Path

import pytest
import torch

from ardevo.evolution.genome import (
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
from ardevo.evolution.mutation import MutationContext, add_macro_node
from ardevo.library import MODULE, ModuleLibrary, graft, macro_resolver, module_level
from ardevo.substrate import decode, decode_recurrent, set_macro_resolver

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
    from ardevo.evolution.train import gradient

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
    from ardevo.evolution.evaluate import weight_samples

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


def test_mutators_never_target_macro_stubs(tmp_path: Path, solving_genome: Genome, linear_genome: Genome) -> None:
    from ardevo.evolution.mutation import add_connection, add_deep_node, add_recurrent_connection, mutate_activation, mutate_aggregation

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
    from ardevo.evolution.crossover import neat

    library, _key = _library_with_xor(tmp_path, solving_genome)
    ctx = MutationContext(innovations=InnovationTracker.from_genomes([linear_genome]), activations=["tanh"], default_activation="tanh", library=library)
    parent_a = add_macro_node(linear_genome, ctx, rng=random.Random(0), prob=1.0)
    child = neat(parent_a, linear_genome, rng=random.Random(0))
    assert child.macros == parent_a.macros
    child_of_plain = neat(linear_genome, parent_a, rng=random.Random(0))
    assert child_of_plain.macros == []  # disjoint macros from the less-fit parent drop


def test_speciation_sees_macro_markers(tmp_path: Path, solving_genome: Genome, linear_genome: Genome) -> None:
    from ardevo.evolution.speciation import compatibility_distance

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
