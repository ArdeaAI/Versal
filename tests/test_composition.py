"""CompositionGenome: assembly semantics, weight sharing, recursion guards, operators, writeback."""

import json
import random
from pathlib import Path

import pytest
import torch

from ardevo.evolution.composition import (
    AssemblyContext,
    CompEdgeGene,
    CompMutationContext,
    CompNodeGene,
    CompNodeKind,
    ComposedNet,
    CompositionAssemblyError,
    CompositionGenome,
    RefSpec,
    add_comp_edge,
    add_module_node,
    assemble,
    comp_from_dict,
    comp_neat,
    comp_to_dict,
    comp_topological_order,
    minimal_composition,
    perturb_glue,
    switch_ref,
    toggle_comp_edge,
    writeback_composition,
)
from ardevo.evolution.genome import Genome, InnovationTracker, genome_to_dict
from ardevo.evolution.train import gradient
from ardevo.library import COMPOSITION, MODULE, ModuleLibrary
from ardevo.substrate import decode

_BANK = "BINARY|K"
_IO = {"inputs": [{"signature": _BANK, "width": 2}], "output": {"signature": _BANK, "width": 1}}


def _ctx(solving_genome: Genome | None = None, library: ModuleLibrary | None = None, **kwargs) -> AssemblyContext:
    resolver = (lambda species_id: solving_genome) if solving_genome is not None else None
    return AssemblyContext(bank_columns={_BANK: [0, 1]}, live_resolver=resolver, library=library, **kwargs)


def _module_comp(ref: str, glue_in: tuple[float, ...] = (1.0, 0.0, 0.0, 1.0), glue_out: tuple[float, ...] = (1.0,)) -> CompositionGenome:
    """INPUT(2) -> MODULE(ref, 2->1) -> OUTPUT(1), identity-ish glue."""
    nodes = {
        0: CompNodeGene(0, CompNodeKind.INPUT, _BANK, 0, 2),
        1: CompNodeGene(1, CompNodeKind.MODULE, ref, 2, 1),
        2: CompNodeGene(2, CompNodeKind.OUTPUT, "head", 1, 0),
    }
    edges = [CompEdgeGene(0, 1, True, 0, glue_in), CompEdgeGene(1, 2, True, 1, glue_out)]
    return CompositionGenome(nodes=nodes, edges=edges)


def test_minimal_composition_is_a_linear_readout() -> None:
    tracker = InnovationTracker(_next_node_id=0)
    comp = minimal_composition([(_BANK, 2)], "head", 1, tracker, random.Random(0))
    net = assemble(comp, _ctx(), n_inputs=2)
    x = torch.tensor([[1.0, 2.0], [0.5, -1.0]])
    glue_by_edge = {(e.in_id, e.out_id): torch.tensor(e.glue).reshape(-1, 1) for e in comp.edges}
    input_id, bias_id = comp.input_ids if comp.nodes[comp.input_ids[0]].ref == _BANK else reversed(comp.input_ids)
    expected = x @ glue_by_edge[(input_id, comp.output_ids[0])] + glue_by_edge[(bias_id, comp.output_ids[0])].reshape(1, 1)
    assert torch.allclose(net(x), expected)


def test_module_node_applies_inner_through_glue(solving_genome: Genome) -> None:
    net = assemble(_module_comp("live:7"), _ctx(solving_genome), n_inputs=2)
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    assert torch.allclose(net(x), decode(solving_genome, 2, 1)(x))


def test_repeated_ref_shares_one_inner_instance(solving_genome: Genome) -> None:
    nodes = {
        0: CompNodeGene(0, CompNodeKind.INPUT, _BANK, 0, 2),
        1: CompNodeGene(1, CompNodeKind.MODULE, "live:7", 2, 1),
        2: CompNodeGene(2, CompNodeKind.MODULE, "live:7", 2, 1),
        3: CompNodeGene(3, CompNodeKind.OUTPUT, "head", 1, 0),
    }
    edges = [
        CompEdgeGene(0, 1, True, 0, (1.0, 0.0, 0.0, 1.0)),
        CompEdgeGene(0, 2, True, 1, (0.0, 1.0, 1.0, 0.0)),
        CompEdgeGene(1, 3, True, 2, (1.0,)),
        CompEdgeGene(2, 3, True, 3, (1.0,)),
    ]
    net = assemble(CompositionGenome(nodes=nodes, edges=edges), _ctx(solving_genome), n_inputs=2)
    assert len(net.inner_modules) == 1  # one instance behind both nodes: literal weight sharing
    net(torch.tensor([[1.0, 0.0]])).sum().backward()
    inner = net.inner_modules["live:7"]
    grad = next(iter(inner.parameters())).grad
    assert grad is not None and float(grad.abs().sum()) > 0.0


def test_library_module_is_frozen_glue_still_trains(tmp_path: Path, solving_genome: Genome, xor_adapter) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    key = library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=_IO, provenance={})
    comp = _module_comp(f"library:{key}", glue_in=(0.8, 0.1, 0.1, 0.8), glue_out=(0.5,))
    net = assemble(comp, _ctx(library=library), n_inputs=2)
    inner_before = [parameter.detach().clone() for parameter in net.inner_modules[f"library:{key}"].parameters()]
    glue_before = {k: v.detach().clone() for k, v in net.glue.items()}
    gradient(comp, net, xor_adapter.encoded, rng=random.Random(0), steps=10, lr=0.05, writeback=False)
    for parameter, before in zip(net.inner_modules[f"library:{key}"].parameters(), inner_before):
        assert torch.equal(parameter.detach(), before)  # frozen inner stayed frozen
    assert any(not torch.equal(net.glue[k].detach(), glue_before[k]) for k in glue_before)  # glue trained


def test_nested_library_composition_inlines_transitively(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    tracker = InnovationTracker(_next_node_id=0)
    inner = minimal_composition([(_BANK, 2)], "head", 1, tracker, random.Random(3))
    key = library.add(entry_type=COMPOSITION, payload=comp_to_dict(inner), io=_IO, provenance={}, level=2)
    outer = _module_comp(f"library:{key}")
    net = assemble(outer, _ctx(library=library), n_inputs=2)
    x = torch.tensor([[1.0, -2.0], [0.25, 4.0]])
    inner_net = assemble(inner, _ctx(), n_inputs=2)
    assert torch.allclose(net(x), inner_net(x))


def test_self_referencing_entry_raises(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    key = "c2_selfref"
    payload = comp_to_dict(_module_comp(f"library:{key}"))
    record = {"key": key, "entry_type": COMPOSITION, "level": 2, "io": _IO, "payload": payload, "weights_frozen": True, "provenance": {}, "stats": {}}
    (tmp_path / "lib" / "entries").mkdir(parents=True)
    (tmp_path / "lib" / "entries" / f"{key}.json").write_text(json.dumps(record))
    with pytest.raises(CompositionAssemblyError, match="cycle"):
        assemble(_module_comp(f"library:{key}"), _ctx(library=library), n_inputs=2)


def test_nesting_depth_guard(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    tracker = InnovationTracker(_next_node_id=0)
    key = library.add(entry_type=COMPOSITION, payload=comp_to_dict(minimal_composition([(_BANK, 2)], "head", 1, tracker, random.Random(0))), io=_IO, provenance={}, level=2)
    for level in (3, 4):
        key = library.add(entry_type=COMPOSITION, payload=comp_to_dict(_module_comp(f"library:{key}")), io=_IO, provenance={}, level=level)
    top = _module_comp(f"library:{key}")
    assert assemble(top, _ctx(library=library), n_inputs=2)(torch.tensor([[1.0, 1.0]])).shape == (1, 1)
    with pytest.raises(CompositionAssemblyError, match="max_inline_depth"):
        assemble(top, _ctx(library=library, max_inline_depth=2), n_inputs=2)


def test_serialization_round_trip(solving_genome: Genome) -> None:
    comp = _module_comp("live:7")
    assert comp_from_dict(comp_to_dict(comp)) == comp


def test_comp_neat_aligns_by_innovation() -> None:
    tracker = InnovationTracker(_next_node_id=0)
    parent_a = minimal_composition([(_BANK, 2)], "head", 1, tracker, random.Random(1))
    parent_b = parent_a.clone()
    perturbed = perturb_glue(parent_b, CompMutationContext(tracker, []), rng=random.Random(2), prob=1.0, sigma=1.0)
    child = comp_neat(parent_a, perturbed, rng=random.Random(0))
    assert len(child.edges) == len(parent_a.edges)
    for edge in child.edges:
        source_a = next(e for e in parent_a.edges if e.innovation == edge.innovation)
        source_b = next(e for e in perturbed.edges if e.innovation == edge.innovation)
        assert edge.glue in (source_a.glue, source_b.glue)  # whole glue vectors, never mixed


def test_add_module_node_grows_and_stays_assemblable(solving_genome: Genome) -> None:
    tracker = InnovationTracker(_next_node_id=0)
    comp = minimal_composition([(_BANK, 2)], "head", 1, tracker, random.Random(0))
    ctx = CompMutationContext(innovations=tracker, ref_catalog=[RefSpec("live:7", 2, 1)])
    child = add_module_node(comp, ctx, rng=random.Random(0), prob=1.0)
    assert len(child.module_ids) == 1
    assert len(child.edges) == len(comp.edges) + 2
    comp_topological_order(child)  # raises on a cycle
    net = assemble(child, _ctx(solving_genome), n_inputs=2)
    assert net(torch.tensor([[1.0, 0.0]])).shape == (1, 1)


def test_switch_ref_preserves_glue_for_same_shape(solving_genome: Genome) -> None:
    comp = _module_comp("live:7")
    ctx = CompMutationContext(innovations=InnovationTracker(_next_node_id=10), ref_catalog=[RefSpec("live:7", 2, 1), RefSpec("live:9", 2, 1), RefSpec("live:8", 3, 1)])
    child = switch_ref(comp, ctx, rng=random.Random(0), prob=1.0)
    assert child.nodes[1].ref == "live:9"  # only the same-shape alternative is eligible
    assert child.edges == comp.edges  # glue untouched


def test_add_and_toggle_comp_edges() -> None:
    tracker = InnovationTracker(_next_node_id=0)
    comp = minimal_composition([(_BANK, 2)], "head", 1, tracker, random.Random(0))
    ctx = CompMutationContext(innovations=tracker, ref_catalog=[])
    grown = add_comp_edge(comp, ctx, rng=random.Random(0), prob=1.0)
    assert grown == comp  # input->output already saturated, nothing to add
    toggled = toggle_comp_edge(comp, ctx, rng=random.Random(0), prob=1.0)
    assert sum(1 for e in toggled.edges if e.enabled) == len(comp.edges) - 1


def test_writeback_composition_copies_trained_glue() -> None:
    tracker = InnovationTracker(_next_node_id=0)
    comp = minimal_composition([(_BANK, 2)], "head", 1, tracker, random.Random(0))
    net = assemble(comp, _ctx(), n_inputs=2)
    with torch.no_grad():
        for parameter in net.glue.values():
            parameter += 1.0
    written = writeback_composition(comp, net)
    for before, after in zip(comp.edges, written.edges):
        assert all(abs(b + 1.0 - a) < 1e-6 for b, a in zip(before.glue, after.glue))


def test_output_node_must_be_unique() -> None:
    comp = _module_comp("live:7")
    comp.nodes[9] = CompNodeGene(9, CompNodeKind.OUTPUT, "head2", 1, 0)
    with pytest.raises(CompositionAssemblyError, match="exactly one OUTPUT"):
        ComposedNet(comp, _ctx(), n_inputs=2)
