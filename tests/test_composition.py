"""CompositionGenome: assembly semantics, weight sharing, recursion guards, operators, writeback."""

import json
import random
from pathlib import Path

import pytest
import torch

from ardevo.evaluation import support_loss
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
    glue_value_count,
    minimal_composition,
    perturb_glue,
    switch_ref,
    toggle_comp_edge,
    writeback_composition,
)
from ardevo.evolution.genome import ConnectionGene, Genome, InnovationTracker, NodeGene, NodeKind, genome_to_dict
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


def _module_chain(library: ModuleLibrary, solving_genome: Genome, links: int) -> list[str]:
    keys: list[str] = []
    for link in range(links):
        payload = genome_to_dict(solving_genome)
        payload["connections"][0]["weight"] += link + 1  # keep every immutable entry key distinct
        if keys:
            input_ids = [node["id"] for node in payload["nodes"] if node["kind"] == "input"]
            stub = next(node["id"] for node in payload["nodes"] if node["kind"] == "hidden")
            payload["macros"] = [{"ref": f"library:{keys[-1]}", "inputs": input_ids, "outputs": [stub], "innovation": 50 + link, "trainable": False}]
        keys.append(library.add(entry_type=MODULE, payload=payload, io=_IO, provenance={}))
    return keys


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


def test_live_module_with_cycle_is_repaired_during_assembly() -> None:
    cyclic = Genome(
        nodes={
            0: NodeGene(0, NodeKind.INPUT, "identity"),
            1: NodeGene(1, NodeKind.INPUT, "identity"),
            2: NodeGene(2, NodeKind.BIAS, "identity"),
            3: NodeGene(3, NodeKind.OUTPUT, "identity"),
            4: NodeGene(4, NodeKind.HIDDEN, "tanh"),
            5: NodeGene(5, NodeKind.HIDDEN, "tanh"),
        },
        connections=[
            ConnectionGene(0, 4, 1.0, True, 0),
            ConnectionGene(4, 5, 1.0, True, 1),
            ConnectionGene(5, 4, 1.0, True, 2),
            ConnectionGene(5, 3, 1.0, True, 3),
        ],
    )
    net = assemble(_module_comp("live:7"), _ctx(cyclic), n_inputs=2)
    assert net(torch.zeros(2, 2)).shape == (2, 1)


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


def test_reference_depth_does_not_reset_at_composition_to_module_boundary(tmp_path: Path, solving_genome: Genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    keys = _module_chain(library, solving_genome, links=5)  # four module-to-module refs
    root = _module_comp(f"library:{keys[-1]}")  # composition-to-module is the fifth ref

    with pytest.raises(CompositionAssemblyError, match="max_inline_depth=4"):
        assemble(root, _ctx(library=library, max_inline_depth=4), n_inputs=2)
    assert assemble(root, _ctx(library=library, max_inline_depth=5), n_inputs=2)(torch.zeros(1, 2)).shape == (1, 1)


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


def test_gradient_skips_composition_with_disconnected_output(solving_genome: Genome, xor_adapter) -> None:
    comp = _module_comp("live:7")
    comp.edges = [comp.edges[0], CompEdgeGene(1, 2, False, 1, comp.edges[1].glue)]
    net = assemble(comp, _ctx(solving_genome), n_inputs=2)

    before_loss = float(support_loss(net, xor_adapter.encoded).detach())
    before_glue = {key: value.detach().clone() for key, value in net.glue.items()}

    gradient(comp, net, xor_adapter.encoded, rng=random.Random(0), steps=5, lr=0.05, writeback=False)

    assert float(support_loss(net, xor_adapter.encoded).detach()) == before_loss
    assert all(torch.equal(net.glue[key].detach(), value) for key, value in before_glue.items())


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


def test_factored_glue_matches_dense_equivalent() -> None:
    """A rank-r edge whose U @ V equals a dense matrix must compute identically."""
    import torch as torch_module

    rng = random.Random(5)
    in_width, out_width, rank = 4, 3, 2
    u = [[rng.gauss(0.0, 0.5) for _ in range(rank)] for _ in range(in_width)]
    v = [[rng.gauss(0.0, 0.5) for _ in range(out_width)] for _ in range(rank)]
    dense_matrix = (torch_module.tensor(u) @ torch_module.tensor(v)).reshape(-1)
    nodes = {
        0: CompNodeGene(0, CompNodeKind.INPUT, _BANK, 0, in_width),
        1: CompNodeGene(1, CompNodeKind.OUTPUT, "head", out_width, 0),
    }
    dense_comp = CompositionGenome(nodes=dict(nodes), edges=[CompEdgeGene(0, 1, True, 0, tuple(float(x) for x in dense_matrix.tolist()))])
    factored_glue = tuple(value for row in u for value in row) + tuple(value for row in v for value in row)
    factored_comp = CompositionGenome(nodes=dict(nodes), edges=[CompEdgeGene(0, 1, True, 0, factored_glue, glue_rank=rank)])
    ctx = AssemblyContext(bank_columns={_BANK: list(range(in_width))})
    x = torch_module.randn(5, in_width)
    assert torch_module.allclose(
        assemble(dense_comp, ctx, in_width)(x), assemble(factored_comp, AssemblyContext(bank_columns={_BANK: list(range(in_width))}), in_width)(x), atol=1e-6
    )


def test_factored_glue_writeback_round_trip() -> None:
    from ardevo.evolution.composition import _glue_for

    rng = random.Random(0)
    glue, rank = _glue_for(6, 5, rng, glue_rank=2, glue_rank_threshold=10)
    assert rank == 2 and len(glue) == 6 * 2 + 2 * 5
    nodes = {
        0: CompNodeGene(0, CompNodeKind.INPUT, _BANK, 0, 6),
        1: CompNodeGene(1, CompNodeKind.OUTPUT, "head", 5, 0),
    }
    comp = CompositionGenome(nodes=nodes, edges=[CompEdgeGene(0, 1, True, 0, glue, glue_rank=rank)])
    net = assemble(comp, AssemblyContext(bank_columns={_BANK: list(range(6))}), n_inputs=6)
    with torch.no_grad():
        for parameter in net.glue_u.values():
            parameter += 0.5
    written = writeback_composition(comp, net)
    assert written.edges[0].glue_rank == 2 and len(written.edges[0].glue) == len(glue)
    renet = assemble(written, AssemblyContext(bank_columns={_BANK: list(range(6))}), n_inputs=6)
    x = torch.randn(3, 6)
    assert torch.allclose(renet(x), net(x), atol=1e-6)  # the trained factors round-tripped


def test_glue_for_auto_select_threshold() -> None:
    from ardevo.evolution.composition import _glue_for

    rng = random.Random(0)
    _glue, rank = _glue_for(3, 2, rng, glue_rank=2, glue_rank_threshold=10)
    assert rank == 0  # 6 <= threshold: dense
    _glue, rank = _glue_for(2, 2, rng, glue_rank=4, glue_rank_threshold=1)
    assert rank == 0  # rank must stay below min(in, out) to actually compress
    comp = minimal_composition([(_BANK, 8)], "head", 8, InnovationTracker(_next_node_id=0), rng, glue_rank=2, glue_rank_threshold=16)
    bank_edge = next(edge for edge in comp.edges if comp.nodes[edge.in_id].ref == _BANK)
    bias_edge = next(edge for edge in comp.edges if comp.nodes[edge.in_id].ref == "__bias__")
    assert bank_edge.glue_rank == 2  # 64 > 16: factored
    assert bias_edge.glue_rank == 0  # 8 <= 16: dense


def test_glue_value_count_matches_dense_and_factored_representations() -> None:
    assert glue_value_count(3, 2) == 6
    assert glue_value_count(3, 2, glue_rank=2, glue_rank_threshold=10) == 6  # below the factoring threshold
    assert glue_value_count(8, 6, glue_rank=2, glue_rank_threshold=16) == 8 * 2 + 2 * 6
    assert glue_value_count(2, 2, glue_rank=4, glue_rank_threshold=1) == 4  # an over-wide rank stays dense
    psicov_seed = 245_025 + glue_value_count(13_966_425, 245_025, glue_rank=8, glue_rank_threshold=4096)
    assert psicov_seed == 113_936_625


def test_comp_neat_never_mixes_glue_ranks() -> None:
    rng = random.Random(0)
    tracker = InnovationTracker(_next_node_id=0)
    parent_a = minimal_composition([(_BANK, 8)], "head", 8, tracker, rng, glue_rank=2, glue_rank_threshold=16)
    parent_b = parent_a.clone()
    from dataclasses import replace as gene_replace

    from ardevo.evolution.composition import _glue_init

    parent_b.edges = [gene_replace(edge, glue=_glue_init(8, 8, rng), glue_rank=0) if edge.glue_rank else edge for edge in parent_b.edges]
    child = comp_neat(parent_a, parent_b, rng=random.Random(1))
    for edge_a, edge_child in zip(parent_a.edges, child.edges):
        if edge_a.glue_rank:
            assert edge_child.glue == edge_a.glue and edge_child.glue_rank == edge_a.glue_rank  # rank mismatch: parent_a wins


def test_glue_rank_serialization_and_legacy() -> None:
    rng = random.Random(0)
    comp = minimal_composition([(_BANK, 8)], "head", 8, InnovationTracker(_next_node_id=0), rng, glue_rank=2, glue_rank_threshold=16)
    rebuilt = comp_from_dict(comp_to_dict(comp))
    assert rebuilt == comp
    legacy = comp_to_dict(comp)
    for edge in legacy["edges"]:
        edge.pop("glue_rank")
    assert all(edge.glue_rank == 0 for edge in comp_from_dict(legacy).edges)  # old entries load dense


def test_output_node_must_be_unique() -> None:
    comp = _module_comp("live:7")
    comp.nodes[9] = CompNodeGene(9, CompNodeKind.OUTPUT, "head2", 1, 0)
    with pytest.raises(CompositionAssemblyError, match="exactly one OUTPUT"):
        ComposedNet(comp, _ctx(), n_inputs=2)
