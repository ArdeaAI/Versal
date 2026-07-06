"""Prune operators: remove_connection / remove_hidden_node, the structural SHRINK moves.

True gene deletion (not toggle-disable), so a free-growth lineage can climb back down in size.
Off-path byte-identity is structural: operators absent from `[evolution.mutation] operators` are
never constructed, which the last test pins.
"""

import random

import torch

from ardevo.evolution.crossover import neat
from ardevo.evolution.genome import ConnectionGene, Genome, InnovationTracker, MacroGene, NodeGene, NodeKind
from ardevo.evolution.mutation import MUTATION, AdaptiveMutationPipeline, MutationContext, remove_connection, remove_hidden_node
from ardevo.evolution.registry import build_evolver
from ardevo.substrate import decode, decode_recurrent
from tests.test_hierarchical_loop import _config as _loop_config
from tests.test_recurrence import _running_parity_genome


def _ctx() -> MutationContext:
    return MutationContext(innovations=InnovationTracker(_next_node_id=100), activations=["tanh"], default_activation="tanh")


def test_remove_connection_deletes_a_gene_and_leaves_the_parent_untouched(solving_genome: Genome) -> None:
    before = len(solving_genome.connections)
    child = remove_connection(solving_genome, _ctx(), rng=random.Random(0), prob=1.0)
    assert len(child.connections) == before - 1
    assert len(solving_genome.connections) == before  # clone semantics: the parent gene list is never mutated


def test_remove_connection_noop_paths_return_the_same_object(solving_genome: Genome) -> None:
    assert remove_connection(solving_genome, _ctx(), rng=random.Random(0), prob=0.0) is solving_genome
    edgeless = Genome(nodes={0: NodeGene(0, NodeKind.INPUT, "identity"), 1: NodeGene(1, NodeKind.OUTPUT, "identity")}, connections=[])
    assert remove_connection(edgeless, _ctx(), rng=random.Random(0), prob=1.0) is edgeless


def test_innovation_memo_survives_deletion() -> None:
    # Delete-then-re-add restores the ORIGINAL number: the memo is keyed by edge, not by gene
    # presence, so crossover alignment cannot drift when pruning and regrowth alternate.
    tracker = InnovationTracker(_next_node_id=10)
    first = tracker.innovation(3, 4)
    tracker.innovation(5, 6)  # unrelated allocation in between
    assert tracker.innovation(3, 4) == first


def test_remove_connection_children_always_decode(solving_genome: Genome) -> None:
    # Whatever edge the draw picks (including one that orphans a hidden node), the child must
    # decode and run forward: orphaned nodes are already exercised behavior under toggle-disable.
    for seed in range(20):
        child = remove_connection(solving_genome, _ctx(), rng=random.Random(seed), prob=1.0)
        output = decode(child, 2, 1)(torch.zeros(4, 2))
        assert output.shape == (4, 1) and torch.isfinite(output).all()


def test_remove_hidden_node_deletes_node_and_every_incident_gene(solving_genome: Genome) -> None:
    child = remove_hidden_node(solving_genome, _ctx(), rng=random.Random(0), prob=1.0)
    removed = (set(solving_genome.nodes) - set(child.nodes)).pop()
    assert solving_genome.nodes[removed].kind is NodeKind.HIDDEN
    assert all(conn.in_id != removed and conn.out_id != removed for conn in child.connections)
    assert child.input_ids == solving_genome.input_ids
    assert child.output_ids == solving_genome.output_ids
    assert child.bias_ids == solving_genome.bias_ids

    net = decode(child, 2, 1)
    output = net(torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
    assert torch.isfinite(output).all()
    output.sum().backward()  # the pruned graph must still be trainable end to end
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in net.parameters())


def test_remove_hidden_node_sweeps_recurrent_genes_too() -> None:
    genome = _running_parity_genome()
    for seed in range(10):
        child = remove_hidden_node(genome, _ctx(), rng=random.Random(seed), prob=1.0)
        removed_ids = set(genome.nodes) - set(child.nodes)
        if not removed_ids:
            continue  # this draw had no legal candidate; nothing to verify
        removed = removed_ids.pop()
        assert all(conn.in_id != removed and conn.out_id != removed for conn in child.connections)
        module = decode_recurrent(child, n_inputs=1, n_outputs=1, mode="last")
        assert torch.isfinite(module(torch.zeros(2, 3, 1))).all()


def test_remove_hidden_node_never_targets_macro_tied_nodes(solving_genome: Genome) -> None:
    # Node 4 is a macro output stub, node 5 feeds the macro (position-mapped at decode): both are
    # off-limits, and with no other hidden node the op must no-op and return the SAME object.
    host = Genome(
        nodes={
            0: NodeGene(0, NodeKind.INPUT, "identity"),
            1: NodeGene(1, NodeKind.INPUT, "identity"),
            2: NodeGene(2, NodeKind.BIAS, "identity"),
            3: NodeGene(3, NodeKind.OUTPUT, "identity"),
            4: NodeGene(4, NodeKind.HIDDEN, "identity"),
            5: NodeGene(5, NodeKind.HIDDEN, "tanh"),
        },
        connections=[ConnectionGene(4, 3, 1.0, True, 0), ConnectionGene(0, 5, 1.0, True, 1)],
        macros=[MacroGene(ref="library:m1_whatever", input_node_ids=(0, 5), output_node_ids=(4,), innovation=100)],
    )
    for seed in range(10):
        assert remove_hidden_node(host, _ctx(), rng=random.Random(seed), prob=1.0) is host


def test_pruned_parent_survives_neat_crossover(solving_genome: Genome) -> None:
    # parent_b still carries the deleted node's genes; they are disjoint against the pruned fitter
    # base and must drop, so every child connection endpoint exists and the child decodes.
    pruned = remove_hidden_node(solving_genome, _ctx(), rng=random.Random(3), prob=1.0)
    assert len(pruned.nodes) < len(solving_genome.nodes)
    for seed in range(10):
        child = neat(pruned, solving_genome, rng=random.Random(seed))
        assert all(conn.in_id in child.nodes and conn.out_id in child.nodes for conn in child.connections)
        assert torch.isfinite(decode(child, 2, 1)(torch.zeros(2, 2))).all()


def test_self_adaptive_pipeline_carries_prune_rates(solving_genome: Genome) -> None:
    specs = [
        ("remove_connection", MUTATION.get("remove_connection"), {"prob": 0.05}),
        ("remove_hidden_node", MUTATION.get("remove_hidden_node"), {"prob": 0.05}),
    ]
    child = AdaptiveMutationPipeline(specs)(solving_genome, _ctx(), rng=random.Random(0))
    assert set(child.operator_rates) == {"remove_connection", "remove_hidden_node"}
    grandchild = AdaptiveMutationPipeline(specs)(child, _ctx(), rng=random.Random(1))
    assert set(grandchild.operator_rates) == set(child.operator_rates)  # inherited, then re-perturbed


def test_operators_absent_from_config_are_never_constructed() -> None:
    from functools import partial

    from ardevo.evolution.mutation import MutationPipeline

    evolver = build_evolver(_loop_config())
    assert isinstance(evolver.mutation, MutationPipeline)
    names = [getattr(operator.func, "__name__", "") for operator in evolver.mutation.operators if isinstance(operator, partial)]
    assert names  # the pipeline is bound partials; an empty list would make the assertions below vacuous
    assert "remove_connection" not in names and "remove_hidden_node" not in names
