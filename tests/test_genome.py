import random

from ardevo.evolution.genome import InnovationTracker, NodeKind, topological_order, would_create_cycle
from ardevo.evolution.mutation import MutationContext, add_connection, add_node, add_rich_node, perturb_weights


def _context(genome) -> MutationContext:
    return MutationContext(InnovationTracker.from_genomes([genome]), ["tanh", "relu"], "tanh")


def test_add_node_splits_edge_and_disables_original(linear_genome):
    ctx = _context(linear_genome)
    before_hidden = len(linear_genome.hidden_ids)
    child = add_node(linear_genome, ctx, rng=random.Random(0), prob=1.0)

    assert len(child.hidden_ids) == before_hidden + 1
    disabled = [conn for conn in child.connections if not conn.enabled]
    assert len(disabled) == 1, "the split edge must be disabled"

    new_id = child.hidden_ids[0]
    into_new = [c for c in child.enabled_connections() if c.out_id == new_id]
    out_of_new = [c for c in child.enabled_connections() if c.in_id == new_id]
    assert len(into_new) == 1 and len(out_of_new) == 1
    assert into_new[0].weight == 1.0, "in -> new edge carries weight 1.0 (NEAT split)"


def test_add_rich_node_wires_multiple_inputs(linear_genome):
    ctx = _context(linear_genome)
    child = add_rich_node(linear_genome, ctx, rng=random.Random(0), prob=1.0, fan_in=2)

    assert len(child.hidden_ids) == 1
    hidden_id = child.hidden_ids[0]
    incoming = [conn for conn in child.enabled_connections() if conn.out_id == hidden_id]
    outgoing = [conn for conn in child.enabled_connections() if conn.in_id == hidden_id]
    assert len(incoming) == 2, "the new node is wired from fan_in sources"
    assert len(outgoing) == len(linear_genome.output_ids), "the new node feeds every output"
    topological_order(child)  # a clean return proves the graph is still a DAG


def test_add_connection_stays_acyclic(solving_genome):
    ctx = _context(solving_genome)
    child = add_connection(solving_genome, ctx, rng=random.Random(1), prob=1.0)
    # topological_order raises on a cycle; a clean return proves the graph is still a DAG.
    topological_order(child)


def test_would_create_cycle_detects_back_edge(solving_genome):
    # 5 is the output; 0 is an input that reaches it, so 5 -> 0 would close a loop.
    assert would_create_cycle(solving_genome, 5, 0) is True
    assert would_create_cycle(solving_genome, 0, 0) is True


def test_perturb_weights_changes_some_weights(linear_genome):
    ctx = _context(linear_genome)
    child = perturb_weights(linear_genome, ctx, rng=random.Random(2), prob=1.0, sigma=0.5)
    before = [c.weight for c in linear_genome.connections]
    after = [c.weight for c in child.connections]
    assert before != after
    # Mutation must not alter the original genome in place.
    assert [c.weight for c in linear_genome.connections] == before


def test_minimal_genome_has_no_hidden_nodes(linear_genome):
    assert linear_genome.hidden_ids == []
    assert linear_genome.complexity() == len(linear_genome.enabled_connections())
    assert all(node.kind is not NodeKind.HIDDEN for node in linear_genome.nodes.values())
