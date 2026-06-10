"""Geometry-biased mutation operators: locality from node coordinates, and back-compat no-ops."""

import random

from ardevo.evolution.genome import ConnectionGene, Genome, InnovationTracker, NodeGene, NodeKind, coordinate_distance
from ardevo.evolution.mutation import MutationContext, add_local_connection, add_local_node, add_shared_motif


def _line_genome() -> Genome:
    """Six input nodes on a 1-D line (coords 0..5) + bias + one output, fully wired to the output."""
    nodes: dict[int, NodeGene] = {index: NodeGene(index, NodeKind.INPUT, "identity", (float(index),)) for index in range(6)}
    nodes[6] = NodeGene(6, NodeKind.BIAS, "identity", None)
    nodes[7] = NodeGene(7, NodeKind.OUTPUT, "identity", None)
    connections = [ConnectionGene(source, 7, 0.1, True, source) for source in range(7)]
    return Genome(nodes=nodes, connections=connections)


def _context(genome: Genome) -> MutationContext:
    return MutationContext(innovations=InnovationTracker.from_genomes([genome]), activations=["tanh", "identity"], default_activation="tanh")


def test_coordinate_distance() -> None:
    assert coordinate_distance((0.0,), (3.0,)) == 3.0
    assert coordinate_distance((0.0, 0.0), (3.0, 4.0)) == 5.0
    assert coordinate_distance(None, (1.0,)) == float("inf")
    assert coordinate_distance((1.0,), (1.0, 2.0)) == float("inf")  # different banks are incomparable


def test_add_local_node_grows_a_local_receptive_field() -> None:
    genome = _line_genome()
    ctx = _context(genome)
    child = add_local_node(genome, ctx, rng=random.Random(0), prob=1.0, fan_in=3)

    new_ids = set(child.hidden_ids)
    assert len(new_ids) == 1
    new_id = new_ids.pop()
    sources = [conn.in_id for conn in child.enabled_connections() if conn.out_id == new_id]
    assert len(sources) == 3
    positions = [coordinate[0] for source in sources if (coordinate := child.nodes[source].coordinate) is not None]
    assert len(positions) == 3
    assert max(positions) - min(positions) <= 2.0  # three nearest on a line are contiguous
    assert child.nodes[new_id].coordinate is not None  # the new node sits at the field centroid


def test_add_shared_motif_replicates_a_detector() -> None:
    genome = add_local_node(_line_genome(), _context(_line_genome()), rng=random.Random(0), prob=1.0, fan_in=3)
    before = len(genome.hidden_ids)
    child = add_shared_motif(genome, _context(genome), rng=random.Random(1), prob=1.0, copies=2)
    assert len(child.hidden_ids) > before  # at least one replicated motif node was added


def test_geometry_ops_noop_without_coordinates(linear_genome: Genome) -> None:
    """Single-task genomes carry no coordinates, so geometry operators must leave them untouched."""
    ctx = _context(linear_genome)
    assert add_local_node(linear_genome, ctx, rng=random.Random(0), prob=1.0, fan_in=3).hidden_ids == []
    assert add_shared_motif(linear_genome, ctx, rng=random.Random(0), prob=1.0).hidden_ids == []
    assert len(add_local_connection(linear_genome, ctx, rng=random.Random(0), prob=1.0).connections) == len(linear_genome.connections)
