"""Recurrence: time-delayed edges, the stepped substrate, the mutator, and serialization."""

import random

import torch

from ardevo.evolution.genome import (
    ConnectionGene,
    Genome,
    InnovationTracker,
    NodeGene,
    NodeKind,
    genome_from_dict,
    genome_to_dict,
    make_acyclic,
    topological_order,
)
from ardevo.evolution.mutation import MutationContext, add_recurrent_connection
from ardevo.evolution.train import _writeback
from ardevo.substrate import decode, decode_recurrent


def _accumulator_genome() -> Genome:
    """input -> hidden (identity, self-loop w=1) -> output: the running sum of the sequence."""
    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity"),
        1: NodeGene(1, NodeKind.BIAS, "identity"),
        2: NodeGene(2, NodeKind.OUTPUT, "identity"),
        3: NodeGene(3, NodeKind.HIDDEN, "identity"),
    }
    connections = [
        ConnectionGene(0, 3, 1.0, True, 0),
        ConnectionGene(3, 2, 1.0, True, 1),
        ConnectionGene(3, 3, 1.0, True, 2, recurrent=True),
    ]
    return Genome(nodes=nodes, connections=connections)


def _running_parity_genome() -> Genome:
    """Exact running parity: h_t = x_t + h_prev - 2 * (x_t * h_prev), via a product node with one
    forward factor (x_t) and one RECURRENT factor (h_prev)."""
    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity"),
        1: NodeGene(1, NodeKind.BIAS, "identity"),
        2: NodeGene(2, NodeKind.OUTPUT, "identity"),
        3: NodeGene(3, NodeKind.HIDDEN, "identity", None, "product"),  # x_t * h_prev
        4: NodeGene(4, NodeKind.HIDDEN, "identity"),  # the parity accumulator h
    }
    connections = [
        ConnectionGene(0, 3, 1.0, True, 0),
        ConnectionGene(4, 3, 1.0, True, 1, recurrent=True),
        ConnectionGene(0, 4, 1.0, True, 2),
        ConnectionGene(3, 4, -2.0, True, 3),
        ConnectionGene(4, 4, 1.0, True, 4, recurrent=True),
        ConnectionGene(4, 2, 1.0, True, 5),
    ]
    return Genome(nodes=nodes, connections=connections)


def _sequence(bits: list[float]) -> torch.Tensor:
    return torch.tensor(bits, dtype=torch.float32).reshape(1, len(bits), 1)


def test_self_loop_accumulator_sums_the_sequence() -> None:
    module = decode_recurrent(_accumulator_genome(), n_inputs=1, n_outputs=1, mode="last")
    out = module(_sequence([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(out, torch.tensor([[10.0]]))


def test_running_parity_genome_is_exact() -> None:
    module = decode_recurrent(_running_parity_genome(), n_inputs=1, n_outputs=1, mode="last")
    for bits, parity in [([1.0, 0.0, 1.0, 1.0], 1.0), ([1.0, 1.0], 0.0), ([0.0, 0.0, 0.0], 0.0), ([1.0, 1.0, 1.0, 1.0, 1.0], 1.0)]:
        assert torch.allclose(module(_sequence(bits)), torch.tensor([[parity]])), bits


def test_seq_to_seq_mode_emits_every_step_t_major() -> None:
    module = decode_recurrent(_running_parity_genome(), n_inputs=1, n_outputs=1, mode="all")
    out = module(_sequence([1.0, 0.0, 1.0, 1.0]))
    assert out.shape == (1, 4)
    assert torch.allclose(out, torch.tensor([[1.0, 1.0, 0.0, 1.0]]))


def test_recurrent_genes_are_inert_under_plain_graphnet() -> None:
    with_recurrence = _accumulator_genome()
    without = Genome(nodes=dict(with_recurrence.nodes), connections=[c for c in with_recurrence.connections if not c.recurrent])
    x = torch.tensor([[3.0], [5.0]])
    assert torch.allclose(decode(with_recurrence, 1, 1)(x), decode(without, 1, 1)(x))


def test_gradient_flows_through_recurrent_weights() -> None:
    module = decode_recurrent(_accumulator_genome(), n_inputs=1, n_outputs=1, mode="last")
    out = module(_sequence([1.0, 1.0, 1.0]))
    out.sum().backward()
    grad = module.recurrent_weights.grad
    assert grad is not None
    # The self-loop weight (hidden -> hidden) must receive gradient through the unrolled steps.
    position = module._position[3]
    assert abs(float(grad[position, position])) > 0.0


def test_export_and_writeback_round_trip_recurrent_weights() -> None:
    genome = _accumulator_genome()
    module = decode_recurrent(genome, n_inputs=1, n_outputs=1, mode="last")
    with torch.no_grad():
        module.recurrent_weights += 0.25  # pretend training moved the self-loop weight
    tuned = module.export_weights()
    assert tuned[(3, 3, True)] == 1.25
    assert tuned[(0, 3, False)] == 1.0
    written = _writeback(genome, module)
    by_key = {(c.in_id, c.out_id, c.recurrent): c.weight for c in written.connections}
    assert by_key[(3, 3, True)] == 1.25
    assert by_key[(0, 3, False)] == 1.0


def test_topological_order_and_make_acyclic_ignore_recurrent_edges() -> None:
    genome = _running_parity_genome()
    order = topological_order(genome)  # would raise if recurrent self-loops counted as cycles
    assert set(order) == set(genome.nodes)
    repaired = make_acyclic(genome)
    recurrent = [c for c in repaired.connections if c.recurrent]
    assert len(recurrent) == 2 and all(c.enabled for c in recurrent)


def test_genome_serialization_round_trips_recurrent_flag() -> None:
    rebuilt = genome_from_dict(genome_to_dict(_running_parity_genome()))
    assert {(c.in_id, c.out_id) for c in rebuilt.recurrent_connections()} == {(4, 3), (4, 4)}


def test_tracker_distinguishes_recurrent_and_accepts_legacy() -> None:
    tracker = InnovationTracker(_next_node_id=10)
    forward = tracker.innovation(1, 2)
    recurrent = tracker.innovation(1, 2, recurrent=True)
    assert forward != recurrent
    assert tracker.innovation(1, 2, recurrent=True) == recurrent  # stable
    rebuilt = InnovationTracker.from_dict(tracker.to_dict())
    assert rebuilt.innovation(1, 2) == forward
    legacy = InnovationTracker.from_dict({"next_node_id": 5, "next_innovation": 1, "edge_innovations": [[1, 2, 0]]})
    assert legacy.innovation(1, 2) == 0  # 3-element entries load as forward edges


def test_add_recurrent_connection_adds_one_self_loop() -> None:
    genome = _accumulator_genome()
    genome.connections = [c for c in genome.connections if not c.recurrent]
    ctx = MutationContext(innovations=InnovationTracker.from_genomes([genome]), activations=["tanh"], default_activation="tanh")
    child = add_recurrent_connection(genome, ctx, rng=random.Random(0), prob=1.0, self_loop_bias=1.0)
    recurrent = child.recurrent_connections()
    assert len(recurrent) == 1 and recurrent[0].in_id == recurrent[0].out_id == 3
    # A second application cannot duplicate the same recurrent edge.
    again = add_recurrent_connection(child, ctx, rng=random.Random(1), prob=1.0, self_loop_bias=1.0)
    assert len(again.recurrent_connections()) == 1
