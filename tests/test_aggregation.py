"""Product-aggregation nodes: exact semantics, gradient flow, serialization, and the mutator."""

import random

import torch

from versal.evaluation import evaluate, support_loss
from versal.evolution.evolver import TaskAdapter
from versal.evolution.genome import ConnectionGene, Genome, InnovationTracker, NodeGene, NodeKind, genome_from_dict, genome_to_dict
from versal.evolution.mutation import MutationContext, mutate_aggregation
from versal.evolution.train import gradient
from versal.substrate import decode


def _product_genome(product_weight: float = 1.0) -> Genome:
    """in0, in1 -> product hidden -> identity output (computes product_weight^2 * in0 * in1 * out_w)."""
    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity"),
        1: NodeGene(1, NodeKind.INPUT, "identity"),
        2: NodeGene(2, NodeKind.BIAS, "identity"),
        3: NodeGene(3, NodeKind.HIDDEN, "identity", None, "product"),
        4: NodeGene(4, NodeKind.OUTPUT, "identity"),
    }
    connections = [
        ConnectionGene(0, 3, product_weight, True, 0),
        ConnectionGene(1, 3, product_weight, True, 1),
        ConnectionGene(3, 4, 1.0, True, 2),
    ]
    return Genome(nodes=nodes, connections=connections)


def _product_xor_genome() -> Genome:
    """Exact XOR via a single product node: out = -1 + 2*x0 + 2*x1 - 4*(x0*x1)."""
    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity"),
        1: NodeGene(1, NodeKind.INPUT, "identity"),
        2: NodeGene(2, NodeKind.BIAS, "identity"),
        3: NodeGene(3, NodeKind.HIDDEN, "identity", None, "product"),
        4: NodeGene(4, NodeKind.OUTPUT, "identity"),
    }
    connections = [
        ConnectionGene(0, 3, 1.0, True, 0),
        ConnectionGene(1, 3, 1.0, True, 1),
        ConnectionGene(0, 4, 2.0, True, 2),
        ConnectionGene(1, 4, 2.0, True, 3),
        ConnectionGene(2, 4, -1.0, True, 4),
        ConnectionGene(3, 4, -4.0, True, 5),
    ]
    return Genome(nodes=nodes, connections=connections)


def test_product_node_computes_exact_product() -> None:
    module = decode(_product_genome(), n_inputs=2, n_outputs=1)
    x = torch.tensor([[2.0, 3.0], [0.5, 4.0], [-1.0, 5.0]])
    out = module(x)
    expected = torch.tensor([[6.0], [2.0], [-5.0]])
    assert torch.allclose(out, expected)


def test_product_node_respects_edge_weights() -> None:
    module = decode(_product_genome(product_weight=2.0), n_inputs=2, n_outputs=1)
    x = torch.tensor([[3.0, 5.0]])
    # (2*3) * (2*5) = 60
    assert torch.allclose(module(x), torch.tensor([[60.0]]))


def test_product_xor_genome_solves_xor(xor_adapter: TaskAdapter) -> None:
    module = xor_adapter.decode(_product_xor_genome())
    metrics = evaluate(module, xor_adapter.encoded, xor_adapter.encoder)
    assert metrics["query_accuracy"] == 1.0
    assert metrics["support_accuracy"] == 1.0


def test_gradient_flows_through_product_node(xor_adapter: TaskAdapter) -> None:
    genome = _product_xor_genome()
    # Detune the readout so there is something to learn back.
    genome.connections = [ConnectionGene(c.in_id, c.out_id, c.weight * 0.3, c.enabled, c.innovation) for c in genome.connections]
    module = xor_adapter.decode(genome)
    before = float(support_loss(module, xor_adapter.encoded).detach())
    gradient(genome, module, xor_adapter.encoded, rng=random.Random(0), steps=60, lr=0.05, writeback=False)
    after = float(support_loss(module, xor_adapter.encoded).detach())
    assert after < before


def test_aggregation_serialization_round_trip() -> None:
    genome = _product_xor_genome()
    rebuilt = genome_from_dict(genome_to_dict(genome))
    assert rebuilt.nodes[3].aggregation == "product"
    assert rebuilt.nodes[4].aggregation == "sum"


def test_legacy_genome_dict_defaults_to_sum() -> None:
    data = genome_to_dict(_product_genome())
    for node in data["nodes"]:
        node.pop("aggregation")
    rebuilt = genome_from_dict(data)
    assert all(node.aggregation == "sum" for node in rebuilt.nodes.values())


def _ctx(genome: Genome) -> MutationContext:
    return MutationContext(innovations=InnovationTracker.from_genomes([genome]), activations=["tanh", "identity"], default_activation="tanh")


def test_mutate_aggregation_flips_eligible_hidden_node() -> None:
    genome = _product_genome()
    genome.nodes[3] = NodeGene(3, NodeKind.HIDDEN, "identity", None, "sum")  # fan-in 2, eligible
    child = mutate_aggregation(genome, _ctx(genome), rng=random.Random(0), prob=1.0)
    assert child.nodes[3].aggregation == "product"
    # Non-hidden nodes are never touched.
    assert child.nodes[0].aggregation == "sum"
    assert child.nodes[4].aggregation == "sum"


def test_mutate_aggregation_skips_out_of_range_fan_in() -> None:
    genome = _product_genome()
    genome.nodes[3] = NodeGene(3, NodeKind.HIDDEN, "identity", None, "sum")
    genome.connections = [genome.connections[0], genome.connections[2]]  # fan-in 1: ineligible
    child = mutate_aggregation(genome, _ctx(genome), rng=random.Random(0), prob=1.0)
    assert child.nodes[3].aggregation == "sum"


def test_mutate_aggregation_always_allows_product_to_sum() -> None:
    genome = _product_genome()
    genome.connections = [genome.connections[0], genome.connections[2]]  # fan-in 1, but already product
    child = mutate_aggregation(genome, _ctx(genome), rng=random.Random(0), prob=1.0)
    assert child.nodes[3].aggregation == "sum"
