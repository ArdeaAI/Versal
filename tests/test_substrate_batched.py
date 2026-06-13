"""BatchedGraphNet: the padded population forward must equal each per-genome forward exactly."""

import torch

from ardevo.evolution.genome import ConnectionGene, Genome, NodeGene, NodeKind
from ardevo.substrate import decode
from ardevo.substrate_batched import BatchedGraphNet


def _mixed_activation_genome() -> Genome:
    """Depth-3 genome exercising relu/sigmoid/tanh grouping and hidden->hidden edges."""
    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity"),
        1: NodeGene(1, NodeKind.INPUT, "identity"),
        2: NodeGene(2, NodeKind.BIAS, "identity"),
        3: NodeGene(3, NodeKind.HIDDEN, "relu"),
        4: NodeGene(4, NodeKind.HIDDEN, "sigmoid"),
        5: NodeGene(5, NodeKind.HIDDEN, "tanh"),
        6: NodeGene(6, NodeKind.OUTPUT, "identity"),
    }
    connections = [
        ConnectionGene(0, 3, 0.7, True, 0),
        ConnectionGene(1, 3, -0.4, True, 1),
        ConnectionGene(2, 4, 0.3, True, 2),
        ConnectionGene(0, 4, 1.1, True, 3),
        ConnectionGene(3, 5, 0.9, True, 4),
        ConnectionGene(4, 5, -1.2, True, 5),
        ConnectionGene(5, 6, 1.5, True, 6),
        ConnectionGene(3, 6, 0.25, True, 7),
        ConnectionGene(2, 6, -0.1, True, 8),
    ]
    return Genome(nodes=nodes, connections=connections)


def test_batched_forward_matches_each_candidate(linear_genome: Genome, solving_genome: Genome) -> None:
    nets = [decode(linear_genome, 2, 1), decode(solving_genome, 2, 1), decode(_mixed_activation_genome(), 2, 1)]
    batched = BatchedGraphNet(nets)
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.3, -2.0]])
    out = batched(x)
    assert out.shape == (3, 5, 1)
    for index, net in enumerate(nets):
        assert torch.allclose(out[index], net(x), atol=1e-6), f"candidate {index} diverged"


def test_padding_never_leaks_across_candidates(linear_genome: Genome) -> None:
    big = _mixed_activation_genome()
    nets = [decode(linear_genome, 2, 1), decode(big, 2, 1)]  # n=4 padded up against n=7
    batched = BatchedGraphNet(nets)
    assert batched.n_max == 7
    x = torch.tensor([[5.0, -3.0]])
    out = batched(x)
    assert torch.allclose(out[0], nets[0](x), atol=1e-6)
    assert torch.allclose(out[1], nets[1](x), atol=1e-6)


def test_unstack_writes_trained_slices_back(solving_genome: Genome) -> None:
    nets = [decode(solving_genome, 2, 1), decode(solving_genome, 2, 1)]
    batched = BatchedGraphNet(nets)
    with torch.no_grad():
        batched.weights += 0.5
    batched.unstack_into(nets)
    x = torch.tensor([[1.0, 0.0]])
    fresh = decode(solving_genome, 2, 1)
    assert not torch.allclose(nets[0](x), fresh(x))
    assert torch.allclose(nets[0](x), nets[1](x))
    assert 0.0 < batched.pad_efficiency() <= 1.0
