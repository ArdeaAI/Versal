import torch

from ardevo.substrate import decode


def test_decode_forward_shape(linear_genome):
    module = decode(linear_genome, n_inputs=2, n_outputs=1)
    batch = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    out = module(batch)
    assert out.shape == (4, 1)


def test_weights_are_trainable_parameters(solving_genome):
    module = decode(solving_genome, n_inputs=2, n_outputs=1)
    assert module.weights.requires_grad
    assert module.has_edges
    assert int(module.mask.sum()) == len(solving_genome.enabled_connections())


def test_hand_built_genome_solves_xor(solving_genome, xor_adapter):
    module = xor_adapter.decode(solving_genome)
    metrics = xor_adapter.evaluate(module)
    assert metrics["query_accuracy"] == 1.0
