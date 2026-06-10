import random

from ardevo.evaluation import support_loss
from ardevo.evolution.train import gradient, no_train


def test_gradient_reduces_support_loss(linear_genome, xor_adapter):
    module = xor_adapter.decode(linear_genome)
    before = float(support_loss(module, xor_adapter.encoded).detach())
    _genome, module = gradient(linear_genome, module, xor_adapter.encoded, rng=random.Random(0), steps=100, lr=0.05, writeback=False)
    after = float(support_loss(module, xor_adapter.encoded).detach())
    assert after < before


def test_gradient_writeback_updates_genome(linear_genome, xor_adapter):
    module = xor_adapter.decode(linear_genome)
    genome, _module = gradient(linear_genome, module, xor_adapter.encoded, rng=random.Random(0), steps=20, lr=0.05, writeback=True)
    before = [c.weight for c in linear_genome.connections]
    after = [c.weight for c in genome.connections]
    assert before != after


def test_none_leaves_module_unchanged(linear_genome, xor_adapter):
    module = xor_adapter.decode(linear_genome)
    genome, returned = no_train(linear_genome, module, xor_adapter.encoded, rng=random.Random(0))
    assert genome is linear_genome
    assert returned is module
