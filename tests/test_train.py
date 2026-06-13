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


def test_gradient_stops_on_nonfinite_loss(monkeypatch, linear_genome, xor_adapter):
    """Bug-6 regression: a NaN/Inf loss must NOT be backpropagated and stepped (Adam does not filter
    non-finite grads, so a single step would poison every weight). The guard breaks the loop first."""
    import torch

    import ardevo.evolution.train as train_module

    module = xor_adapter.decode(linear_genome)

    def nan_loss(decoded, encoded):
        # Depends on a real parameter (so a step WOULD corrupt it) but is NaN.
        return next(iter(decoded.parameters())).sum() * torch.tensor(float("nan"))

    monkeypatch.setattr(train_module, "support_loss", nan_loss)
    _genome, trained = train_module.gradient(linear_genome, module, xor_adapter.encoded, rng=random.Random(0), steps=10, lr=0.05, writeback=False)
    assert all(torch.isfinite(parameter).all() for parameter in trained.parameters())  # weights never poisoned
