import random

from versal.evolution.train import gradient


def test_linear_genome_cannot_solve_xor(linear_genome, xor_adapter):
    """A no-hidden topology tops out at 75% on XOR even after heavy weight training.

    This is what forces structural growth: the only way past 0.75 is to add a hidden node.
    """
    module = xor_adapter.decode(linear_genome)
    gradient(linear_genome, module, xor_adapter.encoded, rng=random.Random(0), steps=300, lr=0.1, writeback=False)
    metrics = xor_adapter.evaluate(module)
    assert metrics["query_accuracy"] <= 0.75 + 1e-6


def test_evaluate_reports_support_and_query_metrics(solving_genome, xor_adapter):
    module = xor_adapter.decode(solving_genome)
    metrics = xor_adapter.evaluate(module)
    assert set(metrics) == {"support_accuracy", "support_loss", "query_accuracy", "query_loss"}
    assert metrics["query_accuracy"] == 1.0
    assert metrics["support_accuracy"] == 1.0
