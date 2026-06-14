import math
import random

import torch

from ardevo.evaluation import split_metrics_from_raw
from ardevo.evolution.train import gradient


def test_split_metrics_sanitizes_non_finite_output(xor_adapter, xor_encoder) -> None:
    """Phase-6 regression: an exploded forward (NaN logits) must score as the WORST (zero accuracy,
    huge finite loss), not propagate a NaN into fitness/speciation."""
    target, mask, descriptor = xor_adapter.encoded.support_target
    raw = torch.full(target.shape, float("nan"))
    accuracy, loss = split_metrics_from_raw(raw, target, mask, descriptor, xor_encoder)
    assert math.isfinite(accuracy) and math.isfinite(loss)  # neither poisons fitness/speciation
    assert loss >= 1.0e9  # the non-finite loss is floored to the worst


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
