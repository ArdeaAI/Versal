import random

import pytest

from ardevo.evolution.init import INIT, estimate_initialization
from ardevo.substrate import decode


@pytest.mark.parametrize(
    ("kind", "params"),
    [("minimal", {}), ("factored", {"rank": 3, "threshold": 0}), ("sparse", {"density": 0.2, "threshold": 0}), ("cppn", {"hidden": 4, "density": 0.5})],
)
def test_builtin_estimate_matches_tiny_seed(kind: str, params: dict) -> None:
    genome = INIT.get(kind)(7, 5, rng=random.Random(3), **params)
    estimate = estimate_initialization(kind, 7, 5, **params)
    assert estimate.nodes == len(genome.nodes)
    assert estimate.edges == len(genome.connections)
    net = decode(genome, 7, 5)
    assert net.weights.numel() == estimate.nodes * (estimate.nodes - 8)
    assert net.mask.numel() == net.weights.numel()
