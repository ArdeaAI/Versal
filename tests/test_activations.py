"""The sin/gaussian palette expansion: semantics, training stability, and evolvability.

sin and gaussian are the periodic + radial primitives (WANN/CPPN precedent) added for the
two-spirals class of boundary. They must stay mutation-only (never seeded) and freely removable,
so the tests pin exactly that plus the numerics.
"""

import math
import random

import pytest
import torch

from ardevo.evaluation import support_loss
from ardevo.evolution.genome import ConnectionGene, Genome, InnovationTracker, NodeGene, NodeKind
from ardevo.evolution.mutation import MutationContext, mutate_activation
from ardevo.evolution.train import gradient
from ardevo.substrate import _ACTIVATIONS, activation_names
from ardevo.utils.config import Config

_NEW_NAMES = {"sin", "gaussian"}


def _hidden_activation_genome(activation: str) -> Genome:
    """2 inputs + bias -> 1 hidden(activation) -> output, nonzero weights so gaussian sits off its
    flat point at pre-activation 0."""
    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity"),
        1: NodeGene(1, NodeKind.INPUT, "identity"),
        2: NodeGene(2, NodeKind.BIAS, "identity"),
        3: NodeGene(3, NodeKind.HIDDEN, activation),
        4: NodeGene(4, NodeKind.OUTPUT, "identity"),
    }
    connections = [
        ConnectionGene(0, 3, 0.7, True, 0),
        ConnectionGene(1, 3, -0.4, True, 1),
        ConnectionGene(2, 3, 0.3, True, 2),
        ConnectionGene(3, 4, 1.2, True, 3),
        ConnectionGene(2, 4, -0.1, True, 4),
    ]
    return Genome(nodes=nodes, connections=connections)


def test_new_activation_semantics() -> None:
    # Pins gaussian = exp(-x^2), NOT the exp(-x^2 / 2) variant.
    values = _ACTIVATIONS["gaussian"](torch.tensor([0.0, 2.0]))
    assert torch.allclose(values, torch.tensor([1.0, math.exp(-4.0)]))
    assert _ACTIVATIONS["sin"] is torch.sin


@pytest.mark.parametrize("activation", ["tanh", "relu", "sigmoid", "identity", "sin", "gaussian"])
def test_decoded_net_with_any_activation_pickles(activation: str, xor_adapter) -> None:
    """The composition assess pool pickles whole trained nets back from workers, so every palette
    callable stored on a decoded net must pickle by reference (the diag_g0 gaussian-lambda crash).
    Safe pickle use: round-tripping an object built in this process, exactly what pool.map does."""
    import pickle

    module = xor_adapter.decode(_hidden_activation_genome(activation))
    clone = pickle.loads(pickle.dumps(module))
    x = torch.tensor([[0.7, -1.3], [0.0, 2.0]])
    assert torch.allclose(clone(x), module(x), atol=1e-6)


@pytest.mark.parametrize("activation", sorted(_NEW_NAMES))
def test_new_activation_genome_trains_without_nans(activation: str, xor_adapter) -> None:
    genome = _hidden_activation_genome(activation)
    module = xor_adapter.decode(genome)
    _genome, trained = gradient(genome, module, xor_adapter.encoded, rng=random.Random(0), steps=60, lr=0.05, writeback=False)
    assert all(torch.isfinite(parameter).all() for parameter in trained.parameters())
    assert math.isfinite(float(support_loss(trained, xor_adapter.encoded).detach()))


def test_mutate_activation_reaches_and_leaves_new_names() -> None:
    palette = ["tanh", "relu", "sigmoid", "identity", "sin", "gaussian"]
    genome = _hidden_activation_genome("tanh")
    ctx = MutationContext(innovations=InnovationTracker.from_genomes([genome]), activations=palette, default_activation="tanh")
    rng = random.Random(0)
    seen = [genome.nodes[3].activation]
    for _ in range(100):
        genome = mutate_activation(genome, ctx, rng=rng, prob=1.0)
        seen.append(genome.nodes[3].activation)
    assert _NEW_NAMES <= set(seen)  # mutation can reach both new names
    first_new = next(index for index, name in enumerate(seen) if name in _NEW_NAMES)
    assert any(name not in _NEW_NAMES for name in seen[first_new:])  # and can leave them again


def test_shipped_palette_is_dispatchable() -> None:
    # A palette typo in config otherwise only fails at decode time mid-run.
    substrate_cfg = Config().current["substrate"]
    assert set(substrate_cfg["available_activations"]) <= set(activation_names())
    assert substrate_cfg["default_activation"] in substrate_cfg["available_activations"]
    assert _NEW_NAMES <= set(substrate_cfg["available_activations"])
