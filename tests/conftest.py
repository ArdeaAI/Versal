"""Shared fixtures: a synthetic XOR task and hand-built genomes (no network access)."""

import pytest
import torch

from ardevo.dataset.icarus import Axis, Field, Level0Encoder, Task, TaskKind, TaskMeta, ValueType
from ardevo.evaluation import encode, input_width, output_features
from ardevo.evolution.evolver import TaskAdapter
from ardevo.evolution.genome import ConnectionGene, Genome, NodeGene, NodeKind

_XOR_ROWS = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]


def _binary_field(values: list[float]) -> Field:
    return Field(torch.tensor(values, dtype=torch.float32), (Axis.EXTRA,), ValueType.BINARY, None, None, None)


@pytest.fixture
def xor_task() -> Task:
    pairs = [(_binary_field([float(x0), float(x1)]), _binary_field([float(y)])) for x0, x1, y in _XOR_ROWS]
    meta = TaskMeta(rung=1, kind=TaskKind.MAP, name="xor", fixed_split=True)
    return Task(meta=meta, support=list(pairs), query=list(pairs))


@pytest.fixture
def xor_encoder() -> Level0Encoder:
    return Level0Encoder(max_flat_dim=2)


@pytest.fixture
def xor_adapter(xor_task: Task, xor_encoder: Level0Encoder) -> TaskAdapter:
    encoded = encode(xor_task, xor_encoder)
    return TaskAdapter(encoded, xor_encoder, input_width(encoded), output_features(encoded))


@pytest.fixture
def linear_genome() -> Genome:
    """2 inputs + bias wired straight to a linear output (no hidden node)."""
    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity"),
        1: NodeGene(1, NodeKind.INPUT, "identity"),
        2: NodeGene(2, NodeKind.BIAS, "identity"),
        3: NodeGene(3, NodeKind.OUTPUT, "identity"),
    }
    connections = [
        ConnectionGene(0, 3, 0.5, True, 0),
        ConnectionGene(1, 3, 0.5, True, 1),
        ConnectionGene(2, 3, 0.0, True, 2),
    ]
    return Genome(nodes=nodes, connections=connections)


@pytest.fixture
def solving_genome() -> Genome:
    """A hand-built XOR solution: OR and AND hidden units, output = OR - AND - 1.

    With saturating weights, h_or ~ +1 when at least one input is on, h_and ~ +1 only when both
    are on. So OR - AND is ~+2 for exactly-one-on (XOR=1) and ~0 otherwise; the -1 output bias
    centers the threshold so the zero-cases fall clearly negative (predicted 0).
    """
    k = 10.0
    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity"),
        1: NodeGene(1, NodeKind.INPUT, "identity"),
        2: NodeGene(2, NodeKind.BIAS, "identity"),
        3: NodeGene(3, NodeKind.HIDDEN, "tanh"),  # OR
        4: NodeGene(4, NodeKind.HIDDEN, "tanh"),  # AND
        5: NodeGene(5, NodeKind.OUTPUT, "identity"),
    }
    connections = [
        ConnectionGene(0, 3, k, True, 0),
        ConnectionGene(1, 3, k, True, 1),
        ConnectionGene(2, 3, -0.5 * k, True, 2),
        ConnectionGene(0, 4, k, True, 3),
        ConnectionGene(1, 4, k, True, 4),
        ConnectionGene(2, 4, -1.5 * k, True, 5),
        ConnectionGene(3, 5, 1.0, True, 6),
        ConnectionGene(4, 5, -1.0, True, 7),
        ConnectionGene(2, 5, -1.0, True, 8),
    ]
    return Genome(nodes=nodes, connections=connections)
