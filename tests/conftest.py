"""Shared fixtures: synthetic XOR / temporal / decomposable tasks and hand-built genomes (no network access)."""

import random

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


def _bit_patterns(n_bits: int, count: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    chosen = rng.sample(range(2**n_bits), count)
    return [[float((value >> bit) & 1) for bit in range(n_bits)] for value in chosen]


def _running_parity(bits: list[float]) -> list[float]:
    out: list[float] = []
    acc = 0
    for bit in bits:
        acc ^= int(bit)
        out.append(float(acc))
    return out


def _temporal_pairs(patterns: list[list[float]], seq_to_seq: bool) -> list[tuple[Field, Field]]:
    pairs: list[tuple[Field, Field]] = []
    for bits in patterns:
        x = Field(torch.tensor(bits, dtype=torch.float32), (Axis.TIME,), ValueType.BINARY, None, None, None)
        parity = _running_parity(bits)
        if seq_to_seq:
            y = Field(torch.tensor(parity, dtype=torch.float32), (Axis.TIME,), ValueType.BINARY, None, None, None)
        else:
            y = Field(torch.tensor([parity[-1]], dtype=torch.float32), (Axis.EXTRA,), ValueType.BINARY, None, None, None)
        pairs.append((x, y))
    return pairs


@pytest.fixture
def temporal_task() -> Task:
    """Seq-to-one running parity over T=8 binary steps: solvable only with state across time."""
    patterns = _bit_patterns(8, 16, seed=7)
    pairs = _temporal_pairs(patterns, seq_to_seq=False)
    meta = TaskMeta(rung=0, kind=TaskKind.MAP, name="running_parity", fixed_split=True)
    return Task(meta=meta, support=pairs[:12], query=pairs[12:])


@pytest.fixture
def temporal_seq_task() -> Task:
    """Seq-to-seq variant: the running parity AT EVERY step (target carries the TIME axis)."""
    patterns = _bit_patterns(8, 16, seed=7)
    pairs = _temporal_pairs(patterns, seq_to_seq=True)
    meta = TaskMeta(rung=0, kind=TaskKind.MAP, name="running_parity_seq", fixed_split=True)
    return Task(meta=meta, support=pairs[:12], query=pairs[12:])


@pytest.fixture
def decomposable_task() -> Task:
    """8-bit input -> 2-bit output where out[g] = parity(input half g).

    Splits cleanly under output_slices (per output bit) AND input_subsets (per input half), so it is
    the canonical fixture for decomposition and orchestrator tests.
    """
    patterns = _bit_patterns(8, 64, seed=11)
    pairs: list[tuple[Field, Field]] = []
    for bits in patterns:
        x = Field(torch.tensor(bits, dtype=torch.float32), (Axis.EXTRA,), ValueType.BINARY, None, None, None)
        y_values = [float(int(sum(bits[:4])) % 2), float(int(sum(bits[4:])) % 2)]
        y = Field(torch.tensor(y_values, dtype=torch.float32), (Axis.EXTRA,), ValueType.BINARY, None, None, None)
        pairs.append((x, y))
    meta = TaskMeta(rung=0, kind=TaskKind.MAP, name="half_parity", fixed_split=True)
    return Task(meta=meta, support=pairs[:48], query=pairs[48:])


@pytest.fixture
def xor_pairs_task() -> Task:
    """4-bit input -> 2-bit output where out[g] = XOR(bit 2g, bit 2g+1). Not linearly separable in
    either output, so a tiny evolve budget fails at depth 0; output_slices yields two
    XOR-with-spectators subtasks the direct strategy solves, and the port-wired skeleton over the
    frozen parts answers the parent exactly. The canonical REAL end-to-end decompose fixture."""
    pairs: list[tuple[Field, Field]] = []
    for value in range(16):
        bits = [float((value >> bit) & 1) for bit in range(4)]
        x = Field(torch.tensor(bits, dtype=torch.float32), (Axis.EXTRA,), ValueType.BINARY, None, None, None)
        y_values = [float(int(bits[0]) != int(bits[1])), float(int(bits[2]) != int(bits[3]))]
        y = Field(torch.tensor(y_values, dtype=torch.float32), (Axis.EXTRA,), ValueType.BINARY, None, None, None)
        pairs.append((x, y))
    meta = TaskMeta(rung=0, kind=TaskKind.MAP, name="xor_pairs", fixed_split=True)
    return Task(meta=meta, support=list(pairs), query=list(pairs))


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
