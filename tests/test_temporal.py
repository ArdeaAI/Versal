"""Temporal encoding + adapter: Level0 parity, end-to-end recurrent solving, BPTT through train."""

import random

import torch

from ardevo.dataset.icarus import Axis, Field, Level0Encoder, Task, TaskKind, TaskMeta, ValueType, support_loader
from ardevo.evaluation import support_loss
from ardevo.evolution.train import gradient
from ardevo.temporal import TemporalEncoder, temporal_adapter
from tests.test_recurrence import _running_parity_genome


def test_t1_encoding_matches_level0() -> None:
    """With T=1 the temporal encoding must be Level0 bit-for-bit (pins normalization + masking)."""
    fields = [
        Field(torch.tensor([[3.0]]), (Axis.CHANNEL, Axis.TIME), ValueType.CONTINUOUS, None, (0.0, 10.0), None),
        Field(torch.tensor([[7.5]]), (Axis.CHANNEL, Axis.TIME), ValueType.CONTINUOUS, None, (0.0, 10.0), torch.tensor([[False]])),
    ]
    task = Task(meta=TaskMeta(rung=0, kind=TaskKind.MAP, name="t1"), support=[(f, f) for f in fields], query=[])
    batched_input, _ = support_loader(task)

    temporal = TemporalEncoder(step_dim=1)
    level0 = Level0Encoder(max_flat_dim=1)
    stepped, descriptor = temporal.encode(batched_input)
    flat, flat_descriptor = level0.encode(batched_input)
    assert descriptor == flat_descriptor
    assert torch.equal(stepped.reshape(flat.shape), flat)


def test_temporal_adapter_seq_to_one_solved_by_parity_genome(temporal_task: Task) -> None:
    adapter = temporal_adapter(temporal_task)
    assert adapter.mode == "last" and adapter.n_inputs == 1 and adapter.n_outputs == 1
    module = adapter.decode(_running_parity_genome())
    metrics = adapter.evaluate(module)
    assert metrics["support_accuracy"] == 1.0
    assert metrics["query_accuracy"] == 1.0


def test_temporal_adapter_seq_to_seq_solved_by_parity_genome(temporal_seq_task: Task) -> None:
    adapter = temporal_adapter(temporal_seq_task)
    assert adapter.mode == "all" and adapter.n_outputs == 1
    module = adapter.decode(_running_parity_genome())
    x = adapter.encoded.support_input[0]
    assert module(x).shape == (x.shape[0], x.shape[1])  # one logit per step, t-major flat
    metrics = adapter.evaluate(module)
    assert metrics["support_accuracy"] == 1.0
    assert metrics["query_accuracy"] == 1.0


def test_bptt_through_gradient_train_op_reduces_loss(temporal_task: Task) -> None:
    adapter = temporal_adapter(temporal_task)
    genome = _running_parity_genome()
    # Detune every weight so the gradient op has something to recover through the unrolled steps.
    from dataclasses import replace

    genome.connections = [replace(conn, weight=conn.weight * 0.7) for conn in genome.connections]
    module = adapter.decode(genome)
    before = float(support_loss(module, adapter.encoded).detach())
    gradient(genome, module, adapter.encoded, rng=random.Random(0), steps=40, lr=0.05, writeback=False)
    after = float(support_loss(module, adapter.encoded).detach())
    assert after < before
