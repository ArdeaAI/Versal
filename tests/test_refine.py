"""Recursive-refinement substrate (TRM): re-apply a tiny network to a static input, threading node
state across passes via recurrent edges. steps=1 is exactly a feedforward pass; steps>1 adds
effective depth without parameters. Evolvable via the refine_steps gene + tweak_refine_steps."""

import random

import torch

from versal.evaluation import support_loss, support_loss_deep
from versal.evolution.genome import ConnectionGene, Genome, InnovationTracker, NodeGene, NodeKind, genome_from_dict, genome_to_dict
from versal.evolution.mutation import MutationContext, tweak_refine_steps
from versal.evolution.train import gradient_refine
from versal.substrate import GraphNet, RefineGraphNet, decode, decode_refine


def _accumulator() -> Genome:
    """1 input -> hidden(identity) with a recurrent self-loop -> output. Over K static passes the
    hidden node accumulates: output(K) = K * input, so refinement provably carries state."""
    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity"),
        1: NodeGene(1, NodeKind.OUTPUT, "identity"),
        2: NodeGene(2, NodeKind.HIDDEN, "identity"),
    }
    connections = [
        ConnectionGene(0, 2, 1.0, True, 0),
        ConnectionGene(2, 2, 1.0, True, 1, recurrent=True),  # the memory edge
        ConnectionGene(2, 1, 1.0, True, 2),
    ]
    return Genome(nodes=nodes, connections=connections)


def test_refine_steps_round_trip_and_legacy_default() -> None:
    genome = _accumulator()
    genome.refine_steps = 4
    restored = genome_from_dict(genome_to_dict(genome))
    assert restored.refine_steps == 4
    legacy = genome_to_dict(genome)
    del legacy["refine_steps"]  # a pre-phase-5 genome dict
    assert genome_from_dict(legacy).refine_steps == 1


def test_clone_preserves_refine_steps() -> None:
    genome = _accumulator()
    genome.refine_steps = 5
    assert genome.clone().refine_steps == 5


def test_refine_steps_one_equals_feedforward(solving_genome: Genome) -> None:
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    feedforward = decode(solving_genome, 2, 1)
    refine_one = decode_refine(solving_genome, 2, 1, steps=1)
    assert isinstance(refine_one, RefineGraphNet)
    assert torch.allclose(feedforward(x), refine_one(x), atol=1e-6)


def test_refine_carries_state_across_passes() -> None:
    accumulator = _accumulator()
    x = torch.tensor([[2.0]])
    one = decode_refine(accumulator, 1, 1, steps=1)(x)
    three = decode_refine(accumulator, 1, 1, steps=3)(x)
    assert torch.allclose(one, torch.tensor([[2.0]]), atol=1e-6)  # single pass: just the input
    assert torch.allclose(three, torch.tensor([[6.0]]), atol=1e-6)  # 3 passes accumulate: 3 * input


def test_refine_trace_returns_per_step_readouts() -> None:
    accumulator = _accumulator()
    module = decode_refine(accumulator, 1, 1, steps=3)
    trace = module.refine_trace(torch.tensor([[2.0], [1.0]]))
    assert trace.shape == (2, 3, 1)
    assert torch.allclose(trace[:, :, 0], torch.tensor([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]]), atol=1e-6)


def test_refine_genome_decodes_through_adapter(xor_adapter, solving_genome: Genome) -> None:
    refine_genome = solving_genome.clone()
    refine_genome.refine_steps = 3
    module = xor_adapter.decode(refine_genome)
    assert isinstance(module, RefineGraphNet) and module.steps == 3
    plain = xor_adapter.decode(solving_genome.clone())  # refine_steps defaults to 1 -> plain GraphNet
    assert isinstance(plain, GraphNet) and not isinstance(plain, RefineGraphNet)


def test_refine_is_differentiable_through_recursion() -> None:
    accumulator = _accumulator()
    module = decode_refine(accumulator, 1, 1, steps=3)
    out = module(torch.tensor([[2.0]]))
    out.sum().backward()  # BPTT through the 3-pass unroll
    assert module.recurrent_weights.grad is not None and torch.isfinite(module.recurrent_weights.grad).all()


def test_tweak_refine_steps_stays_in_bounds() -> None:
    ctx = MutationContext(innovations=InnovationTracker(_next_node_id=0), activations=["identity"], default_activation="identity")
    genome = _accumulator()
    genome.refine_steps = 1
    # Always-fire, many draws: never leaves [1, 4], moves by at most 1 per step.
    for _ in range(50):
        before = genome.refine_steps
        genome = tweak_refine_steps(genome, ctx, rng=random.Random(_), prob=1.0, min_steps=1, max_steps=4)
        assert 1 <= genome.refine_steps <= 4 and abs(genome.refine_steps - before) <= 1


def test_tweak_refine_steps_respects_probability() -> None:
    ctx = MutationContext(innovations=InnovationTracker(_next_node_id=0), activations=["identity"], default_activation="identity")
    genome = _accumulator()
    assert tweak_refine_steps(genome, ctx, rng=random.Random(0), prob=0.0).refine_steps == 1  # never fires


def test_gradient_refine_trains_a_refine_genome(xor_adapter, linear_genome: Genome) -> None:
    refine_genome = linear_genome.clone()
    refine_genome.refine_steps = 3
    module = xor_adapter.decode(refine_genome)
    assert isinstance(module, RefineGraphNet)
    before = float(support_loss_deep(module, xor_adapter.encoded).detach())
    _genome, module = gradient_refine(refine_genome, module, xor_adapter.encoded, rng=random.Random(0), steps=80, lr=0.05, writeback=False)
    after = float(support_loss_deep(module, xor_adapter.encoded).detach())
    assert after < before  # deep-supervised BPTT through the recursion reduces the loss


def test_gradient_refine_falls_back_for_non_refine(xor_adapter, linear_genome: Genome) -> None:
    module = xor_adapter.decode(linear_genome)  # refine_steps defaults to 1 -> plain GraphNet
    assert not hasattr(module, "refine_trace")
    before = float(support_loss(module, xor_adapter.encoded).detach())
    _genome, module = gradient_refine(linear_genome, module, xor_adapter.encoded, rng=random.Random(0), steps=50, lr=0.05, writeback=False)
    after = float(support_loss(module, xor_adapter.encoded).detach())
    assert after < before  # behaves exactly like the standard gradient op
