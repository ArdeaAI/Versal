"""Phase 7 Pillar D: the equilibrium substrate (test-time-tunable depth) + the depth-scaling currency.

EquilibriumGraphNet iterates the recurrent graph to a damped fixed point instead of a gene-fixed
number of refine passes, so ANY genome with one recurrent edge gets variable depth. gradient_equilibrium
deep-supervises the iteration; depth_scaled stamps a depth_scaling_score so recursion can be selected
for. With no recurrent edge / equilibrium off, the path is byte-identical to feedforward."""

import random

import torch

from ardevo.evolution.evaluate import depth_scaled
from ardevo.evolution.evolver import TaskAdapter
from ardevo.evolution.fitness import depth_scaling_score
from ardevo.evolution.genome import ConnectionGene, Genome, NodeGene, NodeKind
from ardevo.evolution.train import gradient_equilibrium
from ardevo.substrate import EquilibriumGraphNet, GraphNet, decode, decode_equilibrium


def _feedforward_genome() -> Genome:
    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity"),
        1: NodeGene(1, NodeKind.INPUT, "identity"),
        2: NodeGene(2, NodeKind.BIAS, "identity"),
        3: NodeGene(3, NodeKind.HIDDEN, "tanh"),
        4: NodeGene(4, NodeKind.OUTPUT, "identity"),
    }
    connections = [
        ConnectionGene(0, 3, 0.7, True, 0),
        ConnectionGene(1, 3, -0.4, True, 1),
        ConnectionGene(2, 3, 0.1, True, 2),
        ConnectionGene(3, 4, 1.2, True, 3),
    ]
    return Genome(nodes=nodes, connections=connections)


def _recurrent_genome() -> Genome:
    genome = _feedforward_genome()
    genome.connections.append(ConnectionGene(3, 3, 0.6, True, 4, recurrent=True))  # hidden self-loop (latent state)
    return genome


def test_equilibrium_with_no_recurrence_matches_feedforward() -> None:
    # damping 1.0 + no recurrent edges -> the fixed point IS the feedforward output (reached in 2 iters).
    genome = _feedforward_genome()
    plain = decode(genome, 2, 1)
    equilibrium = decode_equilibrium(genome, 2, 1, damping=1.0, max_iters=16)
    x = torch.tensor([[1.0, 0.5], [-0.3, 0.8]])
    assert torch.allclose(plain(x), equilibrium(x), atol=1e-5)


def test_equilibrium_recurrent_edge_is_live_and_differs_from_feedforward() -> None:
    genome = _recurrent_genome()
    plain = decode(genome, 2, 1)  # GraphNet ignores recurrent edges
    equilibrium = decode_equilibrium(genome, 2, 1, damping=0.5, max_iters=20)
    x = torch.tensor([[1.0, 0.5]])
    assert not torch.allclose(plain(x), equilibrium(x), atol=1e-3)  # the self-loop actually does work


def test_equilibrium_trace_shape_and_convergence() -> None:
    equilibrium = decode_equilibrium(_recurrent_genome(), 2, 1, damping=0.5, max_iters=20, tol=1e-5)
    trace = equilibrium.equilibrium_trace(torch.tensor([[1.0, 0.5], [0.2, -0.1]]))
    assert trace.dim() == 3 and trace.shape[0] == 2 and trace.shape[2] == 1  # [batch, iters, n_outputs]
    assert 1 <= trace.shape[1] <= 20
    # The last two readouts are within tol (the iteration converged before the cap).
    if trace.shape[1] >= 2:
        assert torch.allclose(trace[:, -1], trace[:, -2], atol=1e-2)


def test_gradient_equilibrium_trains_and_falls_back(xor_adapter) -> None:
    # On an equilibrium module it deep-supervises the iteration; on a plain module it falls back cleanly.
    genome = _recurrent_genome()
    module = decode_equilibrium(genome, 2, 1, damping=0.5, max_iters=8)
    before = module.equilibrium_trace(xor_adapter.encoded.support_input[0])[:, -1].detach().clone()
    gradient_equilibrium(genome, module, xor_adapter.encoded, rng=random.Random(0), steps=10, lr=0.1)  # trains in place
    after = module.equilibrium_trace(xor_adapter.encoded.support_input[0])[:, -1]
    assert not torch.allclose(before, after, atol=1e-4)  # weights moved (the deep-supervised loss trained)
    # Fallback: a plain GraphNet has no equilibrium_trace, so it routes to the refine/gradient path.
    plain = decode(_feedforward_genome(), 2, 1)
    _g, plain_trained = gradient_equilibrium(_feedforward_genome(), plain, xor_adapter.encoded, rng=random.Random(0), steps=2, lr=0.1)
    assert isinstance(plain_trained, GraphNet)


def test_depth_scaled_scores_depth_for_equilibrium_zero_for_feedforward(xor_adapter) -> None:
    equilibrium_adapter = TaskAdapter(xor_adapter.encoded, xor_adapter.encoder, 2, 1, equilibrium={"damping": 0.5, "max_iters": 8})
    recurrent_module = equilibrium_adapter.decode(_recurrent_genome())
    assert isinstance(recurrent_module, EquilibriumGraphNet)
    scored = depth_scaled(_recurrent_genome(), recurrent_module, equilibrium_adapter, low_depth=1, high_depth=8)
    assert "depth_scaling_score" in scored  # a tunable-depth module reports a score
    feedforward_module = equilibrium_adapter.decode(_feedforward_genome())  # no recurrent edge -> plain decode
    assert not isinstance(feedforward_module, EquilibriumGraphNet)
    flat = depth_scaled(_feedforward_genome(), feedforward_module, equilibrium_adapter)
    assert flat["depth_scaling_score"] == 0.0  # nothing to scale


def test_depth_scaling_score_fitness_reads_metric() -> None:
    assert depth_scaling_score(None, {"depth_scaling_score": 0.3}) == 0.3
    assert depth_scaling_score(None, {}) == 0.0  # absent -> inert


def test_adapter_equilibrium_off_is_byte_identical_decode(xor_adapter) -> None:
    # equilibrium None: a recurrent genome decodes through decode_module exactly as before (not equilibrium).
    module = xor_adapter.decode(_recurrent_genome())
    assert not isinstance(module, EquilibriumGraphNet)
