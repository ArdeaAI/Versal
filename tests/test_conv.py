"""Weight-tied convolution (Pillar E): edges sharing a `share_group` decode to ONE shared parameter,
a convolution kernel reused across tiled placements. add_conv_motif grows such a kernel on grid tasks.
The flat/non-grid path stays byte-identical (no shared edges -> no scatter, no shared parameter)."""

import random

import torch

from ardevo.evolution.genome import ConnectionGene, Genome, InnovationTracker, NodeGene, NodeKind, genome_from_dict, genome_to_dict
from ardevo.evolution.init import minimal, stamp_input_coordinates
from ardevo.evolution.mutation import MutationContext, add_conv_motif
from ardevo.substrate import GraphNet, decode


def _nodes() -> dict[int, NodeGene]:
    return {
        0: NodeGene(0, NodeKind.INPUT, "identity"),
        1: NodeGene(1, NodeKind.INPUT, "identity"),
        2: NodeGene(2, NodeKind.HIDDEN, "tanh"),
        3: NodeGene(3, NodeKind.OUTPUT, "identity"),
    }


def _tied() -> Genome:
    """2 inputs -> hidden(tanh) -> output, with BOTH input->hidden edges sharing one weight (group 7)."""
    connections = [
        ConnectionGene(0, 2, 0.3, True, 0, share_group=7),
        ConnectionGene(1, 2, 0.5, True, 1, share_group=7),
        ConnectionGene(2, 3, 1.0, True, 2),
    ]
    return Genome(nodes=_nodes(), connections=connections)


def _untied() -> Genome:
    connections = [ConnectionGene(0, 2, 0.3, True, 0), ConnectionGene(1, 2, 0.5, True, 1), ConnectionGene(2, 3, 1.0, True, 2)]
    return Genome(nodes=_nodes(), connections=connections)


def _ctx() -> MutationContext:
    return MutationContext(innovations=InnovationTracker(_next_node_id=200), activations=["tanh", "identity"], default_activation="tanh")


# --- weight tying -------------------------------------------------------------------------------


def test_shared_group_collapses_to_one_parameter() -> None:
    net = decode(_tied(), 2, 1)
    assert hasattr(net, "shared_weights") and net.shared_weights.numel() == 1
    assert torch.allclose(net.shared_weights.detach(), torch.tensor([0.4]))  # init = mean(0.3, 0.5)
    exported = net.export_weights()
    assert exported[(0, 2, False)] == exported[(1, 2, False)]  # both taps report the SAME shared kernel value
    assert abs(exported[(0, 2, False)] - 0.4) < 1e-6


def test_tied_edges_use_the_same_weight_in_forward() -> None:
    net = decode(_tied(), 2, 1)
    out = net(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    # Both inputs route through the SAME 0.4 weight, so x=[1,0] and x=[0,1] give identical outputs.
    assert torch.allclose(out[0], out[1], atol=1e-6)


def test_tied_kernel_receives_one_accumulated_gradient() -> None:
    net = decode(_tied(), 2, 1)
    net(torch.tensor([[1.0, 1.0]])).sum().backward()
    assert net.shared_weights.grad is not None and net.shared_weights.grad.numel() == 1
    assert torch.isfinite(net.shared_weights.grad).all()


def test_tied_net_trains_and_stays_tied() -> None:
    net = decode(_tied(), 2, 1)
    x = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    target = torch.tensor([[2.0], [-2.0]])
    optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
    before = float(((net(x) - target) ** 2).mean().detach())
    for _ in range(50):
        optimizer.zero_grad()
        loss = ((net(x) - target) ** 2).mean()
        loss.backward()
        optimizer.step()
    assert float(((net(x) - target) ** 2).mean().detach()) < before  # the shared kernel is learnable
    exported = net.export_weights()
    assert exported[(0, 2, False)] == exported[(1, 2, False)]  # still tied after training


# --- byte-identical flat path -------------------------------------------------------------------


def test_untied_genome_has_no_shared_parameter_and_is_unchanged() -> None:
    net = decode(_untied(), 2, 1)
    assert not hasattr(net, "shared_weights")
    assert net._share_flat_index.numel() == 0
    # Independent weights: x=[1,0] uses 0.3, x=[0,1] uses 0.5 -> different outputs (contrast the tie).
    out = net(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert not torch.allclose(out[0], out[1], atol=1e-6)


def test_untied_genome_is_not_batchable_only_when_tied() -> None:
    assert decode(_untied(), 2, 1).core()[0] is not None  # plain GraphNet stays batchable
    assert decode(_tied(), 2, 1).core()[0] is None  # weight-tied falls back to the sequential trainer


# --- add_conv_motif -----------------------------------------------------------------------------


def _grid_genome() -> Genome:
    return stamp_input_coordinates(minimal(9, 1, rng=random.Random(0), default_activation="tanh"), (3, 3))


def test_add_conv_motif_ties_taps_across_tiles() -> None:
    child = add_conv_motif(_grid_genome(), _ctx(), rng=random.Random(0), prob=1.0, kernel=4, copies=2)
    tied = [conn for conn in child.connections if conn.share_group is not None]
    assert tied, "expected weight-tied input->feature edges"
    feature_nodes = {conn.out_id for conn in tied}
    assert len(feature_nodes) >= 2  # tiled across at least two anchors
    # The same kernel tap (share_group) appears on more than one tile: the kernel is genuinely reused.
    from collections import Counter

    group_counts = Counter(conn.share_group for conn in tied)
    assert max(group_counts.values()) >= 2  # at least one tap shared across tiles
    # Readout edges (feature->output) stay independent (the classifier head is not tied).
    readout = [conn for conn in child.connections if conn.in_id in feature_nodes and child.nodes[conn.out_id].kind is NodeKind.OUTPUT]
    assert readout and all(conn.share_group is None for conn in readout)


def test_add_conv_motif_decodes_with_shared_kernel() -> None:
    child = add_conv_motif(_grid_genome(), _ctx(), rng=random.Random(1), prob=1.0, kernel=4, copies=2)
    net = decode(child, 9, 1)
    distinct_groups = {conn.share_group for conn in child.connections if conn.share_group is not None}
    assert hasattr(net, "shared_weights") and net.shared_weights.numel() == len(distinct_groups)
    assert isinstance(net, GraphNet)
    net(torch.randn(4, 9)).sum().backward()  # the tiled kernel is differentiable end to end
    assert net.shared_weights.grad is not None


def test_add_conv_motif_is_a_noop_without_coordinates() -> None:
    flat = minimal(9, 1, rng=random.Random(0), default_activation="tanh")  # no coordinates stamped
    child = add_conv_motif(flat, _ctx(), rng=random.Random(0), prob=1.0, kernel=4, copies=2)
    assert child.connections == flat.connections and child.nodes.keys() == flat.nodes.keys()


def test_add_conv_motif_respects_probability() -> None:
    grid = _grid_genome()
    assert add_conv_motif(grid, _ctx(), rng=random.Random(0), prob=0.0).connections == grid.connections


# --- serialization ------------------------------------------------------------------------------


def test_share_group_round_trips_and_legacy_default() -> None:
    restored = genome_from_dict(genome_to_dict(_tied()))
    groups = {conn.share_group for conn in restored.connections if conn.in_id in (0, 1)}
    assert groups == {7}
    legacy = genome_to_dict(_untied())
    for conn in legacy["connections"]:
        del conn["share_group"]  # a pre-phase-6 genome dict
    assert all(conn.share_group is None for conn in genome_from_dict(legacy).connections)
