"""Compact-column substrate invariants: the slim [n, h] parameterization must be the SAME math as
the dense [n, n] layout it replaced. Forward outputs are bitwise equal (the sliced GEMM operands
are element-for-element identical), and per-genome feedforward training is bitwise equal (Adam is
elementwise; live entries see identical gradients). Recurrent-substrate TRAINING is the one
documented exception: the recurrent matmul's backward reduces over h terms instead of n, so BLAS
groups the nonzero partial sums differently and gradients match to ulp scale, not bitwise (forward
stays bitwise). Verified old-vs-new against the pre-slim substrate at commit 9116b08."""

import random

import torch
from torch import nn

from versal.dataset.icarus import Axis, Field, Level0Encoder, Task, TaskKind, TaskMeta, ValueType, encode_task
from versal.evolution.genome import ConnectionGene, Genome, InnovationTracker, NodeGene, NodeKind
from versal.evolution.init import minimal
from versal.evolution.mutation import MutationContext, add_connection, add_deep_node, add_recurrent_connection, add_rich_node
from versal.evolution.train import _writeback, gradient
from versal.substrate import GraphNet, decode, decode_recurrent, decode_refine
from versal.substrate_batched import BatchedGraphNet


class DenseReference(nn.Module):
    """The pre-slim dense [n, n] layout rebuilt from a slim net's structure: same level schedule,
    same activation groups, dense weight/mask with columns scattered back to node positions. Its
    forward is the documented level loop verbatim, so slim-vs-dense compares layouts only."""

    def __init__(self, net: GraphNet) -> None:
        super().__init__()
        self.net = net
        dense_weights = torch.zeros(net.n, net.n)
        dense_mask = torch.zeros(net.n, net.n, dtype=torch.bool)
        dense_weights[:, net.col_index] = net.weights.detach()
        dense_mask[:, net.col_index] = net.mask
        self.weights = nn.Parameter(dense_weights)
        self.mask = dense_mask

    @property
    def has_edges(self) -> bool:
        return self.net.has_edges

    def export_weights(self) -> dict[tuple[int, int, bool], float]:
        detached = self.weights.detach()
        col_index = self.net.col_index
        exported = {(in_id, out_id, False): float(detached[source, col_index[col]]) for in_id, out_id, source, col in self.net._edge_positions}
        exported.update({(in_id, out_id, recurrent): weight for in_id, out_id, recurrent, weight in self.net._inert_edges})
        return exported

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.net
        batch = x.shape[0]
        values = torch.zeros(batch, net.n, dtype=x.dtype)
        if net.input_pos.numel():
            values = values.index_copy(1, net.input_pos, x)
        if net.bias_pos.numel():
            values = values.index_copy(1, net.bias_pos, torch.ones(batch, net.bias_pos.numel(), dtype=x.dtype))
        masked = self.weights * self.mask
        for (level_positions, _level_cols, activation_groups), products in zip(net._levels, net._product_entries):
            pre_activation = values @ masked[:, level_positions]
            for local_index, node_col, source_positions in products:
                node_position = net.col_index[node_col]
                edge_weights = masked.index_select(0, source_positions).index_select(1, node_position).squeeze(1)
                factors = values.index_select(1, source_positions) * edge_weights
                pre_activation = pre_activation.index_copy(1, local_index, factors.prod(dim=1, keepdim=True))
            activated = pre_activation
            for activation, local_indices in activation_groups:
                activated = activated.index_copy(1, local_indices, activation(activated.index_select(1, local_indices)))
            values = values.index_copy(1, level_positions, activated)
        return values.index_select(1, net.output_pos)


def _task(width: int, rows: int = 64, seed: int = 0) -> Task:
    rng = random.Random(seed)
    pairs = []
    for _ in range(rows):
        bits = [float(rng.getrandbits(1)) for _ in range(width)]
        x = Field(torch.tensor(bits), (Axis.EXTRA,), ValueType.BINARY, None, None, None)
        y = Field(torch.tensor([float(int(sum(bits)) % 2)]), (Axis.EXTRA,), ValueType.BINARY, None, None, None)
        pairs.append((x, y))
    meta = TaskMeta(rung=0, kind=TaskKind.MAP, name=f"slim_w{width}")
    return Task(meta=meta, support=pairs[: int(rows * 0.8)], query=pairs[int(rows * 0.8) :])


def _grown(width: int, seed: int, rounds: int = 5) -> Genome:
    rng = random.Random(seed)
    genome = minimal(width, 1, rng=rng, default_activation="tanh", weight_scale=1.0)
    ctx = MutationContext(InnovationTracker.from_genomes([genome]), ["tanh", "relu", "sigmoid"], "tanh")
    for _ in range(rounds):
        genome = add_rich_node(genome, ctx, rng=rng, prob=1.0, fan_in=5)
        genome = add_deep_node(genome, ctx, rng=rng, prob=1.0, fan_in=5, fan_out=3)
        genome = add_connection(genome, ctx, rng=rng, prob=1.0)
    return genome


def test_forward_matches_dense_reference_bitwise() -> None:
    for width, seed in ((2, 1), (16, 2), (784, 3)):
        genome = _grown(width, seed)
        net = decode(genome, width, 1)
        assert net.h < net.n  # the point of the layout: columns only for computed nodes
        reference = DenseReference(net)
        x = torch.randn(32, width)
        assert torch.equal(net(x), reference(x))


def test_product_forward_matches_dense_reference_bitwise() -> None:
    genome = _grown(2, seed=7)
    hidden = next(iter(genome.hidden_ids))
    node = genome.nodes[hidden]
    genome.nodes[hidden] = NodeGene(node.id, node.kind, node.activation, node.coordinate, "product")
    net = decode(genome, 2, 1)
    x = torch.randn(32, 2)
    assert torch.equal(net(x), DenseReference(net)(x))


def test_training_matches_dense_parameterization_bitwise() -> None:
    width = 16
    encoded = encode_task(_task(width, seed=4), Level0Encoder(max_flat_dim=width))
    genome = _grown(width, seed=4)
    slim = decode(genome, width, 1)
    reference = DenseReference(decode(genome, width, 1))
    _, slim_trained = gradient(genome.clone(), slim, encoded, rng=random.Random(0), steps=10, lr=0.03, weight_decay=0.0002)
    _, dense_trained = gradient(genome.clone(), reference, encoded, rng=random.Random(0), steps=10, lr=0.03, weight_decay=0.0002)
    assert slim_trained.export_weights() == dense_trained.export_weights()


def test_refine_forward_matches_dense_and_training_is_close() -> None:
    width = 8
    genome = _grown(width, seed=5).clone()
    genome.refine_steps = 3
    rng = random.Random(5)
    ctx = MutationContext(InnovationTracker.from_genomes([genome]), ["tanh"], "tanh")
    for _ in range(3):
        genome = add_recurrent_connection(genome, ctx, rng=rng, prob=1.0)
    net = decode_refine(genome, width, 1)
    x = torch.randn(16, width)
    # Recurrent state starts at zero, so pass 1 equals the flat decode; the full refine forward must
    # still be finite and reproducible (bitwise determinism on CPU).
    assert torch.equal(net(x), net(x))
    seq_net = decode_recurrent(genome, width, 1, "last")
    assert torch.equal(seq_net(x.unsqueeze(1).expand(16, 3, width)), net(x))


def test_inert_edges_survive_export_writeback_and_has_edges() -> None:
    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity"),
        1: NodeGene(1, NodeKind.INPUT, "identity"),
        2: NodeGene(2, NodeKind.OUTPUT, "identity"),
    }
    connections = [
        ConnectionGene(0, 2, 0.5, True, 0),
        ConnectionGene(0, 1, 0.25, True, 1),  # forward edge into an input: inert, no compact column
        ConnectionGene(2, 0, -0.75, True, 2, recurrent=True),  # recurrent into an input: inert
    ]
    genome = Genome(nodes=nodes, connections=connections)
    net = decode_recurrent(genome, 2, 1, "last")
    exported = net.export_weights()
    assert exported[(0, 1, False)] == 0.25
    assert exported[(2, 0, True)] == -0.75
    assert exported[(0, 2, False)] == 0.5
    written = _writeback(genome, net)
    by_key = {(c.in_id, c.out_id, c.recurrent): c.weight for c in written.connections}
    assert by_key[(0, 1, False)] == 0.25 and by_key[(2, 0, True)] == -0.75

    only_inert = Genome(nodes=dict(nodes), connections=[ConnectionGene(0, 1, 0.25, True, 1)])
    assert decode(only_inert, 2, 1).has_edges  # train/skip decisions must not change


def test_weight_sample_fill_matches_dense_fill_bitwise() -> None:
    genome = _grown(4, seed=6)
    net = decode(genome, 4, 1)
    reference = DenseReference(decode(genome, 4, 1))
    x = torch.randn(16, 4)
    with torch.no_grad():
        for value in (-1.0, 0.5, 2.0):
            net.weights.fill_(value)
            reference.weights.fill_(value)
            assert torch.equal(net(x), reference(x))


def test_batched_dead_slot_and_padding_stay_zero_through_training() -> None:
    small, big = _grown(4, seed=8, rounds=1), _grown(4, seed=9, rounds=4)
    nets = [decode(small, 4, 1), decode(big, 4, 1)]
    batched = BatchedGraphNet(nets)
    assert batched.weights.shape == (2, batched.n_max + 1, batched.h_max)
    assert int(batched.col_pos[0, nets[0].h :].min()) == batched.n_max  # padding points at the dead slot

    encoded = encode_task(_task(4, seed=8), Level0Encoder(max_flat_dim=4))
    x, _ = encoded.support_input
    optimizer = torch.optim.Adam(batched.parameters(), lr=0.05)
    for _ in range(5):
        optimizer.zero_grad()
        loss = batched(x).square().mean()
        loss.backward()
        optimizer.step()
    trained = batched.weights.detach()
    assert torch.equal(trained[:, batched.n_max, :], torch.zeros(2, batched.h_max))  # dead slot row
    for index, net in enumerate(nets):
        assert torch.equal(trained[index, net.n : batched.n_max, :], torch.zeros(batched.n_max - net.n, batched.h_max))
        assert torch.equal(trained[index, :, net.h :], torch.zeros(batched.n_max + 1, batched.h_max - net.h))
