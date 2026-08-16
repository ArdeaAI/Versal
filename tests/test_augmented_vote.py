"""Phase-4: the augmented_vote evaluate op (test-time dihedral voting, dispatched on structure,
falling back to hybrid off-grid) and the relational-bottleneck mutation primitive."""

import random

import torch

from versal.dataset.icarus import Axis, Field, Level0Encoder, Task, TaskKind, TaskMeta, ValueType
from versal.evaluation import encode, input_width, output_features
from versal.evolution.evaluate import _d4_index_maps, _voted_raw, augmented_vote, hybrid
from versal.evolution.evolver import TaskAdapter
from versal.evolution.genome import ConnectionGene, Genome, InnovationTracker, NodeGene, NodeKind
from versal.evolution.init import minimal
from versal.evolution.mutation import MutationContext, add_relation_node
from versal.substrate import decode


def _grid_task(cells: int = 9) -> Task:
    rng = random.Random(0)
    pairs = []
    for _ in range(8):
        grid = torch.tensor([[rng.random() for _ in range(3)] for _ in range(3)])
        label = torch.tensor([float(int(grid.sum() > 4.5))])
        x = Field(grid, (Axis.HEIGHT, Axis.WIDTH), ValueType.CONTINUOUS, None, (0.0, 1.0), None)
        y = Field(label, (Axis.EXTRA,), ValueType.CATEGORICAL, 2, None, None)
        pairs.append((x, y))
    meta = TaskMeta(rung=0, kind=TaskKind.MAP, name="grid_sum", fixed_split=True)
    return Task(meta=meta, support=pairs[:6], query=pairs[6:])


def _grid_adapter() -> TaskAdapter:
    task = _grid_task()
    encoder = Level0Encoder(max_flat_dim=9)
    encoded = encode(task, encoder)
    return TaskAdapter(encoded, encoder, input_width(encoded), output_features(encoded), grid_shape=(3, 3))


def test_d4_maps_square_grid_has_eight_bijective_views() -> None:
    maps = _d4_index_maps((3, 3))
    assert len(maps) == 8
    assert all(sorted(view.tolist()) == list(range(9)) for view in maps)
    rect = _d4_index_maps((2, 4))
    assert len(rect) == 4  # only shape-preserving views off-square


def test_augmented_vote_falls_back_to_hybrid_off_grid(xor_adapter: TaskAdapter) -> None:
    genome = minimal(2, 1, rng=random.Random(0))
    module = decode(genome, 2, 1)
    voted = augmented_vote(genome, module, xor_adapter)
    plain = hybrid(genome, module, xor_adapter)
    assert voted == plain and "vote_views" not in voted


def test_augmented_vote_matches_unvoted_on_a_permutation_invariant_module() -> None:
    adapter = _grid_adapter()
    genome = minimal(9, 2, rng=random.Random(0))
    # Equal weights per output: the readout sums all pixels, so every dihedral view is identical.
    genome.connections = [ConnectionGene(conn.in_id, conn.out_id, 0.5 if conn.in_id < 9 else 0.1, conn.enabled, conn.innovation) for conn in genome.connections]
    module = decode(genome, 9, 2)
    result = augmented_vote(genome, module, adapter)
    assert result["vote_views"] == 8.0
    assert abs(result["query_accuracy"] - result["unvoted_query_accuracy"]) < 1e-9
    assert abs(result["support_loss"] - result["unvoted_support_loss"]) < 1e-9


def test_grid_to_grid_votes_are_inverse_permuted_back() -> None:
    # Identity map input cell i -> output cell i: perfectly equivariant, so the accumulated votes
    # in original layout must equal the plain forward exactly.
    nodes: dict[int, NodeGene] = {}
    for index in range(9):
        nodes[index] = NodeGene(index, NodeKind.INPUT, "identity")
    nodes[9] = NodeGene(9, NodeKind.BIAS, "identity")
    for index in range(10, 19):
        nodes[index] = NodeGene(index, NodeKind.OUTPUT, "identity")
    connections = [ConnectionGene(index, index + 10, 1.0, True, index) for index in range(9)]
    genome = Genome(nodes=nodes, connections=connections)
    module = decode(genome, 9, 9)
    x = torch.rand(4, 9)
    maps = _d4_index_maps((3, 3))
    votes = _voted_raw(module, x, maps, maps, 9)
    with torch.no_grad():
        assert torch.allclose(votes, module(x), atol=1e-6)


def test_add_relation_node_builds_a_product_similarity_unit() -> None:
    genome = minimal(4, 1, rng=random.Random(0))
    ctx = MutationContext(innovations=InnovationTracker.from_genomes([genome]), activations=["tanh"], default_activation="tanh")
    child = add_relation_node(genome, ctx, rng=random.Random(1), prob=1.0, fan_in=2)
    new_nodes = [node_id for node_id in child.hidden_ids]
    assert len(new_nodes) == 1
    relation = child.nodes[new_nodes[0]]
    assert relation.aggregation == "product" and relation.activation == "tanh"
    fan_in = [conn for conn in child.connections if conn.out_id == relation.id]
    assert len(fan_in) == 2 and all(child.nodes[conn.in_id].kind is not NodeKind.BIAS for conn in fan_in)
    out = decode(child, 4, 1)(torch.rand(3, 4))
    assert out.shape == (3, 1)
