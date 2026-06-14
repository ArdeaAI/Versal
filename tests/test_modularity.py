"""Phase 7 Pillar C: the modularity_bonus fitness term and grid-shape detection.

modularity_bonus rewards LOCAL/tiled edges (the gradient toward weight-tied conv kernels) computed
from a genome's coordinate geometry. It is inert on the flat path (no coordinates), so it only bites
once the direct strategy stamps grid coordinates on image rungs."""

from collections.abc import Mapping

from ardevo.dataset.icarus import Task
from ardevo.evolution.fitness import modularity_bonus
from ardevo.evolution.genome import ConnectionGene, Genome, NodeGene, NodeKind
from ardevo.strategy import DirectStrategy
from tests.test_spatial import _grid_to_grid_task


def _coord_genome(coords: Mapping[int, tuple[float, float] | None], edges: list[tuple[int, int]]) -> Genome:
    nodes = {
        node_id: NodeGene(node_id, NodeKind.INPUT if node_id < 90 else NodeKind.OUTPUT, "identity", coordinate)
        for node_id, coordinate in coords.items()
    }
    connections = [ConnectionGene(a, b, 1.0, True, innovation) for innovation, (a, b) in enumerate(edges)]
    return Genome(nodes=nodes, connections=connections)


def test_modularity_bonus_is_zero_without_coordinates(linear_genome) -> None:
    assert modularity_bonus(linear_genome, {}) == 0.0  # no node carries geometry -> inert on the flat path


def test_modularity_bonus_rewards_local_edges() -> None:
    # Three coordinated edges: two within radius 2 (local), one far (cross-region) -> 2/3.
    coords = {0: (0.0, 0.0), 1: (0.0, 1.0), 2: (5.0, 5.0), 3: (0.0, 0.5)}  # 3 is a hidden detector site
    genome = _coord_genome(coords, edges=[(0, 3), (1, 3), (2, 3)])
    genome.nodes[3] = NodeGene(3, NodeKind.HIDDEN, "tanh", (0.0, 0.5))
    assert abs(modularity_bonus(genome, {}) - 2.0 / 3.0) < 1e-9


def test_modularity_bonus_is_one_when_all_edges_are_local() -> None:
    coords = {0: (0.0, 0.0), 1: (0.0, 1.0), 3: (0.0, 0.5)}
    genome = _coord_genome(coords, edges=[(0, 3), (1, 3)])
    genome.nodes[3] = NodeGene(3, NodeKind.HIDDEN, "tanh", (0.0, 0.5))
    assert modularity_bonus(genome, {}) == 1.0


def test_modularity_bonus_skips_uncoordinated_edges() -> None:
    # Edge to an output with no coordinate is incomparable (skipped), leaving one local edge -> 1.0.
    coords = {0: (0.0, 0.0), 1: (0.0, 1.0), 90: None}
    genome = _coord_genome(coords, edges=[(0, 1), (0, 90)])
    assert modularity_bonus(genome, {}) == 1.0


def test_direct_strategy_detects_grid_shape_only_for_grids(xor_task: Task) -> None:
    grid_task = _grid_to_grid_task(height=4, width=3)
    assert DirectStrategy._grid_shape(grid_task) == (4, 3)  # 2D input -> a grid to stamp coordinates on
    assert DirectStrategy._grid_shape(xor_task) is None  # flat classification -> no geometry
