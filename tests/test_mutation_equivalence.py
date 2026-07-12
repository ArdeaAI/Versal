"""The optimized mutators must be BITWISE-identical to the O(pairs x E) originals they replaced.

The reference implementations below are verbatim copies of the pre-optimization op bodies (the
per-pair `has_connection` scan + per-pair `would_create_cycle` BFS), the same pattern as
test_substrate_slim.py's DenseReference: the reference IS the old algorithm, so slim-vs-reference
equality proves the refactor changed wall-clock only. The removed scans never draw rng, so the
rng streams cannot diverge either."""

import math
import random
import time

from ardevo.evolution.genome import ConnectionGene, Genome, InnovationTracker, MacroGene, NodeGene, NodeKind, coordinate_distance, genome_to_dict, would_create_cycle
from ardevo.evolution.init import minimal, stamp_input_coordinates
from ardevo.evolution.mutation import (
    MutationContext,
    _weighted_choice,
    add_connection,
    add_local_connection,
    add_local_node,
    add_recurrent_connection,
    add_rich_node,
    add_shared_motif,
)


def _reference_add_connection(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.1) -> Genome:
    if rng.random() >= prob:
        return genome
    child = genome.clone()
    sources = [*child.input_ids, *child.bias_ids, *child.hidden_ids]
    targets = [node_id for node_id in (*child.hidden_ids, *child.output_ids) if node_id not in child.macro_output_ids]
    rng.shuffle(sources)
    rng.shuffle(targets)
    for source in sources:
        for target in targets:
            if source == target or child.has_connection(source, target):
                continue
            if would_create_cycle(child, source, target):
                continue
            innovation = ctx.innovations.innovation(source, target)
            child.connections.append(ConnectionGene(source, target, rng.gauss(0.0, 1.0), True, innovation))
            return child
    return child


def _reference_add_local_connection(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.1, radius: float = 1.5) -> Genome:
    if rng.random() >= prob:
        return genome
    child = genome.clone()
    sources = [*child.input_ids, *child.bias_ids, *child.hidden_ids]
    targets = [node_id for node_id in (*child.hidden_ids, *child.output_ids) if node_id not in child.macro_output_ids]
    candidates: list[tuple[int, int]] = []
    weights: list[float] = []
    for source in sources:
        for target in targets:
            if source == target or child.has_connection(source, target) or would_create_cycle(child, source, target):
                continue
            distance = coordinate_distance(child.nodes[source].coordinate, child.nodes[target].coordinate)
            if math.isinf(distance):
                continue
            candidates.append((source, target))
            weights.append(math.exp(-distance / max(radius, 1e-6)))
    if not candidates:
        return child
    source, target = _weighted_choice(candidates, weights, rng)
    child.connections.append(ConnectionGene(source, target, rng.gauss(0.0, 1.0), True, ctx.innovations.innovation(source, target)))
    return child


def _grid_genome(side: int, seed: int, rounds: int = 3) -> Genome:
    rng = random.Random(seed)
    genome = minimal(side * side, 1, rng=rng, default_activation="tanh", weight_scale=1.0)
    genome = stamp_input_coordinates(genome, (side, side))
    ctx = MutationContext(InnovationTracker.from_genomes([genome]), ["tanh", "relu"], "tanh")
    for _ in range(rounds):
        genome = add_rich_node(genome, ctx, rng=rng, prob=1.0, fan_in=4)
        genome = add_local_node(genome, ctx, rng=rng, prob=1.0, fan_in=4)  # coordinated hidden: motif templates
        genome = add_recurrent_connection(genome, ctx, rng=rng, prob=1.0)
    return genome


def _macro_genome() -> Genome:
    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity", (0.0,)),
        1: NodeGene(1, NodeKind.INPUT, "identity", (1.0,)),
        2: NodeGene(2, NodeKind.BIAS, "identity"),
        3: NodeGene(3, NodeKind.OUTPUT, "identity", (0.5,)),
        4: NodeGene(4, NodeKind.HIDDEN, "identity", (0.2,)),
        5: NodeGene(5, NodeKind.HIDDEN, "identity", (0.8,)),
    }
    connections = [
        ConnectionGene(4, 3, 0.5, True, 0),
        ConnectionGene(5, 3, -0.5, True, 1),
    ]
    macros = [MacroGene(ref="library:m1_fake", input_node_ids=(0, 1), output_node_ids=(4, 5), innovation=99)]
    return Genome(nodes=nodes, connections=connections, macros=macros)


def _saturated_genome() -> Genome:
    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity"),
        1: NodeGene(1, NodeKind.BIAS, "identity"),
        2: NodeGene(2, NodeKind.OUTPUT, "identity"),
    }
    connections = [ConnectionGene(0, 2, 1.0, True, 0), ConnectionGene(1, 2, 0.5, True, 1)]
    return Genome(nodes=nodes, connections=connections)


def _cases() -> list[Genome]:
    return [_grid_genome(8, seed=1), _grid_genome(8, seed=2, rounds=5), _macro_genome(), _saturated_genome()]


def test_add_connection_matches_reference_bitwise() -> None:
    for index, genome in enumerate(_cases()):
        ctx_a = MutationContext(InnovationTracker.from_genomes([genome]), ["tanh"], "tanh")
        ctx_b = MutationContext(InnovationTracker.from_genomes([genome]), ["tanh"], "tanh")
        for seed in range(8):
            fast = add_connection(genome, ctx_a, rng=random.Random(seed), prob=1.0)
            slow = _reference_add_connection(genome, ctx_b, rng=random.Random(seed), prob=1.0)
            assert genome_to_dict(fast) == genome_to_dict(slow), f"case {index} seed {seed}"


def test_add_local_connection_matches_reference_bitwise() -> None:
    for index, genome in enumerate(_cases()):
        ctx_a = MutationContext(InnovationTracker.from_genomes([genome]), ["tanh"], "tanh")
        ctx_b = MutationContext(InnovationTracker.from_genomes([genome]), ["tanh"], "tanh")
        for seed in range(8):
            fast = add_local_connection(genome, ctx_a, rng=random.Random(seed), prob=1.0)
            slow = _reference_add_local_connection(genome, ctx_b, rng=random.Random(seed), prob=1.0)
            assert genome_to_dict(fast) == genome_to_dict(slow), f"case {index} seed {seed}"


def test_add_shared_motif_still_deterministic_per_seed() -> None:
    """add_shared_motif's cycle check is structurally always-False mid-loop (the new node is a
    sink until its output edges land after the field loop); the optimization must keep results
    seed-stable and identical across repeated calls."""
    genome = _grid_genome(8, seed=3, rounds=4)
    landed = 0
    for seed in range(12):
        ctx = MutationContext(InnovationTracker.from_genomes([genome]), ["tanh"], "tanh")
        first = add_shared_motif(genome, ctx, rng=random.Random(seed), prob=1.0, copies=2)
        ctx_again = MutationContext(InnovationTracker.from_genomes([genome]), ["tanh"], "tanh")
        second = add_shared_motif(genome, ctx_again, rng=random.Random(seed), prob=1.0, copies=2)
        assert genome_to_dict(first) == genome_to_dict(second), f"seed {seed}"
        landed += len(first.connections) > len(genome.connections)
    assert landed > 0  # at least one seed actually replicated the motif


def test_add_local_connection_speed_guard() -> None:
    """The regression this file exists to prevent: ONE call at CIFAR width was 4.9s pre-fix."""
    rng = random.Random(0)
    genome = minimal(3072, 1, rng=rng, default_activation="tanh", weight_scale=1.0)
    genome = stamp_input_coordinates(genome, (3, 32, 32))
    ctx = MutationContext(InnovationTracker.from_genomes([genome]), ["tanh"], "tanh")
    for _ in range(4):
        genome = add_rich_node(genome, ctx, rng=rng, prob=1.0, fan_in=4)
    start = time.perf_counter()
    add_local_connection(genome, ctx, rng=random.Random(1), prob=1.0)
    assert time.perf_counter() - start < 1.0


def _reference_make_acyclic(genome: Genome) -> Genome:
    """Verbatim pre-optimization make_acyclic: per-edge would_create_cycle on the growing kept genome."""
    from dataclasses import replace

    kept = Genome(nodes=dict(genome.nodes), connections=[], macros=list(genome.macros))
    repaired = []
    for conn in genome.connections:
        if not conn.enabled or conn.recurrent:
            repaired.append(conn)
        elif conn.in_id == conn.out_id or would_create_cycle(kept, conn.in_id, conn.out_id):
            repaired.append(replace(conn, enabled=False))
        else:
            kept.connections.append(conn)
            repaired.append(conn)
    return Genome(nodes=dict(genome.nodes), connections=repaired, macros=list(genome.macros), refine_steps=genome.refine_steps)


def test_make_acyclic_matches_reference_on_cyclic_and_acyclic_genomes() -> None:
    from ardevo.evolution.genome import make_acyclic

    for genome in _cases():
        acyclic = make_acyclic(genome)
        assert genome_to_dict(acyclic) == genome_to_dict(_reference_make_acyclic(genome))  # fast path == repair

        hidden = sorted(genome.hidden_ids)
        if len(hidden) >= 2:
            cyclic = genome.clone()
            # Close a deliberate forward cycle (later gene loses under ordered repair) + a self-loop.
            cyclic.connections.append(ConnectionGene(hidden[1], hidden[0], 1.0, True, 9001))
            cyclic.connections.append(ConnectionGene(hidden[0], hidden[1], 1.0, True, 9002))
            cyclic.connections.append(ConnectionGene(hidden[0], hidden[0], 1.0, True, 9003))
            assert genome_to_dict(make_acyclic(cyclic)) == genome_to_dict(_reference_make_acyclic(cyclic))


def test_make_acyclic_fast_path_speed_guard() -> None:
    """The module pool runs make_acyclic per child per generation; the second parity wedge
    (2026-07-05) was per-edge adjacency rebuilds here. Acyclic genomes must repair in O(V+E)."""
    import time

    from ardevo.evolution.genome import make_acyclic

    genome = _grid_genome(16, seed=11, rounds=6)  # a few hundred genes
    start = time.perf_counter()
    for _ in range(200):
        make_acyclic(genome)
    assert (time.perf_counter() - start) / 200 < 0.005
