"""Mutation operators: an independent stage, separate from crossover.

Each mutator is a single-purpose, registered function `(genome, ctx, *, rng, **params) -> Genome`.
`MutationPipeline` composes a config-ordered list of them, so individual operators are swapped
in or out via `[evolution.mutation].operators` with no code change. The shared `MutationContext`
hands out fresh node ids / innovation numbers and the activation palette.
"""

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Callable

from ardevo.evolution.genome import (
    ConnectionGene,
    Genome,
    InnovationTracker,
    NodeGene,
    NodeKind,
    coordinate_distance,
    set_connection,
    would_create_cycle,
)
from ardevo.evolution.registry import Registry

Mutator = Callable[..., Genome]

MUTATION: Registry[Mutator] = Registry("mutation")


@dataclass
class MutationContext:
    """Per-run state mutators share: id/innovation allocation and the activation palette."""

    innovations: InnovationTracker
    activations: list[str]
    default_activation: str


@dataclass
class MutationPipeline:
    """Applies an ordered list of bound mutators in sequence."""

    operators: Sequence[Mutator]

    def __call__(self, genome: Genome, ctx: MutationContext, *, rng: random.Random) -> Genome:
        for operator in self.operators:
            genome = operator(genome, ctx, rng=rng)
        return genome


@MUTATION.register("perturb_weights")
def perturb_weights(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.8, sigma: float = 0.5) -> Genome:
    child = genome.clone()
    child.connections = [replace(conn, weight=conn.weight + rng.gauss(0.0, sigma)) if rng.random() < prob else conn for conn in child.connections]
    return child


@MUTATION.register("add_connection")
def add_connection(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.1) -> Genome:
    if rng.random() >= prob:
        return genome
    child = genome.clone()
    sources = [*child.input_ids, *child.bias_ids, *child.hidden_ids]
    targets = [*child.hidden_ids, *child.output_ids]
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
    return child  # graph is saturated; nothing to add


@MUTATION.register("add_node")
def add_node(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.03) -> Genome:
    enabled = genome.enabled_connections()
    if rng.random() >= prob or not enabled:
        return genome
    child = genome.clone()
    target = rng.choice(enabled)
    # NEAT split: disable the edge, route in -> new (weight 1) -> out (old weight).
    set_connection(child, replace(target, enabled=False))
    new_id = ctx.innovations.new_node_id()
    child.nodes[new_id] = NodeGene(new_id, NodeKind.HIDDEN, ctx.default_activation)
    child.connections.append(ConnectionGene(target.in_id, new_id, 1.0, True, ctx.innovations.innovation(target.in_id, new_id)))
    child.connections.append(ConnectionGene(new_id, target.out_id, target.weight, True, ctx.innovations.innovation(new_id, target.out_id)))
    return child


@MUTATION.register("add_rich_node")
def add_rich_node(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.1, fan_in: int = 4) -> Genome:
    """Add a hidden node wired from up to `fan_in` random sources and to every output.

    Unlike `add_node` (a single-edge split, which yields a one-input node that adds no capacity on
    tasks like parity), this node sees several inputs immediately, so gradient training can make it
    useful right away. Acyclic by construction: it draws sources from inputs/bias/hidden (never
    outputs) and feeds only outputs.
    """
    if rng.random() >= prob:
        return genome
    sources = [*genome.input_ids, *genome.bias_ids, *genome.hidden_ids]
    outputs = genome.output_ids
    if not sources or not outputs:
        return genome
    child = genome.clone()
    new_id = ctx.innovations.new_node_id()
    child.nodes[new_id] = NodeGene(new_id, NodeKind.HIDDEN, ctx.default_activation)
    for source in rng.sample(sources, min(fan_in, len(sources))):
        child.connections.append(ConnectionGene(source, new_id, rng.gauss(0.0, 1.0), True, ctx.innovations.innovation(source, new_id)))
    for output in outputs:
        child.connections.append(ConnectionGene(new_id, output, rng.gauss(0.0, 1.0), True, ctx.innovations.innovation(new_id, output)))
    return child


@MUTATION.register("add_deep_node")
def add_deep_node(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.1, fan_in: int = 4, fan_out: int = 3) -> Genome:
    """Add a hidden node that feeds OTHER hidden nodes (plus the outputs), building depth.

    `add_rich_node` only wires new nodes to the outputs, so it can only widen a single hidden layer.
    Tasks like two-spirals need depth (hidden -> hidden). This node draws from `fan_in` sources and
    feeds every output (a guaranteed readout) plus up to `fan_out` existing hidden nodes, skipping any
    target that would create a cycle.
    """
    if rng.random() >= prob:
        return genome
    sources = [*genome.input_ids, *genome.bias_ids, *genome.hidden_ids]
    outputs = genome.output_ids
    if not sources or not outputs:
        return genome
    child = genome.clone()
    new_id = ctx.innovations.new_node_id()
    child.nodes[new_id] = NodeGene(new_id, NodeKind.HIDDEN, ctx.default_activation)
    for source in rng.sample(sources, min(fan_in, len(sources))):
        child.connections.append(ConnectionGene(source, new_id, rng.gauss(0.0, 1.0), True, ctx.innovations.innovation(source, new_id)))
    for output in outputs:
        child.connections.append(ConnectionGene(new_id, output, rng.gauss(0.0, 1.0), True, ctx.innovations.innovation(new_id, output)))
    hidden_targets = [node_id for node_id in child.hidden_ids if node_id != new_id]
    rng.shuffle(hidden_targets)
    added = 0
    for target in hidden_targets:
        if added >= fan_out:
            break
        if child.has_connection(new_id, target) or would_create_cycle(child, new_id, target):
            continue
        child.connections.append(ConnectionGene(new_id, target, rng.gauss(0.0, 1.0), True, ctx.innovations.innovation(new_id, target)))
        added += 1
    return child


@MUTATION.register("mutate_activation")
def mutate_activation(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.05) -> Genome:
    # Hidden nodes only: outputs stay linear readouts so they emit raw logits.
    candidates = genome.hidden_ids
    if rng.random() >= prob or not candidates:
        return genome
    child = genome.clone()
    node_id = rng.choice(candidates)
    child.nodes[node_id] = replace(child.nodes[node_id], activation=rng.choice(ctx.activations))
    return child


@MUTATION.register("toggle_connection")
def toggle_connection(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.0) -> Genome:
    if rng.random() >= prob or not genome.connections:
        return genome
    child = genome.clone()
    index = rng.randrange(len(child.connections))
    conn = child.connections[index]
    if conn.enabled:
        child.connections[index] = replace(conn, enabled=False)
    elif not would_create_cycle(child, conn.in_id, conn.out_id):
        # Only re-enable a disabled edge when doing so keeps the graph feedforward.
        child.connections[index] = replace(conn, enabled=True)
    return child


# --- geometry-biased operators -------------------------------------------------------------------
# These read the axis-coordinates the multi-task substrate stamps on input/hidden nodes and bias
# growth toward LOCAL structure (receptive fields, repeated motifs). `coordinate_distance` returns
# inf across incomparable banks/axis-signatures, so a binary bit and a continuous coordinate never
# land in the same receptive field even though they share one growing topology. On the flat single-
# task path (no coordinates) these operators no-op, leaving the non-local operators to do the work.


def _weighted_choice(items: list[tuple[int, int]], weights: list[float], rng: random.Random) -> tuple[int, int]:
    total = sum(weights)
    if total <= 0.0:
        return rng.choice(items)
    threshold = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += weight
        if cumulative >= threshold:
            return item
    return items[-1]


def _centroid(coords: list[tuple[float, ...]]) -> tuple[float, ...]:
    count = len(coords)
    dims = len(coords[0])
    return tuple(sum(coord[dim] for coord in coords) / count for dim in range(dims))


def _nearest(genome: Genome, anchor: tuple[float, ...] | None, candidates: list[int], k: int) -> list[int]:
    scored = [(coordinate_distance(genome.nodes[node_id].coordinate, anchor), node_id) for node_id in candidates]
    finite = sorted((pair for pair in scored if not math.isinf(pair[0])), key=lambda pair: pair[0])
    return [node_id for _distance, node_id in finite[:k]]


@MUTATION.register("add_local_connection")
def add_local_connection(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.1, radius: float = 1.5) -> Genome:
    """Like `add_connection`, but bias the new edge toward coordinate-close (same-bank) node pairs.

    `radius` is in node-coordinate (axis-index) units: the substrate stamps raw unraveled indices, so
    a radius of ~1-2 favors immediate neighbors (a local receptive field) over distant ones.
    """
    if rng.random() >= prob:
        return genome
    child = genome.clone()
    sources = [*child.input_ids, *child.bias_ids, *child.hidden_ids]
    targets = [*child.hidden_ids, *child.output_ids]
    candidates: list[tuple[int, int]] = []
    weights: list[float] = []
    for source in sources:
        for target in targets:
            if source == target or child.has_connection(source, target) or would_create_cycle(child, source, target):
                continue
            distance = coordinate_distance(child.nodes[source].coordinate, child.nodes[target].coordinate)
            if math.isinf(distance):
                continue  # incomparable banks: leave it to the non-local add_connection
            candidates.append((source, target))
            weights.append(math.exp(-distance / max(radius, 1e-6)))
    if not candidates:
        return child
    source, target = _weighted_choice(candidates, weights, rng)
    child.connections.append(ConnectionGene(source, target, rng.gauss(0.0, 1.0), True, ctx.innovations.innovation(source, target)))
    return child


@MUTATION.register("add_local_node")
def add_local_node(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.1, fan_in: int = 4) -> Genome:
    """Grow a hidden node from a LOCAL receptive field: its fan-in is the nearest coordinate-neighbors
    of a seed source, and it sits at their centroid. Reads out to every output head (a shared feature).
    """
    if rng.random() >= prob:
        return genome
    sources = [*genome.input_ids, *genome.bias_ids, *genome.hidden_ids]
    outputs = genome.output_ids
    coordinated = [node_id for node_id in sources if genome.nodes[node_id].coordinate is not None]
    if not coordinated or not outputs:
        return genome  # nothing to be local about; leave it to add_rich_node
    child = genome.clone()
    seed = rng.choice(coordinated)
    field = _nearest(child, child.nodes[seed].coordinate, coordinated, fan_in)
    coords = [coord for node_id in field if (coord := child.nodes[node_id].coordinate) is not None]
    if not coords:
        return child
    new_id = ctx.innovations.new_node_id()
    child.nodes[new_id] = NodeGene(new_id, NodeKind.HIDDEN, ctx.default_activation, _centroid(coords))
    for source in field:
        child.connections.append(ConnectionGene(source, new_id, rng.gauss(0.0, 1.0), True, ctx.innovations.innovation(source, new_id)))
    for output in outputs:
        child.connections.append(ConnectionGene(new_id, output, rng.gauss(0.0, 1.0), True, ctx.innovations.innovation(new_id, output)))
    return child


@MUTATION.register("add_shared_motif")
def add_shared_motif(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.05, copies: int = 2) -> Genome:
    """Replicate an existing local detector's motif at other coordinate locations (independent weights).

    A structural convolution prior: take a hidden node's local fan-in size and output readouts, and
    grow `copies` siblings centered on other seeds, each reading its own local neighborhood. Weights
    are NOT tied (hard weight-sharing would touch the substrate); only the connectivity pattern repeats.
    """
    if rng.random() >= prob:
        return genome
    hidden = [node_id for node_id in genome.hidden_ids if genome.nodes[node_id].coordinate is not None]
    if not hidden:
        return genome
    child = genome.clone()
    template = rng.choice(hidden)
    incoming = [conn.in_id for conn in child.enabled_connections() if conn.out_id == template and child.nodes[conn.in_id].coordinate is not None]
    outputs = [conn.out_id for conn in child.enabled_connections() if conn.in_id == template and child.nodes[conn.out_id].kind is NodeKind.OUTPUT]
    if not incoming or not outputs:
        return child
    field_size = len(incoming)
    sources = [node_id for node_id in (*child.input_ids, *child.hidden_ids) if child.nodes[node_id].coordinate is not None]
    for seed in rng.sample(sources, min(copies, len(sources))):
        field = _nearest(child, child.nodes[seed].coordinate, sources, field_size)
        coords = [coord for node_id in field if (coord := child.nodes[node_id].coordinate) is not None]
        if not coords:
            continue
        new_id = ctx.innovations.new_node_id()
        child.nodes[new_id] = NodeGene(new_id, NodeKind.HIDDEN, child.nodes[template].activation, _centroid(coords))
        for source in field:
            if not would_create_cycle(child, source, new_id):
                child.connections.append(ConnectionGene(source, new_id, rng.gauss(0.0, 1.0), True, ctx.innovations.innovation(source, new_id)))
        for output in outputs:
            child.connections.append(ConnectionGene(new_id, output, rng.gauss(0.0, 1.0), True, ctx.innovations.innovation(new_id, output)))
    return child
