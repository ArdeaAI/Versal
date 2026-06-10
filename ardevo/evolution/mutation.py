"""Mutation operators: an independent stage, separate from crossover.

Each mutator is a single-purpose, registered function `(genome, ctx, *, rng, **params) -> Genome`.
`MutationPipeline` composes a config-ordered list of them, so individual operators are swapped
in or out via `[evolution.mutation].operators` with no code change. The shared `MutationContext`
hands out fresh node ids / innovation numbers and the activation palette.
"""

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
