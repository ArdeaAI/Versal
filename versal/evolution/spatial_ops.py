"""
Registered growth operators for compact neuron placements, without seeded motifs.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Any

from versal.evolution.crossover import CROSSOVER
from versal.evolution.genome import ConnectionGene, InnovationTracker, MacroGene, NodeGene, NodeKind, would_create_cycle
from versal.evolution.init import INIT
from versal.evolution.mutation import MUTATION, MutationContext
from versal.spatial import Binding, Placement, SpatialContract, SpatialGenome


def _edge(genome: SpatialGenome, source: int, target: int, tracker: InnovationTracker, rng: random.Random, rule: Binding | None = None, *, weight: float | None = None) -> None:
    marker = tracker.new_marker()
    if rule is None:
        rule = Binding(shift=rng.randrange(math.prod(genome.node_shape(source))))
    genome.connections.append(ConnectionGene(source, target, rng.gauss(0, 1) if weight is None else weight, True, marker))
    genome.bindings[marker] = rule


def spatial_minimal(n_inputs: int, n_outputs: int, *, rng: random.Random, contract: SpatialContract | None = None, default_activation: str = "tanh") -> SpatialGenome:
    if contract is None:
        raise ValueError("spatial initialization requires a task contract")
    nodes = {0: NodeGene(0, NodeKind.INPUT, "identity"), 1: NodeGene(1, NodeKind.BIAS, "identity"), 2: NodeGene(2, NodeKind.OUTPUT, "identity")}
    genome = SpatialGenome(nodes=nodes, contract=contract)
    tracker = InnovationTracker.from_genomes([genome])
    _edge(genome, 0, 2, tracker, rng)
    _edge(genome, 1, 2, tracker, rng)
    return genome


@MUTATION.register("spatial_add_node")
def add_node(genome: SpatialGenome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.2, fan_in: int = 3) -> SpatialGenome:
    if rng.random() >= prob:
        return genome
    child = genome.clone()
    node = ctx.innovations.new_node_id()
    child.nodes[node] = NodeGene(node, NodeKind.HIDDEN, ctx.default_activation)
    child.placements[node] = Placement()
    child.groups[max(child.groups, default=-1) + 1] = [node]
    # Sample scalar sources without a locality prior; even a faraway cell is one edge.
    sources = [*genome.input_ids, *genome.bias_ids, *genome.hidden_ids]
    for _ in range(rng.randint(1, max(1, fan_in))):
        _edge(child, rng.choice(sources), node, ctx.innovations, rng)
    _edge(child, node, rng.choice(genome.output_ids), ctx.innovations, rng)
    return child


@MUTATION.register("spatial_connect")
def connect(genome: SpatialGenome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.25) -> SpatialGenome:
    if rng.random() >= prob:
        return genome
    child = genome.clone()
    sources = [*child.input_ids, *child.bias_ids, *child.hidden_ids]
    targets = [node for node in [*child.hidden_ids, *child.output_ids] if node not in child.macro_output_ids]
    for _ in range(16):
        source, target = rng.choice(sources), rng.choice(targets)
        if not would_create_cycle(child, source, target):
            _edge(child, source, target, ctx.innovations, rng)
            return child
    return genome


@MUTATION.register("spatial_bind")
def bind(genome: SpatialGenome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.2) -> SpatialGenome:
    if rng.random() >= prob or not genome.connections:
        return genome
    child = genome.clone()
    edge = rng.choice(child.connections)
    rule = child.bindings[edge.innovation]
    source_shape, target_shape = child.node_shape(edge.in_id), child.node_shape(edge.out_id)
    source_size, target_size = math.prod(source_shape), math.prod(target_shape)
    action = rng.randrange(6)
    if action == 0:
        rule = replace(rule, shift=rng.randrange(source_size), matrix=(), offsets=())
    elif action == 1:
        rule = replace(rule, scale=rng.choice([-2, -1, 0, 1, 2]), matrix=(), offsets=())
    elif action == 2:
        rule = replace(rule, fan_in=rng.choice([1, 2, 3, 4, 0]), fan_stride=rng.choice([-2, -1, 1, 2]))
    elif action == 3:
        rule = replace(rule, target_start=rng.randrange(target_size), target_count=rng.choice([0, 1]), target_stride=rng.choice([1, 2, 3]))
    elif action == 4:
        # Every input axis can depend on every target axis; no fixed stencil or
        # channel/time/spatial dispatch. Mutation may eventually discover locality.
        matrix = tuple(tuple(rng.choice([-1, 0, 1]) for _ in target_shape) for _ in source_shape)
        offsets = tuple(rng.randrange(extent) for extent in source_shape)
        rule = replace(rule, matrix=matrix, offsets=offsets, fan_in=1)
    else:
        rule = replace(rule, shift=rule.shift + rng.choice([-1, 1]))
    child.bindings[edge.innovation] = rule
    child.parameters.pop(edge.innovation, None)
    return child


@MUTATION.register("spatial_repeat")
def repeat(genome: SpatialGenome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.1) -> SpatialGenome:
    if rng.random() >= prob or not genome.hidden_ids:
        return genome
    child = genome.clone()
    node = rng.choice(child.hidden_ids)
    members = next((members for members in child.groups.values() if node in members), [node])
    old = child.placements.get(node, Placement())
    assert child.contract is not None
    domain = rng.choice(["fixed", "input", "output"])
    if domain == "fixed":
        placement = Placement(count=max(1, old.size(child.contract) + rng.choice([-1, 1, 2])))
    else:
        dimensions = child.contract.input_shape if domain == "input" else child.contract.logit_shape
        axes = tuple(sorted(rng.sample(range(len(dimensions)), rng.randint(1, len(dimensions))))) if dimensions else ()
        placement = Placement(bank=domain, axes=axes)
    for member in members:
        child.placements[member] = placement
    for edge in child.connections:
        if edge.in_id in members or edge.out_id in members:
            child.parameters.pop(edge.innovation, None)
            rule = child.bindings[edge.innovation]
            # Domain rank changed: retain the flat rule until mutation discovers
            # another coordinate map. Internal circuit edges follow their copy.
            rule = replace(rule, matrix=(), offsets=())
            if edge.in_id in members and edge.out_id in members:
                rule = replace(rule, scale=1, shift=0, fan_in=1)
            child.bindings[edge.innovation] = rule
    return child


def split_group(genome: SpatialGenome, members: list[int], tracker: InnovationTracker, cut: int) -> SpatialGenome:
    """
    Split a repeated circuit into independent definitions without changing its function.
    """
    child = genome.clone()
    assert child.contract is not None
    placement = child.placements.get(members[0], Placement())
    if any(child.placements.get(node, Placement()) != placement for node in members):
        return genome
    low, high = placement.active(child.contract)
    if not low < cut < high:
        return genome
    mapping = {node: tracker.new_node_id() for node in members}
    for node, new in mapping.items():
        child.nodes[new] = replace(child.nodes[node], id=new)
        child.placements[node] = replace(placement, stop=cut)
        child.placements[new] = replace(placement, start=cut)
    for edge in list(child.connections):
        if edge.in_id not in mapping and edge.out_id not in mapping:
            continue
        sources = [edge.in_id, mapping[edge.in_id]] if edge.in_id in mapping else [edge.in_id]
        targets = [edge.out_id, mapping[edge.out_id]] if edge.out_id in mapping else [edge.out_id]
        # Internal bindings may cross the split (including recurrent and nonlocal
        # edges). Active placement intervals select each original scalar edge once.
        for source in sources:
            for target in targets:
                if (source, target) == (edge.in_id, edge.out_id):
                    continue
                marker = tracker.new_marker()
                child.connections.append(replace(edge, in_id=source, out_id=target, innovation=marker))
                child.bindings[marker] = child.bindings[edge.innovation]
                if edge.innovation in child.parameters:
                    child.parameters[marker] = list(child.parameters[edge.innovation])
    for macro in list(child.macros):
        if set(macro.input_node_ids + macro.output_node_ids).issubset(mapping):
            child.macros.append(
                replace(
                    macro,
                    input_node_ids=tuple(mapping[node] for node in macro.input_node_ids),
                    output_node_ids=tuple(mapping[node] for node in macro.output_node_ids),
                    innovation=tracker.new_marker(),
                )
            )
    for group, nodes in list(child.groups.items()):
        if set(nodes) & set(members):
            del child.groups[group]
    group_id = max(child.groups, default=-1) + 1
    child.groups[group_id], child.groups[group_id + 1] = list(members), list(mapping.values())
    return child


@MUTATION.register("spatial_split")
def split(genome: SpatialGenome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.05) -> SpatialGenome:
    if rng.random() >= prob or not genome.hidden_ids:
        return genome
    node = rng.choice(genome.hidden_ids)
    members = next((members for members in genome.groups.values() if node in members), [node])
    assert genome.contract is not None
    low, high = genome.placements.get(node, Placement()).active(genome.contract)
    return split_group(genome, members, ctx.innovations, rng.randrange(low + 1, high)) if high - low > 1 else genome


@MUTATION.register("spatial_share")
def share(genome: SpatialGenome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.08) -> SpatialGenome:
    if rng.random() >= prob or not genome.connections:
        return genome
    child = genome.clone()
    edge = rng.choice(child.connections)
    rule = child.bindings[edge.innovation]
    values = child.parameters.get(edge.innovation, [edge.weight])
    count = rule.upper_count(child.node_shape(edge.in_id), child.node_shape(edge.out_id))
    if count > 1_000_000:
        return genome
    child.parameters[edge.innovation] = [values[0]] * max(1, count) if rule.shared else [sum(values) / len(values)]
    child.bindings[edge.innovation] = replace(rule, shared=not rule.shared)
    return child


@MUTATION.register("spatial_prune")
def prune(genome: SpatialGenome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.1) -> SpatialGenome:
    if rng.random() >= prob or not genome.connections:
        return genome
    child = genome.clone()
    edge = rng.choice(child.connections)
    child.connections.remove(edge)
    child.bindings.pop(edge.innovation, None)
    child.parameters.pop(edge.innovation, None)
    used = {edge.in_id for edge in child.connections} | {edge.out_id for edge in child.connections}
    used.update(node for macro in child.macros for node in (*macro.input_node_ids, *macro.output_node_ids))
    for node in child.hidden_ids:
        if node not in used:
            del child.nodes[node]
            child.placements.pop(node, None)
    child.groups = {key: [node for node in members if node in child.nodes] for key, members in child.groups.items() if any(node in child.nodes for node in members)}
    return child


@MUTATION.register("spatial_weights")
def weights(genome: SpatialGenome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.4, sigma: float = 0.3) -> SpatialGenome:
    if rng.random() >= prob or not genome.connections:
        return genome
    child = genome.clone()
    edge = rng.choice(child.connections)
    values = child.parameters.get(edge.innovation, [edge.weight])
    child.parameters[edge.innovation] = [value + rng.gauss(0, sigma) for value in values]
    child.connections = [replace(item, weight=child.parameters[edge.innovation][0]) if item.innovation == edge.innovation else item for item in child.connections]
    return child


def restamp_spatial(genome: SpatialGenome, tracker: InnovationTracker) -> SpatialGenome:
    child = genome.clone()
    nodes = {node: tracker.new_node_id() for node in sorted(child.nodes)}
    markers = {edge.innovation: tracker.new_marker() for edge in child.connections}
    child.nodes = {nodes[node]: replace(value, id=nodes[node]) for node, value in child.nodes.items()}
    child.connections = [replace(edge, in_id=nodes[edge.in_id], out_id=nodes[edge.out_id], innovation=markers[edge.innovation]) for edge in child.connections]
    child.placements = {nodes[node]: value for node, value in child.placements.items()}
    child.bindings = {markers[marker]: value for marker, value in child.bindings.items() if marker in markers}
    child.parameters = {markers[marker]: value for marker, value in child.parameters.items() if marker in markers}
    child.groups = {group_id: [nodes[node] for node in members] for group_id, members in enumerate(child.groups.values())}
    child.macros = [
        replace(
            macro,
            input_node_ids=tuple(nodes[node] for node in macro.input_node_ids),
            output_node_ids=tuple(nodes[node] for node in macro.output_node_ids),
            innovation=tracker.new_marker(),
        )
        for macro in child.macros
    ]
    return child


def embed_entry(genome: SpatialGenome, entry: Any, tracker: InnovationTracker, rng: random.Random, *, exact: bool = False) -> SpatialGenome:
    """
    Import immutable ordinary modules, spatial modules, or compositions as circuits.
    """
    child = genome.clone()
    inputs = sum(int(row["width"]) for row in entry.io["inputs"])
    outputs = int(entry.io["output"]["width"])
    if entry.entry_type == "module":
        from versal.evolution.genome import genome_from_dict
        from versal.representation import module_ports

        inputs, outputs = module_ports(genome_from_dict(entry.payload))
    assert child.contract is not None
    if exact and (inputs, outputs) != (child.contract.input_width, child.contract.output_width):
        raise ValueError("stored module ports do not match the concrete tensor binding")
    if inputs + outputs > 4096:
        raise ValueError("imported circuit port count exceeds the placement budget")
    input_ids, output_ids = [], []
    for group, width in ((input_ids, inputs), (output_ids, outputs)):
        for _ in range(width):
            node = tracker.new_node_id()
            child.nodes[node] = NodeGene(node, NodeKind.HIDDEN, "identity")
            child.placements[node] = Placement()
            group.append(node)
    for index, node in enumerate(input_ids):
        rule = Binding(shift=index) if exact else None
        _edge(child, child.input_ids[0], node, tracker, rng, rule, weight=1.0 if exact else None)
    for index, node in enumerate(output_ids):
        rule = Binding(target_start=index, target_count=1) if exact else None
        _edge(child, node, child.output_ids[0], tracker, rng, rule, weight=1.0 if exact else None)
    marker = tracker.new_marker()
    child.macros.append(MacroGene(f"library:{entry.key}", tuple(input_ids), tuple(output_ids), marker, False))
    child.groups[max(child.groups, default=-1) + 1] = input_ids + output_ids
    return child


@MUTATION.register("spatial_reuse")
def reuse(genome: SpatialGenome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.05) -> SpatialGenome:
    if rng.random() >= prob or ctx.library is None:
        return genome
    rows = [row for row in ctx.library.summaries() if row.get("representation") != "field" and ctx.library.reference_subtree_depth(row["key"]) < ctx.max_inline_depth]
    if not rows:
        return genome
    try:
        return embed_entry(genome, ctx.library.load(rng.choice(rows)["key"]), ctx.innovations, rng)
    except ValueError:
        return genome


@CROSSOVER.register("spatial")
def crossover(left: SpatialGenome, right: SpatialGenome, *, rng: random.Random, **_params: Any) -> SpatialGenome:
    """
    Inherit one coherent recipe, recombining aligned weights and neuron definitions.
    """
    child = left.clone()
    right_edges = {edge.innovation: edge for edge in right.connections}
    for node in child.hidden_ids:
        if node in right.nodes and rng.random() < 0.5:
            child.nodes[node] = replace(child.nodes[node], activation=right.nodes[node].activation, aggregation=right.nodes[node].aggregation)
    for index, edge in enumerate(child.connections):
        other = right_edges.get(edge.innovation)
        if other is not None and (other.in_id, other.out_id) == (edge.in_id, edge.out_id) and rng.random() < 0.5:
            if child.bindings[edge.innovation] == right.bindings[edge.innovation]:
                child.connections[index] = replace(edge, weight=other.weight)
                if edge.innovation in right.parameters:
                    child.parameters[edge.innovation] = list(right.parameters[edge.innovation])
                else:
                    child.parameters.pop(edge.innovation, None)
    return child


@MUTATION.register("spatial_group")
def group(genome: SpatialGenome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.05) -> SpatialGenome:
    """
    Join connected neuron definitions into a circuit that can later repeat or split.
    """
    if rng.random() >= prob:
        return genome
    candidates = [
        edge
        for edge in genome.enabled_connections()
        if edge.in_id in genome.hidden_ids
        and edge.out_id in genome.hidden_ids
        and genome.placements.get(edge.in_id, Placement()) == genome.placements.get(edge.out_id, Placement())
    ]
    if not candidates:
        return genome
    child = genome.clone()
    edge = rng.choice(candidates)
    members = {edge.in_id, edge.out_id}
    for nodes in child.groups.values():
        if set(nodes) & members:
            members.update(nodes)
    child.groups = {key: nodes for key, nodes in child.groups.items() if not set(nodes) & members}
    child.groups[max(child.groups, default=-1) + 1] = sorted(members)
    return child


@MUTATION.register("spatial_recurrent")
def recurrent(genome: SpatialGenome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.04) -> SpatialGenome:
    if rng.random() >= prob:
        return genome
    child = genome.clone()
    targets = [node for node in [*child.hidden_ids, *child.output_ids] if node not in child.macro_output_ids]
    if not targets:
        return genome
    source, target = rng.choice([*child.hidden_ids, *child.output_ids]), rng.choice(targets)
    _edge(child, source, target, ctx.innovations, rng)
    child.connections[-1] = replace(child.connections[-1], recurrent=True)
    child.refine_steps = max(2, child.refine_steps)
    return child


INIT.register("spatial_minimal")(spatial_minimal)
