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
from typing import TYPE_CHECKING, Callable

from ardevo.evolution.genome import (
    ConnectionGene,
    Genome,
    InnovationTracker,
    MacroGene,
    NodeGene,
    NodeKind,
    coordinate_distance,
    set_connection,
    would_create_cycle,
)
from ardevo.evolution.registry import Registry

if TYPE_CHECKING:
    from ardevo.library import ModuleLibrary

Mutator = Callable[..., Genome]

MUTATION: Registry[Mutator] = Registry("mutation")


@dataclass
class MutationContext:
    """Per-run state mutators share: id/innovation allocation and the activation palette.

    `library` is the LIVE library handle when the owning loop has one (the hierarchical loop passes
    it so library-reading mutators see entries admitted mid-run); the flat path leaves it None and
    those mutators fall back to a by-path cached snapshot."""

    innovations: InnovationTracker
    activations: list[str]
    default_activation: str
    library: "ModuleLibrary | None" = None


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
    return child  # graph is saturated; nothing to add


@MUTATION.register("add_node")
def add_node(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.03) -> Genome:
    enabled = genome.enabled_connections()
    if rng.random() >= prob or not enabled:
        return genome
    child = genome.clone()
    target = rng.choice(enabled)
    # NEAT split: disable the edge, route in -> new (weight 1) -> out (old weight). Splitting a
    # RECURRENT edge keeps the time delay on the INCOMING half: `in -(recurrent)-> new -(forward)-> out`
    # delivers in@t-1 to out@t exactly like the original gene, and creates no forward cycle even for
    # self-loops (the only forward edge is new -> out).
    set_connection(child, replace(target, enabled=False))
    new_id = ctx.innovations.new_node_id()
    child.nodes[new_id] = NodeGene(new_id, NodeKind.HIDDEN, ctx.default_activation)
    child.connections.append(ConnectionGene(target.in_id, new_id, 1.0, True, ctx.innovations.innovation(target.in_id, new_id, target.recurrent), recurrent=target.recurrent))
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
    hidden_targets = [node_id for node_id in child.hidden_ids if node_id != new_id and node_id not in child.macro_output_ids]
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
    # Hidden nodes only: outputs stay linear readouts so they emit raw logits. Macro output stubs
    # are excluded: their values come from the inner network and must pass through unchanged.
    candidates = [node_id for node_id in genome.hidden_ids if node_id not in genome.macro_output_ids]
    if rng.random() >= prob or not candidates:
        return genome
    child = genome.clone()
    node_id = rng.choice(candidates)
    child.nodes[node_id] = replace(child.nodes[node_id], activation=rng.choice(ctx.activations))
    return child


@MUTATION.register("add_recurrent_connection")
def add_recurrent_connection(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.05, self_loop_bias: float = 0.5) -> Genome:
    """Add a TIME-DELAYED edge reading the previous step's value. Self-loops are the memory
    primitive (an accumulator), so they are sampled with their own bias. No cycle check: recurrent
    edges are exempt by definition. Inert under the plain GraphNet, so the operator stays pure even
    on non-temporal runs."""
    if rng.random() >= prob:
        return genome
    stateful = [*genome.hidden_ids, *genome.output_ids]
    if not stateful:
        return genome
    child = genome.clone()
    stateful_targets = [node_id for node_id in (*child.hidden_ids, *child.output_ids) if node_id not in child.macro_output_ids]
    loop_candidates = [node_id for node_id in child.hidden_ids if node_id not in child.macro_output_ids]
    if not stateful_targets:
        return genome
    if rng.random() < self_loop_bias and loop_candidates:
        source = target = rng.choice(loop_candidates)
    else:
        source = rng.choice(stateful)
        target = rng.choice(stateful_targets)
    if child.has_connection(source, target, recurrent=True):
        return child
    innovation = ctx.innovations.innovation(source, target, recurrent=True)
    child.connections.append(ConnectionGene(source, target, rng.gauss(0.0, 1.0), True, innovation, recurrent=True))
    return child


@MUTATION.register("mutate_aggregation")
def mutate_aggregation(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.05, max_fan_in: int = 4) -> Genome:
    """Flip a hidden node between sum and product aggregation.

    Sum -> product only for nodes with 2..max_fan_in enabled incoming edges (a one-input product is
    just scaling, and high fan-in products explode numerically). Product -> sum is always allowed so
    a product node that lost edges can recover. Outputs stay sum so they remain linear readouts.
    """
    if rng.random() >= prob:
        return genome
    fan_in: dict[int, int] = {}
    for conn in genome.enabled_connections():
        fan_in[conn.out_id] = fan_in.get(conn.out_id, 0) + 1
    candidates = [
        node_id
        for node_id in genome.hidden_ids
        if node_id not in genome.macro_output_ids and (genome.nodes[node_id].aggregation == "product" or 2 <= fan_in.get(node_id, 0) <= max_fan_in)
    ]
    if not candidates:
        return genome
    child = genome.clone()
    node_id = rng.choice(candidates)
    node = child.nodes[node_id]
    child.nodes[node_id] = replace(node, aggregation="sum" if node.aggregation == "product" else "product")
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
    elif conn.recurrent or not would_create_cycle(child, conn.in_id, conn.out_id):
        # Re-enable freely for recurrent edges (time-delayed, cycle-exempt); forward edges only when
        # doing so keeps the graph feedforward.
        child.connections[index] = replace(conn, enabled=True)
    return child


_LIBRARY_CACHE: dict[str, object] = {}


def _cached_library(path: str) -> "object":
    # Imported lazily and cached per path: the mutation fires every generation and must not re-read
    # the index from disk each time. ardevo.library imports nothing from this module, so this is safe.
    if path not in _LIBRARY_CACHE:
        from ardevo.library import ModuleLibrary

        _LIBRARY_CACHE[path] = ModuleLibrary(path)
    return _LIBRARY_CACHE[path]


@MUTATION.register("add_library_module")
def add_library_module(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.05, path: str = "library") -> Genome:
    """Inline a stored library module into the host genome as evolved structure.

    Ports become HIDDEN identity pass-throughs (each in-port reads one random host source at weight
    1.0 so the module initially sees a raw signal; out-ports read out to every host output with small
    random weights), the module's bias edges remap onto the host bias node, and internal weights /
    activations / aggregations / recurrence arrive intact. Acyclic by construction: hosts feed ports,
    ports feed module internals, out-ports feed only host outputs.
    """
    if rng.random() >= prob:
        return genome
    from ardevo.library import MODULE as MODULE_ENTRY
    from ardevo.library import ModuleLibrary

    # The live handle sees entries admitted MID-RUN; the by-path cache is the flat-config fallback.
    library = ctx.library if ctx.library is not None else _cached_library(path)
    assert isinstance(library, ModuleLibrary)
    entries = library.query(entry_type=MODULE_ENTRY)
    if not entries:
        return genome
    entry = entries[rng.randrange(len(entries))]

    from ardevo.evolution.genome import genome_from_dict

    source = genome_from_dict(entry.payload)
    child = genome.clone()
    host_sources = [*child.input_ids, *child.bias_ids, *child.hidden_ids]
    host_outputs = child.output_ids
    host_bias = child.bias_ids[0] if child.bias_ids else None
    if not host_sources or not host_outputs:
        return genome

    id_map: dict[int, int] = {}
    for node in source.nodes.values():
        if node.kind is NodeKind.BIAS and host_bias is not None:
            id_map[node.id] = host_bias
            continue
        new_id = ctx.innovations.new_node_id()
        id_map[node.id] = new_id
        activation = node.activation if node.kind is NodeKind.HIDDEN else "identity"
        child.nodes[new_id] = NodeGene(new_id, NodeKind.HIDDEN, activation, None, node.aggregation)
    for conn in source.connections:
        in_id, out_id = id_map[conn.in_id], id_map[conn.out_id]
        child.connections.append(ConnectionGene(in_id, out_id, conn.weight, conn.enabled, ctx.innovations.innovation(in_id, out_id, conn.recurrent), conn.recurrent))
    for macro in source.macros:
        child.macros.append(
            MacroGene(
                ref=macro.ref,
                input_node_ids=tuple(id_map[node_id] for node_id in macro.input_node_ids),
                output_node_ids=tuple(id_map[node_id] for node_id in macro.output_node_ids),
                innovation=ctx.innovations.new_marker(),
                trainable=macro.trainable,
            )
        )

    # Sample DISTINCT host sources for the inlined module's input ports (without replacement) so the
    # ports receive different signals; cycle only when the module has more ports than the host has
    # sources. Wiring every port from one source would defeat the point of inlining found structure.
    ports = [id_map[node_id] for node_id in source.input_ids]
    distinct_sources = rng.sample(host_sources, min(len(ports), len(host_sources)))
    for offset, port in enumerate(ports):
        source_node = distinct_sources[offset % len(distinct_sources)]
        child.connections.append(ConnectionGene(source_node, port, 1.0, True, ctx.innovations.innovation(source_node, port)))
    for port in (id_map[node_id] for node_id in source.output_ids):
        for output in host_outputs:
            child.connections.append(ConnectionGene(port, output, rng.gauss(0.0, 0.3), True, ctx.innovations.innovation(port, output)))
    return child


@MUTATION.register("add_macro_node")
def add_macro_node(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.05, path: str = "library", max_outputs: int = 16) -> Genome:
    """Place a WHOLE library network as a single frozen unit (the LSTM-cell-in-a-perceptron idea).

    Unlike add_library_module (which inlines an unfrozen copy as ordinary evolvable structure), a
    macro keeps the found network's identity and function intact: the genome stores one MacroGene
    plus m HIDDEN identity stubs; the inner network resolves at decode time and never trains.
    Wiring is exact-k from randomly sampled host sources (no input glue in v1: glue belongs to
    compositions); each output stub reads out to every host OUTPUT so the macro contributes
    immediately. Acyclic by construction (fresh output ids, readouts feed only host outputs).
    """
    if rng.random() >= prob:
        return genome
    from ardevo.library import MODULE as MODULE_ENTRY
    from ardevo.library import ModuleLibrary

    library = ctx.library if ctx.library is not None else _cached_library(path)
    assert isinstance(library, ModuleLibrary)
    host_sources = [*genome.input_ids, *genome.bias_ids, *genome.hidden_ids]
    host_outputs = genome.output_ids
    if not host_sources or not host_outputs:
        return genome
    candidates = []
    for entry in library.query(entry_type=MODULE_ENTRY):
        k = sum(item["width"] for item in entry.io["inputs"])
        m = int(entry.io["output"]["width"])
        if 1 <= k <= len(host_sources) and 1 <= m <= max_outputs:
            candidates.append((entry, k, m))
    if not candidates:
        return genome
    entry, k, m = candidates[rng.randrange(len(candidates))]

    child = genome.clone()
    inputs = tuple(rng.sample(host_sources, k))
    outputs = []
    for _ in range(m):
        new_id = ctx.innovations.new_node_id()
        child.nodes[new_id] = NodeGene(new_id, NodeKind.HIDDEN, "identity")
        outputs.append(new_id)
    child.macros.append(MacroGene(ref=f"library:{entry.key}", input_node_ids=inputs, output_node_ids=tuple(outputs), innovation=ctx.innovations.new_marker()))
    for stub in outputs:
        for host_output in host_outputs:
            child.connections.append(ConnectionGene(stub, host_output, rng.gauss(0.0, 0.3), True, ctx.innovations.innovation(stub, host_output)))
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
    targets = [node_id for node_id in (*child.hidden_ids, *child.output_ids) if node_id not in child.macro_output_ids]
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
