"""Mutation operators: an independent stage, separate from crossover.

Each mutator is a single-purpose, registered function `(genome, ctx, *, rng, **params) -> Genome`.
`MutationPipeline` composes a config-ordered list of them, so individual operators are swapped
in or out via `[evolution.mutation].operators` with no code change. The shared `MutationContext`
hands out fresh node ids / innovation numbers and the activation palette.
"""

import heapq
import inspect
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable

from ardevo.evolution.genome import (
    ConnectionGene,
    ForwardReachability,
    Genome,
    InnovationTracker,
    MacroGene,
    NodeGene,
    NodeKind,
    coordinate_distance,
    set_connection,
    topological_order,
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


def _operator_base_prob(operator: Mutator, params: dict[str, Any]) -> float:
    """The rate an operator starts self-adapting from: its configured `prob`, else its own default."""
    if "prob" in params:
        return float(params["prob"])
    parameter = inspect.signature(operator).parameters.get("prob")
    if parameter is not None and parameter.default is not inspect.Parameter.empty:
        return float(parameter.default)
    return 0.1


@dataclass
class AdaptiveMutationPipeline:
    """Self-adaptive mutation (lever F): each genome carries its own per-operator rates as strategy genes.

    ES perturb-and-inherit (Rechenberg/Schwefel): before applying the operators to a child, perturb the
    rates it inherited from its parent with a log-normal step (`rate * exp(learning_rate * N(0, 1))`, a
    multiplicative random walk that stays positive), apply each operator at its own perturbed rate, then
    stamp the perturbed rates onto the child so a fitter lineage's schedule propagates and a bad one dies
    with its owner. The search rate thus adapts per problem instead of being hand-tuned per config.

    Off is the ordinary `MutationPipeline` (a different object entirely), so the fixed-rate path is
    byte-identical; this is constructed only under `[evolution.mutation] self_adaptive = true`. Rates seed
    from each operator's configured `prob` on a genome that has none yet, and clamp to [min_rate, max_rate].
    """

    operators: Sequence[tuple[str, Mutator, dict[str, Any]]]
    learning_rate: float = 0.1
    min_rate: float = 0.001
    max_rate: float = 1.0
    _base_rates: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._base_rates = {name: _operator_base_prob(operator, params) for name, operator, params in self.operators}

    def __call__(self, genome: Genome, ctx: MutationContext, *, rng: random.Random) -> Genome:
        inherited = genome.operator_rates
        perturbed: dict[str, float] = {}
        for name, _operator, _params in self.operators:
            base = inherited.get(name, self._base_rates[name])
            drifted = base * math.exp(self.learning_rate * rng.gauss(0.0, 1.0))
            perturbed[name] = min(self.max_rate, max(self.min_rate, drifted))
        child = genome
        for name, operator, params in self.operators:
            passthrough = {key: value for key, value in params.items() if key != "prob"}
            child = operator(child, ctx, rng=rng, prob=perturbed[name], **passthrough)
        if child is genome:  # no operator fired: never stamp rates onto a genome we do not own
            child = genome.clone()
        child.operator_rates = perturbed
        return child


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
    # Per-call precomputation instead of per-pair O(E) scans: the wide-input pair sweep was the
    # image-rung wedge (see ForwardReachability). Same iteration order, same accept decisions,
    # no rng draws touched, so children are bitwise-identical to the scanning form.
    existing = {(conn.in_id, conn.out_id) for conn in child.connections if not conn.recurrent}
    reach = ForwardReachability(child)
    for source in sources:
        for target in targets:
            if source == target or (source, target) in existing:
                continue
            if reach.creates_cycle(source, target):
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


@MUTATION.register("remove_connection")
def remove_connection(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.05) -> Genome:
    """Delete one connection gene outright, enabled or disabled (toggle_connection owns disable).
    True deletion is the structural shrink move: it cuts clone/payload cost AND the excess/disjoint
    gene counts speciation distance sees, so pruned lineages niche apart from their bloated kin.
    Innovation numbers are memoized per (in, out, recurrent), so a later re-add restores the same
    number and crossover alignment is unaffected; orphaning a downstream node is decode-safe (a
    zero-in-degree hidden node computes an all-zero column, exactly as toggle-disable produces today)."""
    if rng.random() >= prob or not genome.connections:
        return genome
    child = genome.clone()
    del child.connections[rng.randrange(len(child.connections))]
    return child


@MUTATION.register("remove_hidden_node")
def remove_hidden_node(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.05) -> Genome:
    """Delete one hidden node plus EVERY incident gene (enabled/disabled, forward/recurrent).
    The macro filters mirror the add-ops: output stubs carry the inner network's value, and macro
    input references are position-looked-up at decode, so deleting either corrupts the placement.
    The full incident-edge sweep is what keeps NEAT crossover's node-pull and the decode position
    maps consistent: no surviving gene may reference the deleted id."""
    macro_input_ids = {node_id for macro in genome.macros for node_id in macro.input_node_ids}
    candidates = [node_id for node_id in genome.hidden_ids if node_id not in genome.macro_output_ids and node_id not in macro_input_ids]
    if rng.random() >= prob or not candidates:
        return genome
    child = genome.clone()
    node_id = rng.choice(candidates)
    del child.nodes[node_id]
    child.connections = [conn for conn in child.connections if conn.in_id != node_id and conn.out_id != node_id]
    return child


@MUTATION.register("prune_and_regrow")
def prune_and_regrow(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.1, fraction: float = 0.05) -> Genome:
    """SET-style rewiring (Mocanu et al. 2018, Sparse Evolutionary Training): delete the smallest
    magnitude enabled FORWARD edges and regrow the same count at fresh random positions, so a
    sparse-seeded wide genome discovers which connections matter at constant density instead of
    only shrinking (the remove_* ops) or only growing (the add_* ops). Regrown edges are accepted
    only strictly forward in one fixed topological order, which makes any batch of additions
    jointly acyclic with O(1) checks per candidate: the per-edge reachability rebuild a naive batch
    add would pay is exactly the image-rung-wedge shape (see ForwardReachability). Innovation
    numbers come from the run tracker's memo, so a pruned-then-regrown edge restores its original
    number and crossover alignment survives the churn. Macro output stubs are never targeted."""
    if rng.random() >= prob:
        return genome
    prunable = [conn for conn in genome.connections if conn.enabled and not conn.recurrent]
    if not prunable:
        return genome
    child = genome.clone()
    count = min(len(prunable), max(1, round(len(prunable) * fraction)))
    doomed = {id(conn) for conn in heapq.nsmallest(count, prunable, key=lambda conn: abs(conn.weight))}
    child.connections = [conn for conn in child.connections if id(conn) not in doomed]

    order_position = {node_id: index for index, node_id in enumerate(topological_order(child))}
    existing = {(conn.in_id, conn.out_id) for conn in child.connections if not conn.recurrent}
    sources = [*child.input_ids, *child.bias_ids, *child.hidden_ids]
    targets = [node_id for node_id in (*child.hidden_ids, *child.output_ids) if node_id not in child.macro_output_ids]
    if not sources or not targets:
        return child
    added, attempts = 0, 0
    while added < count and attempts < 20 * count:
        attempts += 1
        source = rng.choice(sources)
        target = rng.choice(targets)
        if source == target or (source, target) in existing or order_position[source] >= order_position[target]:
            continue
        child.connections.append(ConnectionGene(source, target, rng.gauss(0.0, 1.0), True, ctx.innovations.innovation(source, target)))
        existing.add((source, target))
        added += 1
    return child


def _hint_weighted_choice(candidates: list[int], hints: dict[int, float] | None, rng: random.Random) -> int:
    """Sample a node id proportional to its growth-hint score, uniform when hints are absent.
    The epsilon floor keeps zero-scored nodes reachable (exploration never fully closes)."""
    if not hints:
        return rng.choice(candidates)
    weights = [hints.get(node_id, 0.0) + 1e-12 for node_id in candidates]
    return rng.choices(candidates, weights=weights, k=1)[0]


@MUTATION.register("add_hinted_connection")
def add_hinted_connection(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.1, attempts: int = 16) -> Genome:
    """NeST-guided add_connection: source sampled by activation mass, target by delta mass (the
    rank-1 marginals of the dormant-edge gradient |dL/dw_ij| = |a_i x delta_j| the train stage
    stashed as growth_hints), so new wiring lands where the loss says signal is missing instead of
    uniformly. Without hints (scoring off, crossover child, non-plain substrate) it degrades to
    uniform sampling: same legality rules as add_connection, gradient only biases WHERE."""
    if rng.random() >= prob:
        return genome
    sources = [*genome.input_ids, *genome.bias_ids, *genome.hidden_ids]
    targets = [node_id for node_id in (*genome.hidden_ids, *genome.output_ids) if node_id not in genome.macro_output_ids]
    if not sources or not targets:
        return genome
    hints = genome.growth_hints or {}
    child = genome.clone()
    existing = {(conn.in_id, conn.out_id) for conn in child.connections if not conn.recurrent}
    reach = ForwardReachability(child)
    for _ in range(attempts):
        source = _hint_weighted_choice(sources, hints.get("source"), rng)
        target = _hint_weighted_choice(targets, hints.get("target"), rng)
        if source == target or (source, target) in existing or reach.creates_cycle(source, target):
            continue
        child.connections.append(ConnectionGene(source, target, rng.gauss(0.0, 1.0), True, ctx.innovations.innovation(source, target)))
        return child
    return child


@MUTATION.register("add_hinted_node")
def add_hinted_node(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.1, fan_in: int = 4) -> Genome:
    """GradMax-lite: install a hidden node where the delta mass says capacity is missing, with a
    ZERO-weight fan-out edge so the child computes the identical function at birth (Net2Net/GradMax
    function preservation: selection never culls fresh structure before gradient training exploits
    it; the fan-in weights are positioned to receive gradient immediately). Fan-in sources sample
    by activation mass. Degrades to uniform sampling without hints."""
    if rng.random() >= prob:
        return genome
    hints = genome.growth_hints or {}
    macro_input_ids = {node_id for macro in genome.macros for node_id in macro.input_node_ids}
    targets = [node_id for node_id in (*genome.hidden_ids, *genome.output_ids) if node_id not in genome.macro_output_ids and node_id not in macro_input_ids]
    sources = [*genome.input_ids, *genome.bias_ids, *genome.hidden_ids]
    if not sources or not targets:
        return genome
    child = genome.clone()
    reach = ForwardReachability(child)
    target = _hint_weighted_choice(targets, hints.get("target"), rng)
    legal_sources = [source for source in sources if source != target and not reach.creates_cycle(source, target)]
    if not legal_sources:
        return genome
    new_id = ctx.innovations.new_node_id()
    child.nodes[new_id] = NodeGene(new_id, NodeKind.HIDDEN, ctx.default_activation)
    chosen: list[int] = []
    for _ in range(min(fan_in, len(legal_sources))):
        pick = _hint_weighted_choice([source for source in legal_sources if source not in chosen], hints.get("source"), rng)
        chosen.append(pick)
    for source in chosen:
        child.connections.append(ConnectionGene(source, new_id, rng.gauss(0.0, 1.0), True, ctx.innovations.innovation(source, new_id)))
    child.connections.append(ConnectionGene(new_id, target, 0.0, True, ctx.innovations.innovation(new_id, target)))
    return child


@MUTATION.register("split_node")
def split_node(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.05) -> Genome:
    """Net2WiderNet: duplicate a hidden node (incoming edges copied verbatim, outgoing weights
    halved on both copies), so the child computes the EXACT same function and gradient training
    decides how the twins specialize. Excluded: macro-tied nodes (decode position maps), nodes
    with recurrent out-edges (state semantics differ across copies), and nodes feeding a product
    target (an extra factor changes the product, breaking neutrality)."""
    if rng.random() >= prob:
        return genome
    macro_input_ids = {node_id for macro in genome.macros for node_id in macro.input_node_ids}
    product_nodes = {node.id for node in genome.nodes.values() if node.aggregation == "product"}

    def splittable(node_id: int) -> bool:
        if node_id in genome.macro_output_ids or node_id in macro_input_ids:
            return False
        for conn in genome.connections:
            if conn.recurrent and node_id in (conn.in_id, conn.out_id):
                return False  # a recurrent in-edge makes the twins compute different values; out-edges double state
            if conn.in_id == node_id and conn.out_id in product_nodes:
                return False
        return True

    candidates = [node_id for node_id in genome.hidden_ids if splittable(node_id)]
    if not candidates:
        return genome
    child = genome.clone()
    original_id = rng.choice(candidates)
    original = child.nodes[original_id]
    new_id = ctx.innovations.new_node_id()
    child.nodes[new_id] = replace(original, id=new_id)
    rebuilt: list[ConnectionGene] = []
    for conn in child.connections:
        if conn.out_id == original_id and not conn.recurrent:
            rebuilt.append(conn)
            rebuilt.append(ConnectionGene(conn.in_id, new_id, conn.weight, conn.enabled, ctx.innovations.innovation(conn.in_id, new_id, conn.recurrent)))
        elif conn.in_id == original_id and not conn.recurrent:
            rebuilt.append(replace(conn, weight=conn.weight / 2.0))
            rebuilt.append(ConnectionGene(new_id, conn.out_id, conn.weight / 2.0, conn.enabled, ctx.innovations.innovation(new_id, conn.out_id, conn.recurrent)))
        else:
            rebuilt.append(conn)
    child.connections = rebuilt
    return child


@MUTATION.register("add_relation_node")
def add_relation_node(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.05, fan_in: int = 2) -> Genome:
    """Relational-bottleneck primitive (Webb et al.): a PRODUCT-aggregation hidden node over a few
    bounded activations approximates a similarity detector, and downstream nodes that read it see
    relations between values instead of the values themselves: the inductive bias behind
    sample-efficient abstract-rule learning (RAVEN/PGM class). One gene plus wiring, built from the
    existing product machinery: evolution decides whether relations pay."""
    if rng.random() >= prob:
        return genome
    sources = [*genome.input_ids, *genome.hidden_ids]  # bias excluded: a constant factor only rescales
    outputs = genome.output_ids
    if len(sources) < 2 or not outputs:
        return genome
    child = genome.clone()
    new_id = ctx.innovations.new_node_id()
    child.nodes[new_id] = NodeGene(new_id, NodeKind.HIDDEN, "tanh", aggregation="product")
    for source in rng.sample(sources, min(fan_in, len(sources))):
        child.connections.append(ConnectionGene(source, new_id, rng.gauss(0.0, 1.0), True, ctx.innovations.innovation(source, new_id)))
    for output in outputs:
        child.connections.append(ConnectionGene(new_id, output, rng.gauss(0.0, 1.0), True, ctx.innovations.innovation(new_id, output)))
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
    from ardevo.substrate import _MAX_MACRO_DEPTH

    library = ctx.library if ctx.library is not None else _cached_library(path)
    assert isinstance(library, ModuleLibrary)
    host_sources = [*genome.input_ids, *genome.bias_ids, *genome.hidden_ids]
    host_outputs = genome.output_ids
    if not host_sources or not host_outputs:
        return genome
    candidates = []
    for entry in library.query(entry_type=MODULE_ENTRY):
        # k/m are the inner genome's actual INPUT/OUTPUT NODE counts (what the macro decode validates
        # against), NOT the io widths: a temporal module's io width is the FLATTENED width (e.g. 24)
        # while its genome has only per-step input nodes (e.g. 3), and trusting io would wire a macro
        # the decoder rejects with a shape mismatch.
        nodes = entry.payload.get("nodes", [])
        k = sum(1 for node in nodes if node.get("kind") == "input")
        m = sum(1 for node in nodes if node.get("kind") == "output")
        if not (1 <= k <= len(host_sources) and 1 <= m <= max_outputs):
            continue
        # Embedding this entry nests its whole macro chain one level deeper; past the decode cap
        # the child is a dead phenotype (the wall ledger's seed-then-embed cycles get there fast).
        if library.macro_subtree_depth(entry.key) > _MAX_MACRO_DEPTH - 1:
            continue
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


@MUTATION.register("tweak_refine_steps")
def tweak_refine_steps(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.05, min_steps: int = 1, max_steps: int = 8) -> Genome:
    """Nudge the genome's refinement depth +/-1 within [min_steps, max_steps] (the TRM lever: more
    passes = more effective depth without more parameters). Inert until the genome also carries
    recurrent edges to thread state across passes, so it pairs with add_recurrent_connection."""
    if rng.random() >= prob:
        return genome
    child = genome.clone()
    child.refine_steps = max(min_steps, min(max_steps, child.refine_steps + rng.choice((-1, 1))))
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
    # This full pair sweep (every input x every target on a stamped grid) was THE image-rung wedge:
    # per-pair has_connection/would_create_cycle scans measured 4.9s PER CALL at width 3072 on the
    # main thread. Precompute once; the sweep itself is preserved verbatim (same candidate order,
    # same weights, same rng draws), so evolution is bitwise-identical.
    existing = {(conn.in_id, conn.out_id) for conn in child.connections if not conn.recurrent}
    reach = ForwardReachability(child)
    for source in sources:
        for target in targets:
            if source == target or (source, target) in existing or reach.creates_cycle(source, target):
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
def add_shared_motif(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.05, copies: int = 2, tied: bool = False) -> Genome:
    """Replicate an existing local detector's motif at other coordinate locations.

    A structural convolution prior: take a hidden node's local fan-in size and output readouts, and
    grow `copies` siblings centered on other seeds, each reading its own local neighborhood. With
    `tied = false` (the default, byte-identical to the historical operator) only the connectivity
    pattern repeats and every copy trains its own weights. With `tied = true` the template's edges
    are tagged with tie groups (fan-in matched to copies by RELATIVE coordinate offset from the
    detector's center, the convolution correspondence) and each copy's edges join those groups, so
    all copies share ONE trainable weight per template edge: hard weight sharing, the mechanic that
    lets translation-invariant feature detectors become affordable at image widths."""
    if rng.random() >= prob:
        return genome
    hidden = [node_id for node_id in genome.hidden_ids if genome.nodes[node_id].coordinate is not None]
    if not hidden:
        return genome
    child = genome.clone()
    template = rng.choice(hidden)
    incoming = [conn for conn in child.enabled_connections() if conn.out_id == template and child.nodes[conn.in_id].coordinate is not None]
    output_edges = [conn for conn in child.enabled_connections() if conn.in_id == template and child.nodes[conn.out_id].kind is NodeKind.OUTPUT]
    if not incoming or not output_edges:
        return child

    def offset_key(node_id: int, center: tuple[float, ...] | None) -> tuple[float, ...]:
        coordinate = child.nodes[node_id].coordinate
        if coordinate is None or center is None or len(coordinate) != len(center):
            return (math.inf,)
        return tuple(value - anchor for value, anchor in zip(coordinate, center))

    template_center = child.nodes[template].coordinate
    incoming = sorted(incoming, key=lambda conn: offset_key(conn.in_id, template_center))
    if tied:
        # Tag the template's own genes first (idempotent: existing groups are reused), so the
        # template and every copy share parameters. Group ids are run-unique tracker markers.
        retagged: dict[int, int] = {}
        for index, conn in enumerate(incoming):
            group = conn.tie_group if conn.tie_group is not None else ctx.innovations.new_marker()
            retagged[id(conn)] = group
        out_groups: dict[int, int] = {}
        for conn in output_edges:
            out_groups[id(conn)] = conn.tie_group if conn.tie_group is not None else ctx.innovations.new_marker()
        child.connections = [
            replace(conn, tie_group=retagged[id(conn)]) if id(conn) in retagged else replace(conn, tie_group=out_groups[id(conn)]) if id(conn) in out_groups else conn
            for conn in child.connections
        ]
        incoming = sorted(
            [conn for conn in child.enabled_connections() if conn.out_id == template and child.nodes[conn.in_id].coordinate is not None],
            key=lambda conn: offset_key(conn.in_id, template_center),
        )
        output_edges = [conn for conn in child.enabled_connections() if conn.in_id == template and child.nodes[conn.out_id].kind is NodeKind.OUTPUT]

    field_size = len(incoming)
    sources = [node_id for node_id in (*child.input_ids, *child.hidden_ids) if child.nodes[node_id].coordinate is not None]
    for seed in rng.sample(sources, min(copies, len(sources))):
        field = _nearest(child, child.nodes[seed].coordinate, sources, field_size)
        coords = [coord for node_id in field if (coord := child.nodes[node_id].coordinate) is not None]
        if not coords:
            continue
        new_id = ctx.innovations.new_node_id()
        center = _centroid(coords)
        child.nodes[new_id] = NodeGene(new_id, NodeKind.HIDDEN, child.nodes[template].activation, center)
        # One reachability snapshot per copy instead of a per-source adjacency rebuild. Mid-loop
        # appends only add edges INTO new_id, which cannot extend any walk FROM new_id, so the
        # snapshot answers exactly what the per-call rebuild answered (structurally always False
        # today: new_id is a sink until its output readouts land below).
        reach = ForwardReachability(child)
        field_sorted = sorted(field, key=lambda node_id: offset_key(node_id, center)) if tied else field
        for index, source in enumerate(field_sorted):
            if reach.creates_cycle(source, new_id):
                continue
            if tied and index < len(incoming):
                matched = incoming[index]  # offset-rank correspondence: i-th offset shares the i-th template weight
                child.connections.append(ConnectionGene(source, new_id, matched.weight, True, ctx.innovations.innovation(source, new_id), tie_group=matched.tie_group))
            else:
                child.connections.append(ConnectionGene(source, new_id, rng.gauss(0.0, 1.0), True, ctx.innovations.innovation(source, new_id)))
        for edge_index, output_edge in enumerate(output_edges):
            if tied:
                child.connections.append(
                    ConnectionGene(new_id, output_edge.out_id, output_edge.weight, True, ctx.innovations.innovation(new_id, output_edge.out_id), tie_group=output_edge.tie_group)
                )
            else:
                child.connections.append(ConnectionGene(new_id, output_edge.out_id, rng.gauss(0.0, 1.0), True, ctx.innovations.innovation(new_id, output_edge.out_id)))
    return child


@MUTATION.register("untie_motif_weights")
def untie_motif_weights(genome: Genome, ctx: MutationContext, *, rng: random.Random, prob: float = 0.05) -> Genome:
    """Dissolve one tie group: every member keeps the shared value as its own independent weight,
    so the child computes the identical function at birth and gradient training may then specialize
    the copies (the inverse of add_shared_motif(tied=true); evolution owns the share/unshare dial)."""
    if rng.random() >= prob:
        return genome
    groups = sorted({conn.tie_group for conn in genome.connections if conn.tie_group is not None})
    if not groups:
        return genome
    child = genome.clone()
    doomed = rng.choice(groups)
    child.connections = [replace(conn, tie_group=None) if conn.tie_group == doomed else conn for conn in child.connections]
    return child
