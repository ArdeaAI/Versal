"""CompositionGenome: ONE recursive representation for every level above the mini-model.

A composition is a small DAG whose nodes are typed ports: INPUT nodes read slices of the encoded
task input (`ref` = bank signature, or the synthetic `__bias__` source), MODULE nodes reference a
reusable lower-level entry (`ref` = "live:<species_id>" for the co-evolving module population or
"library:<entry_key>" for an admitted solution), and one OUTPUT node is the task head. Edges carry
GLUE: a dense trainable linear map between the source's out-ports and the target's in-ports.

Recursion comes from the library, not from new types: a library entry can itself be a saved
composition, and assembly inlines it transitively (depth/cycle guarded). Level 2 is a composition of
mini-models; level 3 is a composition that references level-2 entries. Repeated refs inside one
composition resolve to the SAME inner module instance, so repetition means literal weight sharing
(the structural prior behind convolution and deep repeats).
"""

import math
import random
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable

import torch
from torch import nn

from ardevo.evolution.genome import Genome, InnovationTracker, genome_from_dict, make_acyclic
from ardevo.evolution.registry import Registry
from ardevo.library import COMPOSITION, MODULE, ModuleLibrary
from ardevo.substrate import SubstrateModule, decode

BIAS_REF = "__bias__"


class CompNodeKind(Enum):
    INPUT = "input"
    MODULE = "module"
    OUTPUT = "output"


@dataclass(frozen=True, slots=True)
class CompNodeGene:
    id: int
    kind: CompNodeKind
    ref: str
    in_width: int  # 0 for INPUT nodes (they are sources)
    out_width: int  # 0 for the OUTPUT node (it is the readout)
    aggregation: str = "sum"  # how multiple inbound glue contributions combine ("sum" | "product")
    trainable: bool = True  # False pins the resolved inner module's weights even for live refs


@dataclass(frozen=True, slots=True)
class CompEdgeGene:
    in_id: int
    out_id: int
    enabled: bool
    innovation: int
    glue: tuple[float, ...]  # row-major [source.out_width x target.in_width] linear map


@dataclass
class CompositionGenome:
    """Duck-types the parts of `Genome` the fitness components read (complexity, hidden_ids)."""

    nodes: dict[int, CompNodeGene] = field(default_factory=dict)
    edges: list[CompEdgeGene] = field(default_factory=list)

    def clone(self) -> "CompositionGenome":
        return CompositionGenome(nodes=dict(self.nodes), edges=list(self.edges))

    def enabled_edges(self) -> list[CompEdgeGene]:
        return [edge for edge in self.edges if edge.enabled]

    def ids_of(self, kind: CompNodeKind) -> list[int]:
        return sorted(node.id for node in self.nodes.values() if node.kind is kind)

    @property
    def input_ids(self) -> list[int]:
        return self.ids_of(CompNodeKind.INPUT)

    @property
    def module_ids(self) -> list[int]:
        return self.ids_of(CompNodeKind.MODULE)

    @property
    def output_ids(self) -> list[int]:
        return self.ids_of(CompNodeKind.OUTPUT)

    @property
    def hidden_ids(self) -> list[int]:
        # Module nodes play the role hidden nodes play in a flat genome, so hidden_penalty just works.
        return self.module_ids

    def enabled_connections(self) -> list[CompEdgeGene]:
        return self.enabled_edges()

    def complexity(self) -> int:
        return len(self.enabled_edges()) + len(self.module_ids)

    def has_edge(self, in_id: int, out_id: int) -> bool:
        return any(edge.in_id == in_id and edge.out_id == out_id for edge in self.edges)

    def refs(self) -> list[str]:
        return [node.ref for node in self.nodes.values() if node.kind is CompNodeKind.MODULE]


def comp_would_create_cycle(comp: CompositionGenome, in_id: int, out_id: int) -> bool:
    if in_id == out_id:
        return True
    adjacency: dict[int, list[int]] = {}
    for edge in comp.enabled_edges():
        adjacency.setdefault(edge.in_id, []).append(edge.out_id)
    queue = deque([out_id])
    seen = {out_id}
    while queue:
        current = queue.popleft()
        if current == in_id:
            return True
        for nxt in adjacency.get(current, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def comp_topological_order(comp: CompositionGenome) -> list[int]:
    incoming: dict[int, int] = {node_id: 0 for node_id in comp.nodes}
    adjacency: dict[int, list[int]] = {}
    for edge in comp.enabled_edges():
        adjacency.setdefault(edge.in_id, []).append(edge.out_id)
        incoming[edge.out_id] += 1
    ready = deque(sorted(node_id for node_id, count in incoming.items() if count == 0))
    order: list[int] = []
    while ready:
        current = ready.popleft()
        order.append(current)
        for nxt in adjacency.get(current, []):
            incoming[nxt] -= 1
            if incoming[nxt] == 0:
                ready.append(nxt)
    if len(order) != len(comp.nodes):
        raise ValueError("composition graph contains a cycle")
    return order


def comp_to_dict(comp: CompositionGenome) -> dict[str, Any]:
    return {
        "nodes": [
            {"id": n.id, "kind": n.kind.value, "ref": n.ref, "in_width": n.in_width, "out_width": n.out_width, "aggregation": n.aggregation, "trainable": n.trainable}
            for n in comp.nodes.values()
        ],
        "edges": [{"in": e.in_id, "out": e.out_id, "enabled": e.enabled, "innovation": e.innovation, "glue": list(e.glue)} for e in comp.edges],
    }


def comp_from_dict(data: dict[str, Any]) -> CompositionGenome:
    nodes = {
        int(n["id"]): CompNodeGene(
            int(n["id"]), CompNodeKind(n["kind"]), n["ref"], int(n["in_width"]), int(n["out_width"]), n.get("aggregation", "sum"), bool(n.get("trainable", True))
        )
        for n in data["nodes"]
    }
    edges = [CompEdgeGene(int(e["in"]), int(e["out"]), bool(e["enabled"]), int(e["innovation"]), tuple(float(v) for v in e["glue"])) for e in data["edges"]]
    return CompositionGenome(nodes=nodes, edges=edges)


# --- assembly --------------------------------------------------------------------------------------


class CompositionAssemblyError(Exception):
    """A composition cannot be built (ref vanished, glue shape drifted, recursion too deep)."""


@dataclass
class AssemblyContext:
    """Everything `assemble` needs to resolve refs. Create a FRESH context per decoded candidate so
    sibling compositions never share trainable inner copies; the cache inside one context is what
    gives repeated refs literal weight sharing WITHIN a composition."""

    bank_columns: dict[str, list[int]]
    live_resolver: Callable[[str], Genome] | None = None
    library: ModuleLibrary | None = None
    max_inline_depth: int = 4
    instance_cache: dict[str, SubstrateModule] = field(default_factory=dict)
    expansion_stack: list[str] = field(default_factory=list)


def _decode_ref_module(genome: Genome, n_inputs: int, n_outputs: int) -> SubstrateModule:
    try:
        return decode(genome, n_inputs, n_outputs)
    except ValueError as error:
        try:
            return decode(make_acyclic(genome), n_inputs, n_outputs)
        except ValueError as repaired_error:
            raise CompositionAssemblyError(str(repaired_error)) from error


def _resolve_module(node: CompNodeGene, ctx: AssemblyContext) -> SubstrateModule:
    if node.ref in ctx.instance_cache:
        return ctx.instance_cache[node.ref]
    if node.ref.startswith("live:"):
        if ctx.live_resolver is None:
            raise CompositionAssemblyError(f"no live resolver for ref {node.ref!r}")
        genome = ctx.live_resolver(node.ref.removeprefix("live:"))
        inner = _decode_ref_module(genome, node.in_width, node.out_width)
        if not node.trainable:
            for parameter in inner.parameters():
                parameter.requires_grad_(False)
    elif node.ref.startswith("library:"):
        if ctx.library is None:
            raise CompositionAssemblyError(f"no library attached for ref {node.ref!r}")
        key = node.ref.removeprefix("library:")
        if key in ctx.expansion_stack:
            raise CompositionAssemblyError(f"library entry {key!r} references itself (cycle)")
        if len(ctx.expansion_stack) >= ctx.max_inline_depth:
            raise CompositionAssemblyError(f"composition nesting exceeds max_inline_depth={ctx.max_inline_depth}")
        try:
            entry = ctx.library.load(key)
        except KeyError as error:
            raise CompositionAssemblyError(str(error)) from error
        if entry.entry_type == MODULE:
            inner = _decode_ref_module(genome_from_dict(entry.payload), node.in_width, node.out_width)
        elif entry.entry_type == COMPOSITION:
            inner_comp = comp_from_dict(entry.payload)
            ctx.expansion_stack.append(key)
            try:
                inner = ComposedNet(inner_comp, _nested_context(inner_comp, ctx), n_inputs=node.in_width)
            finally:
                ctx.expansion_stack.pop()
        else:
            raise CompositionAssemblyError(f"unknown entry type {entry.entry_type!r} for {key!r}")
        if entry.weights_frozen or not node.trainable:
            for parameter in inner.parameters():
                parameter.requires_grad_(False)
    else:
        raise CompositionAssemblyError(f"unresolvable module ref {node.ref!r}")
    ctx.instance_cache[node.ref] = inner
    return inner


def _nested_context(inner_comp: CompositionGenome, outer: AssemblyContext) -> AssemblyContext:
    """A nested composition reads its aggregated input vector positionally: consecutive column
    ranges per INPUT node in id order, matching how its in_width was computed at reference time."""
    columns: dict[str, list[int]] = {}
    cursor = 0
    for node_id in inner_comp.input_ids:
        node = inner_comp.nodes[node_id]
        if node.ref == BIAS_REF:
            continue
        columns[node.ref] = list(range(cursor, cursor + node.out_width))
        cursor += node.out_width
    return AssemblyContext(
        bank_columns=columns,
        live_resolver=outer.live_resolver,
        library=outer.library,
        max_inline_depth=outer.max_inline_depth,
        instance_cache={},  # nested scope: sharing is per composition, not across nesting levels
        expansion_stack=outer.expansion_stack,  # shared: the cycle/depth guard spans the whole tree
    )


def input_width_of(comp: CompositionGenome) -> int:
    """The flat input width a composition consumes (sum of non-bias INPUT widths, id order)."""
    return sum(comp.nodes[node_id].out_width for node_id in comp.input_ids if comp.nodes[node_id].ref != BIAS_REF)


class ComposedNet(SubstrateModule):
    """Executable composition: inner modules wired by trainable glue matrices.

    Glue parameters are always trainable; inner modules train only when live and `trainable`
    (library entries are frozen). `export_weights` is intentionally NOT implemented: composition
    weights are written back structurally via `writeback_composition` + per-ref module export.
    """

    def __init__(self, comp: CompositionGenome, ctx: AssemblyContext, n_inputs: int) -> None:
        super().__init__()
        if len(comp.output_ids) != 1:
            raise CompositionAssemblyError(f"v1 compositions need exactly one OUTPUT node, got {len(comp.output_ids)}")
        self.comp = comp
        self._order = comp_topological_order(comp)
        self._n_inputs = n_inputs
        self._output_id = comp.output_ids[0]

        self._columns: dict[int, torch.Tensor] = {}
        for node_id in comp.input_ids:
            node = comp.nodes[node_id]
            if node.ref == BIAS_REF:
                continue
            columns = ctx.bank_columns.get(node.ref)
            if columns is None or len(columns) != node.out_width:
                raise CompositionAssemblyError(f"input ref {node.ref!r} has no {node.out_width}-wide column mapping")
            self._columns[node_id] = torch.tensor(columns, dtype=torch.long)

        self.inner_modules: dict[str, SubstrateModule] = {}
        self._inner = nn.ModuleDict()  # registration only, so .parameters() sees inner weights
        for node_id in comp.module_ids:
            node = comp.nodes[node_id]
            if node.ref not in self.inner_modules:
                inner = _resolve_module(node, ctx)
                self.inner_modules[node.ref] = inner
                self._inner[str(len(self._inner))] = inner

        self.glue = nn.ParameterDict()
        self._incoming: dict[int, list[tuple[int, str]]] = {}
        for edge in comp.enabled_edges():
            source, target = comp.nodes[edge.in_id], comp.nodes[edge.out_id]
            expected = source.out_width * target.in_width
            if len(edge.glue) != expected:
                raise CompositionAssemblyError(f"glue on {edge.in_id}->{edge.out_id} has {len(edge.glue)} values, expected {expected}")
            key = f"{edge.in_id}->{edge.out_id}"
            self.glue[key] = nn.Parameter(torch.tensor(edge.glue, dtype=torch.float32).reshape(source.out_width, target.in_width))
            self._incoming.setdefault(edge.out_id, []).append((edge.in_id, key))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values: dict[int, torch.Tensor] = {}
        for node_id in self._order:
            node = self.comp.nodes[node_id]
            if node.kind is CompNodeKind.INPUT:
                if node.ref == BIAS_REF:
                    values[node_id] = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
                else:
                    values[node_id] = x.index_select(1, self._columns[node_id])
                continue
            contributions = [values[in_id] @ self.glue[key] for in_id, key in self._incoming.get(node_id, [])]
            if not contributions:
                combined = torch.zeros(x.shape[0], node.in_width, dtype=x.dtype, device=x.device)
            elif node.aggregation == "product" and len(contributions) > 1:
                combined = contributions[0]
                for contribution in contributions[1:]:
                    combined = combined * contribution
            else:
                combined = sum(contributions[1:], start=contributions[0])
            if node.kind is CompNodeKind.MODULE:
                values[node_id] = self.inner_modules[node.ref](combined)
            else:  # OUTPUT: a linear readout of its aggregated glue
                values[node_id] = combined
        return values[self._output_id]

    @property
    def has_edges(self) -> bool:
        return bool(self.comp.enabled_edges())


def assemble(comp: CompositionGenome, ctx: AssemblyContext, n_inputs: int) -> ComposedNet:
    return ComposedNet(comp, ctx, n_inputs)


def writeback_composition(comp: CompositionGenome, net: ComposedNet) -> CompositionGenome:
    """Copy the net's trained glue back onto the matching enabled edge genes (Lamarckian glue)."""
    child = comp.clone()
    updated: list[CompEdgeGene] = []
    for edge in child.edges:
        key = f"{edge.in_id}->{edge.out_id}"
        if edge.enabled and key in net.glue:
            updated.append(replace(edge, glue=tuple(float(v) for v in net.glue[key].detach().reshape(-1).tolist())))
        else:
            updated.append(edge)
    child.edges = updated
    return child


# --- seeding and mutation ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefSpec:
    """One referenceable building block the mutators can draw from."""

    ref: str
    in_width: int
    out_width: int


@dataclass
class CompMutationContext:
    innovations: InnovationTracker
    ref_catalog: list[RefSpec]


CompMutator = Callable[..., CompositionGenome]
COMP_MUTATION: Registry[CompMutator] = Registry("comp_mutation")
COMP_CROSSOVER: Registry[Callable[..., CompositionGenome]] = Registry("comp_crossover")


def _glue_init(in_width: int, out_width: int, rng: random.Random, scale: float | None = None) -> tuple[float, ...]:
    sigma = scale if scale is not None else 1.0 / math.sqrt(max(in_width, 1))
    return tuple(rng.gauss(0.0, sigma) for _ in range(in_width * out_width))


def minimal_composition(
    input_specs: list[tuple[str, int]],
    output_ref: str,
    output_width: int,
    tracker: InnovationTracker,
    rng: random.Random,
    *,
    glue_scale: float | None = None,
) -> CompositionGenome:
    """The seed skeleton: INPUT banks (+ a bias source) glued straight to the OUTPUT head."""
    comp = CompositionGenome()
    sources: list[int] = []
    for ref, width in [*input_specs, (BIAS_REF, 1)]:
        node_id = tracker.new_node_id()
        comp.nodes[node_id] = CompNodeGene(node_id, CompNodeKind.INPUT, ref, 0, width)
        sources.append(node_id)
    output_id = tracker.new_node_id()
    comp.nodes[output_id] = CompNodeGene(output_id, CompNodeKind.OUTPUT, output_ref, output_width, 0)
    for source in sources:
        width = comp.nodes[source].out_width
        comp.edges.append(CompEdgeGene(source, output_id, True, tracker.innovation(source, output_id), _glue_init(width, output_width, rng, glue_scale)))
    return comp


def add_module_between(
    comp: CompositionGenome,
    spec: RefSpec,
    source_id: int,
    target_id: int,
    tracker: InnovationTracker,
    rng: random.Random,
    *,
    glue_scale: float | None = None,
) -> int:
    """Insert a MODULE node reading `source_id` and feeding `target_id`; returns the new node id."""
    node_id = tracker.new_node_id()
    comp.nodes[node_id] = CompNodeGene(node_id, CompNodeKind.MODULE, spec.ref, spec.in_width, spec.out_width)
    comp.edges.append(CompEdgeGene(source_id, node_id, True, tracker.innovation(source_id, node_id), _glue_init(comp.nodes[source_id].out_width, spec.in_width, rng, glue_scale)))
    comp.edges.append(CompEdgeGene(node_id, target_id, True, tracker.innovation(node_id, target_id), _glue_init(spec.out_width, comp.nodes[target_id].in_width, rng, glue_scale)))
    return node_id


@COMP_MUTATION.register("add_module_node")
def add_module_node(comp: CompositionGenome, ctx: CompMutationContext, *, rng: random.Random, prob: float = 0.2) -> CompositionGenome:
    """Grow the composition by one building block, wired source -> module -> readout-capable node."""
    if rng.random() >= prob or not ctx.ref_catalog:
        return comp
    sources = [node_id for node_id, node in comp.nodes.items() if node.out_width > 0]
    targets = [node_id for node_id, node in comp.nodes.items() if node.in_width > 0]
    if not sources or not targets:
        return comp
    child = comp.clone()
    spec = ctx.ref_catalog[rng.randrange(len(ctx.ref_catalog))]
    source_id = rng.choice(sources)
    # Inserting source -> module -> target closes a cycle iff target can already reach source.
    candidates = [t for t in targets if t != source_id and not comp_would_create_cycle(child, source_id, t)]
    if not candidates:
        return comp
    add_module_between(child, spec, source_id, rng.choice(candidates), ctx.innovations, rng)
    return child


@COMP_MUTATION.register("switch_ref")
def switch_ref(comp: CompositionGenome, ctx: CompMutationContext, *, rng: random.Random, prob: float = 0.1) -> CompositionGenome:
    """Repoint one MODULE node at a different SAME-SHAPE building block; trained glue is preserved
    because the port widths are unchanged."""
    if rng.random() >= prob or not comp.module_ids:
        return comp
    child = comp.clone()
    node_id = rng.choice(child.module_ids)
    node = child.nodes[node_id]
    alternatives = [spec for spec in ctx.ref_catalog if spec.in_width == node.in_width and spec.out_width == node.out_width and spec.ref != node.ref]
    if not alternatives:
        return comp
    child.nodes[node_id] = replace(node, ref=alternatives[rng.randrange(len(alternatives))].ref)
    return child


@COMP_MUTATION.register("add_comp_edge")
def add_comp_edge(comp: CompositionGenome, ctx: CompMutationContext, *, rng: random.Random, prob: float = 0.15) -> CompositionGenome:
    if rng.random() >= prob:
        return comp
    child = comp.clone()
    sources = [node_id for node_id, node in child.nodes.items() if node.out_width > 0]
    targets = [node_id for node_id, node in child.nodes.items() if node.in_width > 0]
    rng.shuffle(sources)
    rng.shuffle(targets)
    for source in sources:
        for target in targets:
            if source == target or child.has_edge(source, target) or comp_would_create_cycle(child, source, target):
                continue
            glue = _glue_init(child.nodes[source].out_width, child.nodes[target].in_width, rng)
            child.edges.append(CompEdgeGene(source, target, True, ctx.innovations.innovation(source, target), glue))
            return child
    return comp


@COMP_MUTATION.register("toggle_comp_edge")
def toggle_comp_edge(comp: CompositionGenome, ctx: CompMutationContext, *, rng: random.Random, prob: float = 0.05) -> CompositionGenome:
    if rng.random() >= prob or not comp.edges:
        return comp
    child = comp.clone()
    index = rng.randrange(len(child.edges))
    edge = child.edges[index]
    if edge.enabled:
        child.edges[index] = replace(edge, enabled=False)
    elif not comp_would_create_cycle(child, edge.in_id, edge.out_id):
        child.edges[index] = replace(edge, enabled=True)
    return child


@COMP_MUTATION.register("perturb_glue")
def perturb_glue(comp: CompositionGenome, ctx: CompMutationContext, *, rng: random.Random, prob: float = 0.8, sigma: float = 0.3) -> CompositionGenome:
    child = comp.clone()
    child.edges = [replace(edge, glue=tuple(value + rng.gauss(0.0, sigma) for value in edge.glue)) if rng.random() < prob else edge for edge in child.edges]
    return child


@dataclass
class CompMutationPipeline:
    operators: list[CompMutator]

    def __call__(self, comp: CompositionGenome, ctx: CompMutationContext, *, rng: random.Random) -> CompositionGenome:
        for operator in self.operators:
            comp = operator(comp, ctx, rng=rng)
        return comp


@COMP_CROSSOVER.register("none")
def comp_asexual(parent_a: CompositionGenome, parent_b: CompositionGenome, *, rng: random.Random) -> CompositionGenome:
    return parent_a.clone()


@COMP_CROSSOVER.register("comp_neat")
def comp_neat(parent_a: CompositionGenome, parent_b: CompositionGenome, *, rng: random.Random) -> CompositionGenome:
    """Innovation-aligned crossover: matching edges inherit whole glue vectors from either parent
    (when port shapes agree); disjoint/excess structure comes from the fitter `parent_a`."""
    by_innovation_b = {edge.innovation: edge for edge in parent_b.edges}
    child_edges: list[CompEdgeGene] = []
    for edge_a in parent_a.edges:
        edge_b = by_innovation_b.get(edge_a.innovation)
        if edge_b is None or len(edge_b.glue) != len(edge_a.glue) or rng.random() < 0.5:
            child_edges.append(edge_a)
        else:
            child_edges.append(edge_b)
    nodes: dict[int, CompNodeGene] = dict(parent_a.nodes)
    for edge in child_edges:
        for node_id in (edge.in_id, edge.out_id):
            if node_id not in nodes:
                nodes[node_id] = parent_b.nodes[node_id]
    return CompositionGenome(nodes=nodes, edges=child_edges)
