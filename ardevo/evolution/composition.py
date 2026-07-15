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

import base64
import math
import random
from array import array
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, TypeAlias

import torch
from torch import nn

from ardevo.evolution.genome import Genome, InnovationTracker, genome_from_dict, make_acyclic
from ardevo.evolution.registry import Registry
from ardevo.library import COMPOSITION, MODULE, ModuleLibrary, macro_resolver
from ardevo.reference_depth import DEFAULT_MAX_INLINE_DEPTH
from ardevo.substrate import SubstrateModule, decode_module

BIAS_REF = "__bias__"
GlueValues: TypeAlias = tuple[float, ...] | array


@dataclass(frozen=True, slots=True)
class IndexRun:
    """One compact contiguous gather/scatter run on a fixed composition edge."""

    source_start: int
    target_start: int
    length: int


@dataclass(frozen=True, slots=True)
class PortMap:
    """Immutable axis-derived wiring, stored as runs rather than a zero-heavy matrix."""

    runs: tuple[IndexRun, ...]

    @property
    def selected_count(self) -> int:
        return sum(run.length for run in self.runs)


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
    # Dense (glue_rank = 0): row-major [out_width x in_width] linear map. Factored (glue_rank = r):
    # U then V concatenated row-major (out_width*r + r*in_width floats); the effective map is U @ V,
    # never materialized. Factoring is the scale guard for wide ports (a 784x784 dense edge is 614k
    # floats; rank 8 is 12.5k).
    glue: GlueValues
    glue_rank: int = 0
    # A mapped edge has empty glue and never becomes a trainable Parameter.
    port_map: PortMap | None = None


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
    def edge_to_dict(edge: CompEdgeGene) -> dict[str, Any]:
        encoded: dict[str, Any] = {
            "in": edge.in_id,
            "out": edge.out_id,
            "enabled": edge.enabled,
            "innovation": edge.innovation,
            "glue_rank": edge.glue_rank,
        }
        if edge.port_map is not None:
            encoded["port_map"] = [{"source_start": run.source_start, "target_start": run.target_start, "length": run.length} for run in edge.port_map.runs]
        if isinstance(edge.glue, array):
            encoded["glue_f32_b64"] = base64.b64encode(memoryview(edge.glue).cast("B")).decode("ascii")
            encoded["glue_count"] = len(edge.glue)
        else:
            encoded["glue"] = list(edge.glue)
        return encoded

    return {
        "nodes": [
            {"id": n.id, "kind": n.kind.value, "ref": n.ref, "in_width": n.in_width, "out_width": n.out_width, "aggregation": n.aggregation, "trainable": n.trainable}
            for n in comp.nodes.values()
        ],
        "edges": [edge_to_dict(edge) for edge in comp.edges],
    }


def comp_from_dict(data: dict[str, Any]) -> CompositionGenome:
    def edge_glue(value: dict[str, Any]) -> GlueValues:
        payload = value.get("glue_f32_b64")
        if payload is None:
            return tuple(float(item) for item in value.get("glue", []))
        decoded = array("f")
        decoded.frombytes(base64.b64decode(str(payload), validate=True))
        expected = int(value.get("glue_count", len(decoded)))
        if len(decoded) != expected:
            raise ValueError(f"compact glue payload has {len(decoded)} values, expected {expected}")
        return decoded

    nodes = {
        int(n["id"]): CompNodeGene(
            int(n["id"]), CompNodeKind(n["kind"]), n["ref"], int(n["in_width"]), int(n["out_width"]), n.get("aggregation", "sum"), bool(n.get("trainable", True))
        )
        for n in data["nodes"]
    }
    edges = []
    for e in data["edges"]:
        raw_map = e.get("port_map")
        port_map = None
        if raw_map is not None:
            port_map = PortMap(tuple(IndexRun(int(run["source_start"]), int(run["target_start"]), int(run["length"])) for run in raw_map))
        edges.append(CompEdgeGene(int(e["in"]), int(e["out"]), bool(e["enabled"]), int(e["innovation"]), edge_glue(e), int(e.get("glue_rank", 0)), port_map))
    return CompositionGenome(nodes=nodes, edges=edges)


# --- assembly --------------------------------------------------------------------------------------


class CompositionAssemblyError(Exception):
    """A composition cannot be built (ref vanished, glue shape drifted, recursion too deep)."""


@dataclass
class AssemblyContext:
    """Everything `assemble` needs to resolve refs. Create a FRESH context per decoded candidate so
    sibling compositions never share trainable inner copies; the cache inside one context is what
    gives repeated refs literal weight sharing WITHIN a composition."""

    bank_columns: dict[str, Sequence[int]]
    live_resolver: Callable[[str], Genome] | None = None
    library: ModuleLibrary | None = None
    max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH
    # Root payloads start at zero. Every followed library: ref increments this counter, including
    # transitions between composition and module payloads; live: refs do not consume a persistent
    # library boundary. `expansion_stack` is separate so a root key can seed cycle detection without
    # incorrectly consuming one level.
    reference_depth: int = 0
    instance_cache: dict[str, SubstrateModule] = field(default_factory=dict)
    expansion_stack: list[str] = field(default_factory=list)


def _decode_ref_module(
    genome: Genome,
    n_inputs: int,
    n_outputs: int,
    library: ModuleLibrary | None = None,
    *,
    max_inline_depth: int,
    reference_depth: int,
    reference_stack: tuple[str, ...],
) -> SubstrateModule:
    resolver = macro_resolver(library) if library is not None else None
    try:
        return decode_module(
            genome,
            n_inputs,
            n_outputs,
            macro_resolver=resolver,
            max_inline_depth=max_inline_depth,
            _reference_depth=reference_depth,
            _reference_stack=reference_stack,
        )
    except ValueError as error:
        try:
            return decode_module(
                make_acyclic(genome),
                n_inputs,
                n_outputs,
                macro_resolver=resolver,
                max_inline_depth=max_inline_depth,
                _reference_depth=reference_depth,
                _reference_stack=reference_stack,
            )
        except ValueError as repaired_error:
            raise CompositionAssemblyError(str(repaired_error)) from error


def _resolve_module(node: CompNodeGene, ctx: AssemblyContext) -> SubstrateModule:
    if node.ref in ctx.instance_cache:
        return ctx.instance_cache[node.ref]
    if node.ref.startswith("live:"):
        if ctx.live_resolver is None:
            raise CompositionAssemblyError(f"no live resolver for ref {node.ref!r}")
        genome = ctx.live_resolver(node.ref.removeprefix("live:"))
        inner = _decode_ref_module(
            genome,
            node.in_width,
            node.out_width,
            ctx.library,
            max_inline_depth=ctx.max_inline_depth,
            reference_depth=ctx.reference_depth,
            reference_stack=tuple(ctx.expansion_stack),
        )
        if not node.trainable:
            for parameter in inner.parameters():
                parameter.requires_grad_(False)
    elif node.ref.startswith("library:"):
        if ctx.library is None:
            raise CompositionAssemblyError(f"no library attached for ref {node.ref!r}")
        key = node.ref.removeprefix("library:")
        if key in ctx.expansion_stack:
            raise CompositionAssemblyError(f"library reference cycle through entry {key!r}")
        if ctx.reference_depth >= ctx.max_inline_depth:
            raise CompositionAssemblyError(f"library reference depth exceeds max_inline_depth={ctx.max_inline_depth}")
        try:
            entry = ctx.library.load(key)
        except KeyError as error:
            raise CompositionAssemblyError(str(error)) from error
        next_depth = ctx.reference_depth + 1
        ctx.expansion_stack.append(key)
        try:
            if entry.entry_type == MODULE:
                inner = _decode_ref_module(
                    genome_from_dict(entry.payload),
                    node.in_width,
                    node.out_width,
                    ctx.library,
                    max_inline_depth=ctx.max_inline_depth,
                    reference_depth=next_depth,
                    reference_stack=tuple(ctx.expansion_stack),
                )
            elif entry.entry_type == COMPOSITION:
                inner_comp = comp_from_dict(entry.payload)
                inner = ComposedNet(inner_comp, _nested_context(inner_comp, ctx), n_inputs=node.in_width)
            else:
                raise CompositionAssemblyError(f"unknown entry type {entry.entry_type!r} for {key!r}")
        finally:
            ctx.expansion_stack.pop()
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
    columns: dict[str, Sequence[int]] = {}
    cursor = 0
    for node_id in inner_comp.input_ids:
        node = inner_comp.nodes[node_id]
        if node.ref == BIAS_REF:
            continue
        columns[node.ref] = range(cursor, cursor + node.out_width)
        cursor += node.out_width
    return AssemblyContext(
        bank_columns=columns,
        live_resolver=outer.live_resolver,
        library=outer.library,
        max_inline_depth=outer.max_inline_depth,
        reference_depth=outer.reference_depth + 1,
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
        try:
            self._order = comp_topological_order(comp)
        except ValueError as error:
            # Operators guard cycles, but a malformed comp (bad seed, id collision) must floor the
            # candidate via the assembly-error path, never crash the whole run.
            raise CompositionAssemblyError(str(error)) from error
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
        self.glue_u = nn.ParameterDict()  # factored edges: the effective map is glue_u[key] @ glue_v[key]
        self.glue_v = nn.ParameterDict()
        self._port_indices: dict[str, tuple[str, str]] = {}
        self._incoming: dict[int, list[tuple[int, str]]] = {}
        for edge in comp.enabled_edges():
            source, target = comp.nodes[edge.in_id], comp.nodes[edge.out_id]
            key = f"{edge.in_id}->{edge.out_id}"
            if edge.port_map is not None:
                if edge.glue_rank or len(edge.glue):
                    raise CompositionAssemblyError(f"fixed port map on {key} must not carry trainable glue")
                source_indices: list[int] = []
                target_indices: list[int] = []
                for run in edge.port_map.runs:
                    if run.length <= 0 or run.source_start < 0 or run.target_start < 0:
                        raise CompositionAssemblyError(f"invalid fixed port-map run on {key}: {run}")
                    if run.source_start + run.length > source.out_width or run.target_start + run.length > target.in_width:
                        raise CompositionAssemblyError(f"fixed port map on {key} exceeds {source.out_width}->{target.in_width} ports")
                    source_indices.extend(range(run.source_start, run.source_start + run.length))
                    target_indices.extend(range(run.target_start, run.target_start + run.length))
                source_name = f"port_source_{edge.in_id}_{edge.out_id}"
                target_name = f"port_target_{edge.in_id}_{edge.out_id}"
                self.register_buffer(source_name, torch.tensor(source_indices, dtype=torch.long), persistent=False)
                self.register_buffer(target_name, torch.tensor(target_indices, dtype=torch.long), persistent=False)
                self._port_indices[key] = (source_name, target_name)
            elif edge.glue_rank > 0:
                rank = edge.glue_rank
                expected = source.out_width * rank + rank * target.in_width
                if len(edge.glue) != expected:
                    raise CompositionAssemblyError(f"factored glue on {edge.in_id}->{edge.out_id} has {len(edge.glue)} values, expected {expected} (rank {rank})")
                split = source.out_width * rank
                values = _glue_tensor(edge.glue)
                self.glue_u[key] = nn.Parameter(values[:split].reshape(source.out_width, rank))
                self.glue_v[key] = nn.Parameter(values[split:].reshape(rank, target.in_width))
            else:
                expected = source.out_width * target.in_width
                if len(edge.glue) != expected:
                    raise CompositionAssemblyError(f"glue on {edge.in_id}->{edge.out_id} has {len(edge.glue)} values, expected {expected}")
                self.glue[key] = nn.Parameter(_glue_tensor(edge.glue).reshape(source.out_width, target.in_width))
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
            contributions = []
            for in_id, key in self._incoming.get(node_id, []):
                if key in self._port_indices:
                    source_name, target_name = self._port_indices[key]
                    source_indices = getattr(self, source_name)
                    target_indices = getattr(self, target_name)
                    gathered = values[in_id].index_select(1, source_indices)
                    placed = torch.zeros(x.shape[0], node.in_width, dtype=x.dtype, device=x.device)
                    placed.scatter_add_(1, target_indices.unsqueeze(0).expand(x.shape[0], -1), gathered)
                    contributions.append(placed)
                elif key in self.glue_u:
                    contributions.append((values[in_id] @ self.glue_u[key]) @ self.glue_v[key])
                else:
                    contributions.append(values[in_id] @ self.glue[key])
            if not contributions:
                combined = torch.zeros(x.shape[0], node.in_width, dtype=x.dtype, device=x.device)
            elif node.aggregation == "product":
                # Honor the aggregation mode independently of the contribution count: a single
                # contribution is just the product/sum of one term (identical value, clearer intent).
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


def _glue_tensor(values: GlueValues) -> torch.Tensor:
    if isinstance(values, array):
        return torch.frombuffer(values, dtype=torch.float32).clone()
    return torch.tensor(values, dtype=torch.float32)


def _glue_from_tensor(values: torch.Tensor, *, storage: str) -> GlueValues:
    flat = values.detach().to(device="cpu", dtype=torch.float32).contiguous().reshape(-1)
    if storage == "f32":
        compact = array("f")
        compact.frombytes(memoryview(flat.numpy()).cast("B"))
        return compact
    return tuple(float(value) for value in flat.tolist())


def coerce_glue_storage(values: GlueValues, storage: str) -> GlueValues:
    """Convert deterministic hand-built glue without changing its numeric values."""

    if storage == "f32":
        return values if isinstance(values, array) else array("f", values)
    return tuple(values)


def edge_storage_value_count(edge: CompEdgeGene) -> int:
    """Float-equivalent resident estimate including expanded gather/scatter index buffers."""

    if edge.port_map is not None:
        return 8 * edge.port_map.selected_count  # indices plus gather/scatter working buffers
    return len(edge.glue)


def writeback_composition(comp: CompositionGenome, net: ComposedNet) -> CompositionGenome:
    """Copy the net's trained glue back onto the matching enabled edge genes (Lamarckian glue)."""
    child = comp.clone()
    updated: list[CompEdgeGene] = []
    for edge in child.edges:
        key = f"{edge.in_id}->{edge.out_id}"
        storage = "f32" if isinstance(edge.glue, array) else "tuple"
        if edge.port_map is not None:
            updated.append(edge)
        elif edge.enabled and key in net.glue_u:
            factors = torch.cat((net.glue_u[key].detach().reshape(-1), net.glue_v[key].detach().reshape(-1)))
            updated.append(replace(edge, glue=_glue_from_tensor(factors, storage=storage)))
        elif edge.enabled and key in net.glue:
            updated.append(replace(edge, glue=_glue_from_tensor(net.glue[key], storage=storage)))
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
    # New glue auto-factorizes (rank = glue_rank) when in*out exceeds glue_rank_threshold (> 0).
    glue_rank: int = 0
    glue_rank_threshold: int = 0
    glue_storage: str = "tuple"


CompMutator = Callable[..., CompositionGenome]
COMP_MUTATION: Registry[CompMutator] = Registry("comp_mutation")
COMP_CROSSOVER: Registry[Callable[..., CompositionGenome]] = Registry("comp_crossover")


def _random_glue(count: int, rng: random.Random, sigma: float, storage: str) -> GlueValues:
    if storage == "tuple":
        return tuple(rng.gauss(0.0, sigma) for _ in range(count))
    if storage != "f32":
        raise ValueError(f"unknown glue storage {storage!r}; expected 'tuple' or 'f32'")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(rng.getrandbits(63))
    values = torch.randn(count, generator=generator, dtype=torch.float32).mul_(sigma)
    compact = array("f")
    compact.frombytes(memoryview(values.numpy()).cast("B"))
    return compact


def _glue_init(in_width: int, out_width: int, rng: random.Random, scale: float | None = None, storage: str = "tuple") -> GlueValues:
    sigma = scale if scale is not None else 1.0 / math.sqrt(max(in_width, 1))
    return _random_glue(in_width * out_width, rng, sigma, storage)


def _selected_glue_rank(in_width: int, out_width: int, glue_rank: int, glue_rank_threshold: int) -> int:
    if glue_rank > 0 and glue_rank_threshold > 0 and in_width * out_width > glue_rank_threshold and glue_rank < min(in_width, out_width):
        return glue_rank
    return 0


def glue_value_count(in_width: int, out_width: int, *, glue_rank: int = 0, glue_rank_threshold: int = 0) -> int:
    """Exact gene-value count for the dense/factored representation selected by `_glue_for`."""

    selected_rank = _selected_glue_rank(in_width, out_width, glue_rank, glue_rank_threshold)
    if selected_rank > 0:
        return in_width * selected_rank + selected_rank * out_width
    return in_width * out_width


def _glue_for(
    in_width: int,
    out_width: int,
    rng: random.Random,
    *,
    glue_rank: int = 0,
    glue_rank_threshold: int = 0,
    glue_scale: float | None = None,
    glue_storage: str = "tuple",
) -> tuple[GlueValues, int]:
    """Choose dense or factored glue for a NEW edge. Factored entries draw with sigma chosen so
    var((U @ V)_ij) = rank * sigma^4 matches the dense 1/in_width init variance."""
    selected_rank = _selected_glue_rank(in_width, out_width, glue_rank, glue_rank_threshold)
    if selected_rank > 0:
        sigma = (in_width * selected_rank) ** -0.25
        values = _random_glue(glue_value_count(in_width, out_width, glue_rank=selected_rank, glue_rank_threshold=glue_rank_threshold), rng, sigma, glue_storage)
        return values, selected_rank
    return _glue_init(in_width, out_width, rng, glue_scale, glue_storage), 0


def minimal_composition(
    input_specs: list[tuple[str, int]],
    output_ref: str,
    output_width: int,
    tracker: InnovationTracker,
    rng: random.Random,
    *,
    glue_scale: float | None = None,
    glue_rank: int = 0,
    glue_rank_threshold: int = 0,
    glue_storage: str = "tuple",
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
        glue, rank = _glue_for(width, output_width, rng, glue_rank=glue_rank, glue_rank_threshold=glue_rank_threshold, glue_scale=glue_scale, glue_storage=glue_storage)
        comp.edges.append(CompEdgeGene(source, output_id, True, tracker.innovation(source, output_id), glue, rank))
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
    glue_rank: int = 0,
    glue_rank_threshold: int = 0,
    glue_storage: str = "tuple",
) -> int:
    """Insert a MODULE node reading `source_id` and feeding `target_id`; returns the new node id."""
    node_id = tracker.new_node_id()
    comp.nodes[node_id] = CompNodeGene(node_id, CompNodeKind.MODULE, spec.ref, spec.in_width, spec.out_width)
    in_glue, in_rank = _glue_for(
        comp.nodes[source_id].out_width,
        spec.in_width,
        rng,
        glue_rank=glue_rank,
        glue_rank_threshold=glue_rank_threshold,
        glue_scale=glue_scale,
        glue_storage=glue_storage,
    )
    comp.edges.append(CompEdgeGene(source_id, node_id, True, tracker.innovation(source_id, node_id), in_glue, in_rank))
    out_glue, out_rank = _glue_for(
        spec.out_width,
        comp.nodes[target_id].in_width,
        rng,
        glue_rank=glue_rank,
        glue_rank_threshold=glue_rank_threshold,
        glue_scale=glue_scale,
        glue_storage=glue_storage,
    )
    comp.edges.append(CompEdgeGene(node_id, target_id, True, tracker.innovation(node_id, target_id), out_glue, out_rank))
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
    add_module_between(
        child,
        spec,
        source_id,
        rng.choice(candidates),
        ctx.innovations,
        rng,
        glue_rank=ctx.glue_rank,
        glue_rank_threshold=ctx.glue_rank_threshold,
        glue_storage=ctx.glue_storage,
    )
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
            glue, rank = _glue_for(
                child.nodes[source].out_width,
                child.nodes[target].in_width,
                rng,
                glue_rank=ctx.glue_rank,
                glue_rank_threshold=ctx.glue_rank_threshold,
                glue_storage=ctx.glue_storage,
            )
            child.edges.append(CompEdgeGene(source, target, True, ctx.innovations.innovation(source, target), glue, rank))
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
    updated: list[CompEdgeGene] = []
    for edge in child.edges:
        if rng.random() >= prob or edge.port_map is not None:
            updated.append(edge)
            continue
        if isinstance(edge.glue, array):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(rng.getrandbits(63))
            values = _glue_tensor(edge.glue)
            values.add_(torch.randn(values.shape, generator=generator, dtype=torch.float32), alpha=sigma)
            updated.append(replace(edge, glue=_glue_from_tensor(values, storage="f32")))
        else:
            updated.append(replace(edge, glue=tuple(value + rng.gauss(0.0, sigma) for value in edge.glue)))
    child.edges = updated
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
        if edge_b is None or edge_b.glue_rank != edge_a.glue_rank or edge_b.port_map != edge_a.port_map or len(edge_b.glue) != len(edge_a.glue) or rng.random() < 0.5:
            child_edges.append(edge_a)
        else:
            child_edges.append(edge_b)
    nodes: dict[int, CompNodeGene] = dict(parent_a.nodes)
    for edge in child_edges:
        for node_id in (edge.in_id, edge.out_id):
            if node_id not in nodes:
                nodes[node_id] = parent_b.nodes[node_id]
    return CompositionGenome(nodes=nodes, edges=child_edges)
