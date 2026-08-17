"""Recursive network rendering with a pure semantic-spec builder and hybrid raster/vector draw.

Missing, cyclic, or oversized references degrade to labeled opaque boxes; rendering is never part
of experiment correctness.
"""

import math
import os
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from versal.evolution.composition import CompNodeKind, CompositionGenome, comp_from_dict, comp_topological_order
from versal.evolution.genome import Genome, NodeKind, genome_from_dict, macro_implied_edges, make_acyclic, topological_order
from versal.library import COMPOSITION, MODULE, LibraryEntry, ModuleLibrary
from versal.motifs import FORWARD_EDGE, MACRO_EDGE, RECURRENT_EDGE, MotifRecord, NodeLabel
from versal.reference_depth import DEFAULT_MAX_INLINE_DEPTH

# async rendering
# Every RUNTIME render site (admission net portraits, speciation plots, the overmind portrait)
# funnels through `submit_render` so matplotlib's pyplot state only ever runs on ONE thread (pyplot
# is not thread-safe). Async mode ([run] render_async) moves renders off the run loop (a big
# admission PNG takes seconds); `flush_renders` joins before artifact upload, before library GC
# (which deletes images), and at run end. An async render failure logs and is dropped: a portrait
# is observability, never run state. Sync mode (the default) is byte-identical to before.
_RENDER_EXECUTOR: Any = None
_PENDING_RENDERS: list[Any] = []


def enable_async_rendering() -> None:
    global _RENDER_EXECUTOR
    if _RENDER_EXECUTOR is None:
        from concurrent.futures import ThreadPoolExecutor

        _RENDER_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="render")


def submit_render(render: Callable[..., object], *args: Any, **kwargs: Any) -> None:
    if _RENDER_EXECUTOR is None:
        render(*args, **kwargs)
        return

    def _safe() -> None:
        try:
            render(*args, **kwargs)
        except Exception as error:
            from versal.utils.logging import Logger

            Logger.get_logger().warning("async render failed: %s: %s", type(error).__name__, error)

    _PENDING_RENDERS.append(_RENDER_EXECUTOR.submit(_safe))


def flush_renders() -> None:
    global _PENDING_RENDERS
    pending, _PENDING_RENDERS = _PENDING_RENDERS, []
    for future in pending:
        future.result()


THEME: dict[str, Any] = {
    "background": "#10131a",
    "panel_even": "#171c26",
    "panel_odd": "#1b2130",
    "panel_opaque": "#2a3142",
    "container_edge": "#3d4666",
    "label": "#8b93b5",
    "title": "#c8cede",
    "edge_forward": "#aab3c5",
    "edge_positive": "#7dcfff",
    "edge_negative": "#ff6f91",
    "edge_mixed": "#7d86a3",
    "edge_recurrent": "#e0af68",
    "edge_macro": "#bb9af7",
    "edge_glue": "#7dcfff",
    "edge_callout": "#6fd08c",
    "edge_pathway": "#ff9e64",  # observed expert-to-expert routing traffic in the overmind
    "edge_entry": "#7aa2f7",  # overmind input feed (step-0 gate mass); echoes the input-node hue
    "edge_exit": "#f7768e",  # overmind output feed (final-step gate mass); echoes the output-node hue
    "node_input": "#7aa2f7",
    "node_bias": "#566190",
    "node_output": "#f7768e",
    "node_module": "#6fd08c",
    "node_anchor": "#f2cc60",
    "cmap": "viridis",
    "cmap_range": (0.25, 1.0),  # truncate the dark low end so layer-0 hidden nodes pop on the dark bg
}

# Compatibility export for callers/tests that named the former renderer-only constant. Runtime
# render entry points now receive the authoritative composition reference-depth policy explicitly.
RENDER_MAX_DEPTH = DEFAULT_MAX_INLINE_DEPTH
DEFAULT_NODE_BUDGET = 1500
GALLERY_NODE_BUDGET = 400

_H_GAP = 1.6
_V_GAP = 0.7
_PAD = 0.9
_CALLOUT_GAP = 1.4  # vertical clearance between the host network and the callout band
_NETWORK_ANCHOR_INSET = 0.3
_MAX_COLUMN_NODES = 64  # a layer taller than this wraps into a near-square block of sub-columns
_MAX_STRAIGHT_EDGES = 60_000  # line-collection ceiling before a scene becomes a density portrait
_MAX_CURVED_EDGES = 2_000  # FancyArrowPatch is one artist per curve; larger scenes use density
_DENSITY_WIDTH = 4800
_DENSITY_HEIGHT = 3200
_DENSITY_EDGE_CHUNK = 250_000
_EXPLICIT_EDGE_LIMIT = 512
_HYBRID_MAX_RASTER_DIMENSION = 4800
_HYBRID_MAX_RASTER_PIXELS = 4800 * 3200
_RENDER_DPI = 300
_MAX_RENDER_PIXELS = 36_000_000

_FEED_EDGE_ROLES = frozenset({"routing-entry", "routing-exit"})
_ROUTING_EDGE_ROLES = frozenset({"routing-observed", "routing-potential"})

ResolveFn = Callable[[str], LibraryEntry | None]


@dataclass(slots=True)
class SpecNode:
    x: float
    y: float
    color: str
    size: float = 1.0  # relative multiplier; draw_spec converts to point area from pixel density
    marker: str = "o"
    alpha: float = 1.0
    role: str = "node"


@dataclass(slots=True)
class SpecEdge:
    x0: float
    y0: float
    x1: float
    y1: float
    width: float
    color: str
    style: str = "solid"  # "solid" | "dashed"
    curve: float = 0.0  # arc3 rad; 0 draws via the fast LineCollection path
    alpha: float = 0.4
    role: str = "forward"
    magnitude: float = 1.0
    signed_weight: float | None = None


@dataclass(slots=True)
class SpecContainer:
    x0: float
    y0: float
    x1: float
    y1: float
    label: str
    depth: int
    opaque: bool = False


@dataclass(slots=True)
class SpecText:
    """Free-standing text in the shared frame (band labels, legend rows). Container labels stay on
    the container; this exists for text that belongs to no box."""

    x: float
    y: float
    text: str
    size: float = 8.0  # points, independent of data units, like container labels
    color: str = THEME["label"]
    ha: str = "left"
    va: str = "center"


@dataclass(slots=True)
class RenderSpec:
    nodes: list[SpecNode] = field(default_factory=list)
    edges: list[SpecEdge] = field(default_factory=list)
    containers: list[SpecContainer] = field(default_factory=list)
    texts: list[SpecText] = field(default_factory=list)
    width: float = 1.0
    height: float = 1.0
    flow_label: str | None = None

    @property
    def node_count(self) -> int:
        return len(self.nodes)


@dataclass(slots=True)
class _LargeRenderMetadata:
    """Honest accounting for one aggregate portrait (kept internal; `render_entry` still returns a path)."""

    node_count: int
    enabled_edge_count: int
    isolated_input_count: int
    rendered_edge_count: int
    semantic_layout_mode: str
    canvas_width: int = _DENSITY_WIDTH
    canvas_height: int = _DENSITY_HEIGHT
    fallback_reason: str | None = None


@dataclass(slots=True)
class _DensityPanel:
    x0: float
    y0: float
    x1: float
    y1: float
    label: str
    node_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class _DensityLayout:
    positions: dict[int, tuple[float, float]]
    input_ids: list[int]
    computed_ids: list[int]
    panels: list[_DensityPanel]
    mode: str
    fallback_reason: str | None


@dataclass(slots=True)
class _FlowStats:
    count: int = 0
    magnitude: float = 0.0
    signed: float = 0.0

    def add(self, weight: float) -> None:
        self.count += 1
        self.magnitude += abs(weight)
        self.signed += weight


def library_resolver(library: ModuleLibrary) -> ResolveFn:
    def resolve(key: str) -> LibraryEntry | None:
        try:
            return library.load(key)
        except KeyError:
            return None

    return resolve


# internal recursion


@dataclass(slots=True)
class _Budget:
    remaining: int


@dataclass(slots=True)
class _Built:
    """A nested network in LOCAL coordinates (extent [0, width] x [0, height]). The parent translates
    everything at placement. `output_nodes` retains the outer network's concrete outputs so parent
    flow edges leave the computation rather than an arbitrary card point."""

    spec: RenderSpec
    label: str = ""
    opaque: bool = False
    output_nodes: list[SpecNode] = field(default_factory=list)

    def translate(self, dx: float, dy: float) -> None:
        for node in self.spec.nodes:
            node.x += dx
            node.y += dy
        for edge in self.spec.edges:
            edge.x0 += dx
            edge.y0 += dy
            edge.x1 += dx
            edge.y1 += dy
        for box in self.spec.containers:
            box.x0 += dx
            box.y0 += dy
            box.x1 += dx
            box.y1 += dy
        for text in self.spec.texts:
            text.x += dx
            text.y += dy


@dataclass(slots=True)
class _Item:
    """One placeable node cell in the layered host layout."""

    key: int
    layer: int
    preds: list[int]
    sort_rank: tuple[int, int]


def _layer_shape(count: int) -> tuple[int, int]:
    """(rows, sub_columns) for one layer. A layer taller than _MAX_COLUMN_NODES wraps into a
    near-square block; below the cap it stays a single column, byte-identical to the old layout.
    Wide I/O banks (an ARC output field is 9,000 nodes) would otherwise stretch the frame to
    thousands of units tall while set_aspect("equal") crushes the X axis into a one-pixel line."""
    if count <= _MAX_COLUMN_NODES:
        return count, 1
    rows = math.ceil(math.sqrt(count))
    return rows, math.ceil(count / rows)


def _place_items(items: list[_Item]) -> tuple[dict[int, tuple[float, float]], float, float]:
    """Layered layout: x = layer column, y = stacked with barycenter ordering to reduce crossings.
    Oversized layers wrap into blocks of sub-columns, filled column-major in sort order (a grid
    field stamped in raster order redraws as its grid).

    The frame is [0, width] x [0, height] with every column vertically centered. Returns
    (center positions, width, height)."""
    by_layer: dict[int, list[_Item]] = {}
    for item in items:
        by_layer.setdefault(item.layer, []).append(item)
    layers = sorted(by_layer)
    if not layers:
        return {}, 1.0, 1.0

    pitch = 1.0 + _V_GAP  # node pitch, shared by stacked rows and wrapped sub-columns
    shapes = {k: _layer_shape(len(by_layer[k])) for k in layers}
    col_x: dict[int, float] = {}
    cursor_x = 0.5
    for k in layers:
        col_x[k] = cursor_x
        cursor_x += (shapes[k][1] - 1) * pitch + 1.0 + _H_GAP
    width = cursor_x - 1.0 - _H_GAP + 0.5

    height = max(shapes[k][0] + _V_GAP * (shapes[k][0] - 1) for k in layers)
    centers: dict[int, tuple[float, float]] = {}
    for index, k in enumerate(layers):
        column = by_layer[k]
        if index == 0:
            column.sort(key=lambda item: item.sort_rank)
        else:

            def barycenter(item: _Item) -> float:
                placed = [centers[pred][1] for pred in item.preds if pred in centers]
                return sum(placed) / len(placed) if placed else 0.0

            column.sort(key=barycenter, reverse=True)  # highest predecessor mass lands on top
        rows = shapes[k][0]
        total = rows + _V_GAP * (rows - 1)
        top = height / 2 + total / 2 - 0.5
        for position, item in enumerate(column):
            sub_column, row = divmod(position, rows)
            centers[item.key] = (col_x[k] + sub_column * pitch, top - row * pitch)

    return centers, width, height


def _opaque_built(label: str) -> _Built:
    return _Built(spec=RenderSpec(width=1.6, height=1.2), label=label, opaque=True)


def _entry_size_label(entry: LibraryEntry) -> str:
    nodes = len(entry.payload.get("nodes", []))
    edge_field = "connections" if entry.entry_type == MODULE else "edges"
    edges = len(entry.payload.get(edge_field, []))
    return f"{nodes:,} nodes · {edges:,} edges"


def _place_child(spec: RenderSpec, child: _Built, center: tuple[float, float], depth: int) -> None:
    """Translate a child into the parent frame at `center` and merge it, wrapped in its container."""
    cx, cy = center
    half_w, half_h = child.spec.width / 2, child.spec.height / 2
    child.translate(cx - half_w, cy - half_h)
    spec.nodes.extend(child.spec.nodes)
    spec.edges.extend(child.spec.edges)
    spec.containers.extend(child.spec.containers)
    spec.texts.extend(child.spec.texts)
    spec.containers.append(SpecContainer(cx - half_w - _PAD, cy - half_h - _PAD, cx + half_w + _PAD, cy + half_h + _PAD, label=child.label, depth=depth, opaque=child.opaque))


def _attach_callouts(
    spec: RenderSpec,
    callouts: list[tuple[_Built, list[tuple[float, float]]]],
    host_width: float,
    host_height: float,
    depth: int,
) -> tuple[float, float, list[tuple[float, float]]]:
    """Pack expanded child boxes across the TOP of the host frame. Each parent footprint connects
    to a gold input anchor at the nested card's top-left. Returns the combined extent and each card's
    center in callout order."""
    if not callouts:
        return host_width, host_height, []
    boxes = [(child.spec.width + 2 * _PAD, child.spec.height + 2 * _PAD) for child, _ in callouts]
    available = max(host_width, max(box_w for box_w, _ in boxes))

    # Center the host under a wider callout band so the green lines hang symmetrically.
    host_shift = (available - host_width) / 2
    if host_shift > 0:
        _Built(spec=spec).translate(host_shift, 0.0)
        callouts = [(child, [(x + host_shift, y) for x, y in source_points]) for child, source_points in callouts]

    rows: list[list[int]] = [[]]
    cursor = 0.0
    for index, (box_w, _box_h) in enumerate(boxes):
        advance = box_w if not rows[-1] else box_w + _H_GAP
        if rows[-1] and cursor + advance > available:
            rows.append([index])
            cursor = box_w
        else:
            rows[-1].append(index)
            cursor += advance

    centers: list[tuple[float, float]] = [(0.0, 0.0)] * len(callouts)
    row_base = host_height + _CALLOUT_GAP
    for row in rows:
        row_height = max(boxes[i][1] for i in row)
        row_width = sum(boxes[i][0] for i in row) + _H_GAP * (len(row) - 1)
        x_cursor = (available - row_width) / 2  # center each row over the host
        for i in row:
            box_w, box_h = boxes[i]
            child, source_points = callouts[i]
            cx, cy = x_cursor + box_w / 2, row_base + box_h / 2
            centers[i] = (cx, cy)
            _place_child(spec, child, (cx, cy), depth)
            target_x, target_y = x_cursor + _NETWORK_ANCHOR_INSET, row_base + box_h - _NETWORK_ANCHOR_INSET
            spec.nodes.append(SpecNode(target_x, target_y, color=THEME["node_anchor"], size=0.65, role="network-input-anchor"))
            for source_x, source_y in source_points:
                spec.edges.append(SpecEdge(source_x, source_y, target_x, target_y, width=1.0, color=THEME["edge_callout"], alpha=0.55, role="nested-network"))
            x_cursor += box_w + _H_GAP
        row_base += row_height + _V_GAP

    return available, row_base - _V_GAP, centers


def _build_entry(
    entry: LibraryEntry,
    *,
    resolve: ResolveFn | None,
    budget: _Budget,
    depth: int,
    reference_depth: int,
    stack: tuple[str, ...],
    max_inline_depth: int,
) -> _Built:
    label = f"{entry.key}  L{entry.level}"
    if len(entry.payload["nodes"]) > budget.remaining:
        return _opaque_built(f"{label}\n{_entry_size_label(entry)}\ndetail exceeds render budget")
    if entry.entry_type == MODULE:
        built = _build_genome(
            genome_from_dict(entry.payload),
            resolve=resolve,
            budget=budget,
            depth=depth,
            reference_depth=reference_depth,
            stack=stack,
            max_inline_depth=max_inline_depth,
        )
        if "field_template" in entry.payload:
            version = entry.payload["field_template"].get("version", "unknown")
            label = f"{label}  repeated field H×W  {version}"
            built.spec.containers.append(SpecContainer(0.0, 0.0, built.spec.width, built.spec.height, label=label, depth=depth))
    elif entry.entry_type == COMPOSITION:
        built = _build_comp(
            comp_from_dict(entry.payload),
            resolve=resolve,
            budget=budget,
            depth=depth,
            reference_depth=reference_depth,
            stack=stack,
            max_inline_depth=max_inline_depth,
        )
    else:
        return _opaque_built(f"{label}  ?")
    built.label = label
    return built


def _build_ref(
    ref: str,
    *,
    resolve: ResolveFn | None,
    budget: _Budget,
    depth: int,
    reference_depth: int,
    stack: tuple[str, ...],
    max_inline_depth: int,
) -> _Built:
    if not ref.startswith("library:"):
        return _opaque_built(ref)  # live refs only exist mid-run; renders happen on admitted entries
    key = ref.removeprefix("library:")
    if key in stack or reference_depth >= max_inline_depth or resolve is None:
        return _opaque_built(key)
    entry = resolve(key)
    if entry is None:
        return _opaque_built(key)
    try:
        return _build_entry(
            entry,
            resolve=resolve,
            budget=budget,
            depth=depth + 1,
            reference_depth=reference_depth + 1,
            stack=(*stack, key),
            max_inline_depth=max_inline_depth,
        )
    except Exception:
        return _opaque_built(f"{entry.key}  L{entry.level}  ?")


def _edge_width(strength: float) -> float:
    return 0.6 + 2.4 * min(abs(strength), 3.0) / 3.0


def _layer_color(layer: int, max_layer: int) -> str:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colors as mcolors
    from matplotlib import pyplot as plt

    low, high = THEME["cmap_range"]
    t = low + (high - low) * (layer / max(max_layer, 1))
    return str(mcolors.to_hex(plt.get_cmap(THEME["cmap"])(t)))


def _build_genome(
    genome: Genome,
    *,
    resolve: ResolveFn | None,
    budget: _Budget,
    depth: int,
    reference_depth: int,
    stack: tuple[str, ...],
    max_inline_depth: int,
) -> _Built:
    spec = RenderSpec()
    if not genome.nodes:
        return _Built(spec=spec)

    # Layering: longest-path depth over forward + macro-implied edges. Cyclic genomes (crossover
    # artifacts) fall back to a pruned copy for LAYOUT only; edges still draw from the original.
    layout_genome = genome
    try:
        order = topological_order(layout_genome)
    except ValueError:
        layout_genome = make_acyclic(genome)
        order = topological_order(layout_genome)
    incoming: dict[int, list[int]] = {}
    for conn in layout_genome.forward_connections():
        incoming.setdefault(conn.out_id, []).append(conn.in_id)
    for source, target in macro_implied_edges(layout_genome):
        incoming.setdefault(target, []).append(source)
    layer: dict[int, int] = {}
    for node_id in order:
        predecessors = incoming.get(node_id, [])
        layer[node_id] = 0 if not predecessors else 1 + max(layer[pred] for pred in predecessors)

    stub_ids = genome.macro_output_ids

    degree: dict[int, int] = {node_id: 0 for node_id in genome.nodes}
    for conn in genome.enabled_connections():
        degree[conn.in_id] = degree.get(conn.in_id, 0) + 1
        degree[conn.out_id] = degree.get(conn.out_id, 0) + 1
    for source, target in macro_implied_edges(genome):
        degree[source] = degree.get(source, 0) + 1
        degree[target] = degree.get(target, 0) + 1

    kind_rank = {NodeKind.INPUT: 0, NodeKind.BIAS: 1, NodeKind.HIDDEN: 2, NodeKind.OUTPUT: 3}
    items = [_Item(key=node.id, layer=layer.get(node.id, 0), preds=incoming.get(node.id, []), sort_rank=(kind_rank[node.kind], node.id)) for node in genome.nodes.values()]
    positions, host_width, host_height = _place_items(items)
    budget.remaining -= len(genome.nodes)

    drawn_max_layer = max(layer.values(), default=1) or 1
    output_nodes: list[SpecNode] = []
    for node in genome.nodes.values():
        x, y = positions[node.id]
        node_degree = degree.get(node.id, 0)
        isolated = node_degree == 0
        if node.id in stub_ids:
            # A macro's footprint in the host: a green hexagon (distinct from viridis hidden
            # circles), matching the callout line up to its expansion.
            color, marker, alpha, role = THEME["node_module"], "h", 1.0, "macro-footprint"
        elif node.kind is NodeKind.INPUT:
            color, marker, alpha, role = THEME["node_input"], "s", 1.0, "input"
        elif node.kind is NodeKind.BIAS:
            color, marker, alpha, role = THEME["node_bias"], "s", 0.7, "bias"
        elif node.kind is NodeKind.OUTPUT:
            color, marker, alpha, role = THEME["node_output"], "s", 1.0, "output"
        else:
            color, marker, alpha, role = _layer_color(layer.get(node.id, 0), drawn_max_layer), ("D" if node.aggregation == "product" else "o"), 1.0, "hidden"
        size = 0.5 if isolated else min(1.0 + 0.15 * node_degree, 2.5)
        drawn = SpecNode(x, y, color, size=size, marker=marker, alpha=0.25 if isolated else alpha, role="isolated" if isolated else role)
        spec.nodes.append(drawn)
        if node.kind is NodeKind.OUTPUT:
            output_nodes.append(drawn)

    for conn in genome.enabled_connections():
        source, target = positions.get(conn.in_id), positions.get(conn.out_id)
        if source is None or target is None:
            continue
        if conn.recurrent:
            spec.edges.append(
                SpecEdge(
                    source[0],
                    source[1],
                    target[0],
                    target[1],
                    width=_edge_width(conn.weight),
                    color=THEME["edge_recurrent"],
                    style="dashed",
                    curve=0.25,
                    alpha=0.6,
                    role="recurrent",
                    magnitude=abs(conn.weight),
                    signed_weight=conn.weight,
                )
            )
        else:
            spec.edges.append(
                SpecEdge(
                    source[0],
                    source[1],
                    target[0],
                    target[1],
                    width=_edge_width(conn.weight),
                    color=THEME["edge_positive"] if conn.weight >= 0.0 else THEME["edge_negative"],
                    alpha=0.42,
                    role="forward-positive" if conn.weight >= 0.0 else "forward-negative",
                    magnitude=abs(conn.weight),
                    signed_weight=conn.weight,
                )
            )
    for source, target in macro_implied_edges(genome):
        source_pos, target_pos = positions.get(source), positions.get(target)
        if source_pos is None or target_pos is None:
            continue
        spec.edges.append(SpecEdge(source_pos[0], source_pos[1], target_pos[0], target_pos[1], width=1.0, color=THEME["edge_macro"], alpha=0.5, role="macro-implied"))

    callouts: list[tuple[_Built, list[tuple[float, float]]]] = []
    for macro in genome.macros:
        child = _build_ref(
            macro.ref,
            resolve=resolve,
            budget=budget,
            depth=depth,
            reference_depth=reference_depth,
            stack=stack,
            max_inline_depth=max_inline_depth,
        )
        anchors = [positions[stub_id] for stub_id in macro.output_node_ids if stub_id in positions]
        callouts.append((child, anchors))
    spec.width, spec.height, _centers = _attach_callouts(spec, callouts, host_width, host_height, depth + 1)
    return _Built(spec=spec, output_nodes=output_nodes)


def _build_comp(
    comp: CompositionGenome,
    *,
    resolve: ResolveFn | None,
    budget: _Budget,
    depth: int,
    reference_depth: int,
    stack: tuple[str, ...],
    max_inline_depth: int,
) -> _Built:
    spec = RenderSpec()
    if not comp.nodes:
        return _Built(spec=spec)

    layer: dict[int, int] = {}
    incoming: dict[int, list[int]] = {}
    for edge in comp.enabled_edges():
        incoming.setdefault(edge.out_id, []).append(edge.in_id)
    try:
        for node_id in comp_topological_order(comp):
            predecessors = incoming.get(node_id, [])
            layer[node_id] = 0 if not predecessors else 1 + max(layer[pred] for pred in predecessors)
    except ValueError:
        layer = {node_id: 0 for node_id in comp.nodes}

    kind_rank = {CompNodeKind.INPUT: 0, CompNodeKind.MODULE: 1, CompNodeKind.OUTPUT: 2}
    items = [_Item(key=node.id, layer=layer.get(node.id, 0), preds=incoming.get(node.id, []), sort_rank=(kind_rank[node.kind], node.id)) for node in comp.nodes.values()]
    positions, host_width, host_height = _place_items(items)
    budget.remaining -= len(comp.nodes)

    output_nodes: list[SpecNode] = []
    for node in comp.nodes.values():
        x, y = positions[node.id]
        if node.kind is CompNodeKind.MODULE:
            color = THEME["node_module"]
            marker = "h"  # hexagon: the footprint shape, matching the green callout line
            size = 1.0 + math.log2(1 + max(node.in_width, node.out_width)) / 4
            role = "module-footprint"
        elif node.kind is CompNodeKind.INPUT:
            color = THEME["node_bias"] if node.ref == "__bias__" else THEME["node_input"]
            marker = "s"
            size = 1.0 + math.log2(1 + node.out_width) / 4
            role = "bias" if node.ref == "__bias__" else "input"
        else:
            color = THEME["node_output"]
            marker = "s"
            size = 1.0 + math.log2(1 + node.in_width) / 4
            role = "output"
        drawn = SpecNode(x, y, color, size=size, marker=marker, role=role)
        spec.nodes.append(drawn)
        if node.kind is CompNodeKind.OUTPUT:
            output_nodes.append(drawn)

    # Glue is a width x width linear map, so each comp edge draws as ONE aggregate strand between
    # node centers, never a per-neuron fan-out.
    for edge in comp.enabled_edges():
        source, target = positions.get(edge.in_id), positions.get(edge.out_id)
        if source is None or target is None:
            continue
        strength = max((abs(value) for value in edge.glue), default=0.0)
        signed = sum(edge.glue)
        spec.edges.append(
            SpecEdge(
                source[0],
                source[1],
                target[0],
                target[1],
                width=_edge_width(strength),
                color=THEME["edge_positive"] if signed >= 0.0 else THEME["edge_negative"],
                alpha=0.55,
                role="composition-glue-positive" if signed >= 0.0 else "composition-glue-negative",
                magnitude=sum(abs(value) for value in edge.glue),
                signed_weight=signed,
            )
        )

    callouts: list[tuple[_Built, list[tuple[float, float]]]] = []
    for node in comp.nodes.values():
        if node.kind is CompNodeKind.MODULE:
            child = _build_ref(
                node.ref,
                resolve=resolve,
                budget=budget,
                depth=depth,
                reference_depth=reference_depth,
                stack=stack,
                max_inline_depth=max_inline_depth,
            )
            callouts.append((child, [positions[node.id]]))
    spec.width, spec.height, _centers = _attach_callouts(spec, callouts, host_width, host_height, depth + 1)
    return _Built(spec=spec, output_nodes=output_nodes)


# public builders


def build_genome_spec(
    genome: Genome,
    *,
    resolve: ResolveFn | None = None,
    node_budget: int = DEFAULT_NODE_BUDGET,
    max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH,
) -> RenderSpec:
    spec = _build_genome(
        genome,
        resolve=resolve,
        budget=_Budget(node_budget),
        depth=0,
        reference_depth=0,
        stack=(),
        max_inline_depth=max_inline_depth,
    ).spec
    spec.flow_label = "potential influence flow · weights/topology, not activations"
    return spec


def build_composition_spec(
    comp: CompositionGenome,
    *,
    resolve: ResolveFn | None = None,
    node_budget: int = DEFAULT_NODE_BUDGET,
    max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH,
) -> RenderSpec:
    spec = _build_comp(
        comp,
        resolve=resolve,
        budget=_Budget(node_budget),
        depth=0,
        reference_depth=0,
        stack=(),
        max_inline_depth=max_inline_depth,
    ).spec
    spec.flow_label = "potential influence flow · topology/glue, not activations"
    return spec


def build_entry_spec(
    entry: LibraryEntry,
    *,
    resolve: ResolveFn | None = None,
    node_budget: int = DEFAULT_NODE_BUDGET,
    max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH,
) -> RenderSpec:
    try:
        built = _build_entry(
            entry,
            resolve=resolve,
            budget=_Budget(node_budget),
            depth=0,
            reference_depth=0,
            stack=(entry.key,),
            max_inline_depth=max_inline_depth,
        )
        if built.opaque:
            built.spec.containers.append(SpecContainer(0.0, 0.0, built.spec.width, built.spec.height, label=built.label, depth=0, opaque=True))
        built.spec.flow_label = (
            "potential influence flow · weights/topology, not activations" if entry.entry_type == MODULE else "potential influence flow · topology/glue, not activations"
        )
        return built.spec
    except Exception:
        pass
    built = _opaque_built(f"{entry.key}  ?")
    built.spec.containers.append(SpecContainer(0.0, 0.0, built.spec.width, built.spec.height, label=built.label, depth=0, opaque=True))
    built.spec.flow_label = "potential influence flow · topology, not activations"
    return built.spec


# painting


def _edge_segments(edge: SpecEdge) -> list[tuple[float, float, float, float]]:
    """Turn an edge into deterministic line segments that Datashader can aggregate.

    Curves are quadratic approximations of Matplotlib's arc cue. Dashed edges omit alternating
    segments, so recurrent structure remains recognizable even after density aggregation.
    """
    if edge.curve == 0.0:
        return [(edge.x0, edge.y0, edge.x1, edge.y1)]
    dx, dy = edge.x1 - edge.x0, edge.y1 - edge.y0
    control_x = (edge.x0 + edge.x1) / 2 - dy * edge.curve
    control_y = (edge.y0 + edge.y1) / 2 + dx * edge.curve
    points: list[tuple[float, float]] = []
    for index in range(13):
        t = index / 12
        inverse = 1.0 - t
        points.append(
            (
                inverse * inverse * edge.x0 + 2 * inverse * t * control_x + t * t * edge.x1,
                inverse * inverse * edge.y0 + 2 * inverse * t * control_y + t * t * edge.y1,
            )
        )
    segments = [(x0, y0, x1, y1) for (x0, y0), (x1, y1) in zip(points, points[1:])]
    return segments[::2] if edge.style == "dashed" else segments


def _rasterized_spec_edges(
    spec: RenderSpec,
    *,
    pixel_width: int,
    pixel_height: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[Any, int]:
    """Rasterize every scene edge by semantic color; no sampling or per-edge artists."""
    import datashader as ds
    import datashader.transfer_functions as tf
    import numpy as np
    import pandas as pd

    canvas = ds.Canvas(plot_width=max(pixel_width, 2), plot_height=max(pixel_height, 2), x_range=x_range, y_range=y_range)
    layers: dict[tuple[str, int], Any] = {}
    rows: dict[tuple[str, int], list[tuple[float, float, float, float, float]]] = {}
    pending_segments = 0

    def flush() -> None:
        nonlocal pending_segments
        for visual, color_rows in rows.items():
            if not color_rows:
                continue
            frame = pd.DataFrame(color_rows, columns=["x0", "y0", "x1", "y1", "weight"])
            layers[visual] = _accumulate_density_lines(canvas, frame, ds, np, layers.get(visual))
        rows.clear()
        pending_segments = 0

    for edge in spec.edges:
        alpha = max(0.0, min(float(edge.alpha), 1.0))
        if alpha <= 0.0:
            continue
        magnitude = edge.magnitude if math.isfinite(edge.magnitude) else 0.0
        # Zero-weight structural edges must remain visible, but never dominate weighted density.
        weight = max(abs(magnitude), 0.05)
        segments = _edge_segments(edge)
        visual = (edge.color, round(alpha * 255))
        rows.setdefault(visual, []).extend((*segment, weight) for segment in segments)
        pending_segments += len(segments)
        if pending_segments >= _DENSITY_EDGE_CHUNK:
            flush()
    flush()

    shaded = [
        tf.shade(
            layer.where(layer > 0.0),
            cmap=[color, color],
            how="log",
            min_alpha=max(1, round(alpha_byte * 0.16)),
            alpha=alpha_byte,
        )
        for (color, alpha_byte), layer in layers.items()
    ]
    if not shaded:
        return np.zeros((max(pixel_height, 2), max(pixel_width, 2), 4), dtype=np.uint8), 0
    return np.asarray(tf.stack(*shaded, how="over").to_pil()), len(spec.edges)


def _rasterized_spec_nodes(
    spec: RenderSpec,
    *,
    pixel_width: int,
    pixel_height: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> Any:
    """Create a subtle Datashader node-density glow beneath the explicit semantic markers."""
    import datashader as ds
    import datashader.transfer_functions as tf
    import numpy as np
    import pandas as pd

    if len(spec.nodes) <= 256:
        return None
    canvas = ds.Canvas(plot_width=max(pixel_width, 2), plot_height=max(pixel_height, 2), x_range=x_range, y_range=y_range)
    shaded = []
    by_color: dict[str, list[SpecNode]] = {}
    for node in spec.nodes:
        density_color = {
            "input": THEME["node_input"],
            "bias": THEME["node_bias"],
            "output": THEME["node_output"],
            "macro-footprint": THEME["node_module"],
            "module-footprint": THEME["node_module"],
            "network-input-anchor": THEME["node_anchor"],
            "isolated": THEME["edge_mixed"],
        }.get(node.role, "#2eb5a7")
        by_color.setdefault(density_color, []).append(node)
    for color, nodes in by_color.items():
        frame = pd.DataFrame({"x": [node.x for node in nodes], "y": [node.y for node in nodes], "weight": [max(node.size, 0.05) for node in nodes]})
        aggregate = canvas.points(frame, x="x", y="y", agg=ds.sum("weight"))
        shaded.append(tf.shade(aggregate.where(aggregate > 0.0), cmap=[color, color], how="log", min_alpha=24, alpha=135))
    return np.asarray(tf.stack(*shaded, how="over").to_pil())


def _draw_classic_edges(axis: Any, edges: list[SpecEdge], *, directional: bool, zorder: float = 2.2) -> None:
    """Crisp semantic overlay for small scenes and a resilient fallback if rasterization fails."""
    from matplotlib import colors as mcolors
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyArrowPatch

    if directional:
        for edge in edges:
            axis.add_patch(
                FancyArrowPatch(
                    (edge.x0, edge.y0),
                    (edge.x1, edge.y1),
                    connectionstyle=f"arc3,rad={edge.curve}",
                    color=mcolors.to_rgba(edge.color, edge.alpha),
                    linewidth=edge.width,
                    linestyle=edge.style,
                    arrowstyle="-|>",
                    mutation_scale=5.0,
                    shrinkA=1.5,
                    shrinkB=1.5,
                    zorder=zorder,
                )
            )
        return

    # Medium scenes use collections directly; the same caps keep failure fallback bounded.
    straight = [edge for edge in edges if edge.curve == 0.0][:_MAX_STRAIGHT_EDGES]
    curved = [edge for edge in edges if edge.curve != 0.0][:_MAX_CURVED_EDGES]
    for style in ("solid", "dashed"):
        group = [edge for edge in straight if edge.style == style]
        if group:
            axis.add_collection(
                LineCollection(
                    [((edge.x0, edge.y0), (edge.x1, edge.y1)) for edge in group],
                    colors=[mcolors.to_rgba(edge.color, edge.alpha) for edge in group],
                    linewidths=[edge.width for edge in group],
                    linestyle=style,
                    zorder=zorder,
                )
            )
    for edge in curved:
        axis.add_patch(
            FancyArrowPatch(
                (edge.x0, edge.y0),
                (edge.x1, edge.y1),
                connectionstyle=f"arc3,rad={edge.curve}",
                color=mcolors.to_rgba(edge.color, edge.alpha),
                linewidth=edge.width,
                linestyle=edge.style,
                arrowstyle="-",
                zorder=zorder,
            )
        )


def _draw_potential_flow_legend(axis: Any, spec: RenderSpec) -> None:
    if not spec.flow_label or not spec.flow_label.startswith("potential influence flow"):
        return
    roles = {edge.role for edge in spec.edges}
    entries: list[tuple[str, str]] = []
    if "forward-positive" in roles or "composition-glue-positive" in roles:
        entries.append((THEME["edge_positive"], "positive influence"))
    if "forward-negative" in roles or "composition-glue-negative" in roles:
        entries.append((THEME["edge_negative"], "negative influence"))
    if "recurrent" in roles:
        entries.append((THEME["edge_recurrent"], "recurrent"))
    if "macro-implied" in roles:
        entries.append((THEME["edge_macro"], "macro-implied"))
    if "nested-network" in roles:
        entries.append((THEME["edge_callout"], "nested flow"))
    if not entries:
        return
    step = min(0.19, 0.94 / len(entries))
    for index, (color, label) in enumerate(entries):
        x = 0.02 + index * step
        axis.plot((x, x + 0.025), (0.034, 0.034), transform=axis.transAxes, color=color, linewidth=2.0, zorder=6)
        axis.text(x + 0.031, 0.034, label, transform=axis.transAxes, color=THEME["label"], fontsize=5.5, ha="left", va="center", zorder=6)


def draw_spec(axis: Any, spec: RenderSpec, *, title: str | None = None, x_padding: float = _PAD, show_footer: bool = True) -> None:
    """Paint a semantic scene with Datashader density beneath a crisp Matplotlib overlay."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colors as mcolors
    from matplotlib.patches import FancyBboxPatch

    axis.set_facecolor(THEME["background"])
    for box in sorted(spec.containers, key=lambda item: item.depth):
        fill = THEME["panel_opaque"] if box.opaque else (THEME["panel_even"] if box.depth % 2 == 0 else THEME["panel_odd"])
        axis.add_patch(
            FancyBboxPatch(
                (box.x0, box.y0),
                box.x1 - box.x0,
                box.y1 - box.y0,
                boxstyle="round,pad=0,rounding_size=0.3",
                facecolor=fill,
                edgecolor=THEME["container_edge"],
                linewidth=0.8,
                zorder=1,
            )
        )
        if box.opaque:
            # An opaque box has no inner drawing; the label IS its content.
            axis.text((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, box.label, fontsize=max(8 - box.depth, 5), color=THEME["label"], ha="center", va="center", zorder=4)
        elif box.x1 - box.x0 >= 2.5 and box.depth < 3:
            # Above the box, in the parent's padding gap, so it never collides with inner nodes.
            axis.text(box.x0 + 0.15, box.y1 + 0.08, box.label, fontsize=max(8 - box.depth, 5), color=THEME["label"], ha="left", va="bottom", zorder=4)

    for text in spec.texts:
        axis.text(text.x, text.y, text.text, fontsize=text.size, color=text.color, ha=text.ha, va=text.va, zorder=4)

    figure = axis.figure
    fig_w, fig_h = figure.get_size_inches()
    x_range = (-x_padding, spec.width + x_padding)
    y_range = (-_PAD, spec.height + _PAD)
    raw_pixel_width = max(64, int(fig_w * figure.dpi))
    raw_pixel_height = max(64, int(fig_h * figure.dpi))
    max_raster_dimension = _HYBRID_MAX_RASTER_DIMENSION
    max_raster_pixels = _HYBRID_MAX_RASTER_PIXELS
    raster_scale = min(
        1.0,
        max_raster_dimension / raw_pixel_width,
        max_raster_dimension / raw_pixel_height,
        math.sqrt(max_raster_pixels / (raw_pixel_width * raw_pixel_height)),
    )
    pixel_width = max(64, int(raw_pixel_width * raster_scale))
    pixel_height = max(64, int(raw_pixel_height * raster_scale))
    feed_edges = [edge for edge in spec.edges if edge.role in _FEED_EDGE_ROLES]
    routing_edges = [edge for edge in spec.edges if edge.role in _ROUTING_EDGE_ROLES]
    legend_edges = [edge for edge in spec.edges if edge.role == "legend"]
    network_edges = [edge for edge in spec.edges if edge.role not in _FEED_EDGE_ROLES | _ROUTING_EDGE_ROLES and edge.role != "legend"]
    rendered_edges = 0
    edge_note = ""
    straight_count = sum(edge.curve == 0.0 for edge in network_edges)
    curved_count = len(network_edges) - straight_count
    rasterize_feeds = len(feed_edges) > _EXPLICIT_EDGE_LIMIT
    rasterize_network = len(network_edges) > _EXPLICIT_EDGE_LIMIT and (straight_count > _MAX_STRAIGHT_EDGES or curved_count > _MAX_CURVED_EDGES)
    try:
        feed_image = None
        rasterized_feeds = 0
        if rasterize_feeds:
            feed_image, rasterized_feeds = _rasterized_spec_edges(
                RenderSpec(edges=feed_edges),
                pixel_width=pixel_width,
                pixel_height=pixel_height,
                x_range=x_range,
                y_range=y_range,
            )
        edge_image = None
        rasterized_edges = 0
        if rasterize_network:
            raster_spec = RenderSpec(edges=network_edges)
            edge_image, rasterized_edges = _rasterized_spec_edges(
                raster_spec,
                pixel_width=pixel_width,
                pixel_height=pixel_height,
                x_range=x_range,
                y_range=y_range,
            )

        if feed_image is not None:
            interpolation = "nearest" if raster_scale == 1.0 else "bilinear"
            axis.imshow(feed_image, extent=(*x_range, *y_range), origin="upper", interpolation=interpolation, aspect="auto", zorder=1.8)
            rendered_edges += rasterized_feeds
        elif feed_edges:
            _draw_classic_edges(axis, feed_edges, directional=len(feed_edges) <= _EXPLICIT_EDGE_LIMIT, zorder=1.8)
            rendered_edges += len(feed_edges)

        if len(network_edges) <= _EXPLICIT_EDGE_LIMIT:
            _draw_classic_edges(axis, network_edges, directional=True, zorder=2.2)
            rendered_edges += len(network_edges)
        elif not rasterize_network:
            _draw_classic_edges(axis, network_edges, directional=False, zorder=2.2)
            rendered_edges += len(network_edges)
        elif edge_image is not None:
            interpolation = "nearest" if raster_scale == 1.0 else "bilinear"
            axis.imshow(edge_image, extent=(*x_range, *y_range), origin="upper", interpolation=interpolation, aspect="auto", zorder=2)
            rendered_edges += rasterized_edges

        if routing_edges:
            _draw_classic_edges(axis, routing_edges, directional=len(routing_edges) <= _EXPLICIT_EDGE_LIMIT, zorder=2.4)
            rendered_edges += len(routing_edges)
        if legend_edges:
            _draw_classic_edges(axis, legend_edges, directional=True, zorder=3.2)
            rendered_edges += len(legend_edges)

        edge_note = f"all {rendered_edges:,} scene edges included"
    except Exception as error:
        from versal.utils.logging import Logger

        Logger.get_logger().warning("hybrid Datashader edge layer failed: %s: %s", type(error).__name__, error)
        rendered_edges = min(len(spec.edges), _MAX_STRAIGHT_EDGES + _MAX_CURVED_EDGES)
        _draw_classic_edges(axis, spec.edges, directional=len(spec.edges) <= _EXPLICIT_EDGE_LIMIT)
        edge_note = f"classic fallback showing {rendered_edges:,} of {len(spec.edges):,} scene edges"
        axis.text(0.005, 0.005, "Datashader unavailable; classic fallback", transform=axis.transAxes, fontsize=6, color=THEME["label"], ha="left", va="bottom", zorder=5)

    try:
        node_image = _rasterized_spec_nodes(spec, pixel_width=pixel_width, pixel_height=pixel_height, x_range=x_range, y_range=y_range)
        if node_image is not None:
            interpolation = "nearest" if raster_scale == 1.0 else "bilinear"
            axis.imshow(node_image, extent=(*x_range, *y_range), origin="upper", interpolation=interpolation, aspect="auto", zorder=2.6)
    except Exception as error:
        from versal.utils.logging import Logger

        Logger.get_logger().warning("hybrid Datashader node layer failed: %s: %s", type(error).__name__, error)

    pixels_per_unit = min(fig_w * 72 / max(spec.width, 1e-6), fig_h * 72 / max(spec.height, 1e-6))
    base_area = min(max((0.5 * pixels_per_unit) ** 2, 16.0), 700.0)
    for marker in sorted({node.marker for node in spec.nodes}):
        group_nodes = [node for node in spec.nodes if node.marker == marker]
        rgba_nodes = [mcolors.to_rgba(node.color, node.alpha) for node in group_nodes]
        axis.scatter(
            [node.x for node in group_nodes],
            [node.y for node in group_nodes],
            s=[base_area * node.size for node in group_nodes],
            c=rgba_nodes,
            marker=marker,
            linewidths=0.0,
            zorder=3,
        )

    if show_footer and spec.flow_label:
        _draw_potential_flow_legend(axis, spec)
        axis.text(
            0.5,
            0.006,
            f"{spec.flow_label} · hybrid Datashader · {edge_note}",
            transform=axis.transAxes,
            fontsize=6.5,
            color=THEME["label"],
            ha="center",
            va="bottom",
            zorder=5,
        )

    axis.set_xlim(*x_range)
    axis.set_ylim(*y_range)
    axis.set_aspect("equal")
    axis.axis("off")
    if title is not None:
        axis.set_title(title, fontsize=11, color=THEME["title"])


def _render_figure_size(spec: RenderSpec, dpi: int) -> tuple[float, float]:
    """Aspect-aware figure inches with a hard final-canvas memory bound."""
    # Aspect-preserving sizing: clipping each side independently used to hand a tall-narrow spec a
    # square figure, and set_aspect("equal") filled the rest with dead background.
    scale = min(1.0, 30.0 / (0.6 * max(spec.width, spec.height, 1e-6)))
    fig_w = max(spec.width * 0.6 * scale, 4.0)
    fig_h = max(spec.height * 0.6 * scale, 4.0)
    pixel_count = fig_w * fig_h * dpi * dpi
    if pixel_count > _MAX_RENDER_PIXELS:
        pixel_scale = math.sqrt(_MAX_RENDER_PIXELS / pixel_count)
        fig_w *= pixel_scale
        fig_h *= pixel_scale
    return fig_w, fig_h


def _render_spec_png(out_path: Path, spec: RenderSpec, title: str, *, dpi: int = _RENDER_DPI, x_padding: float = _PAD) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_w, fig_h = _render_figure_size(spec, dpi)
    temporary = _temporary_sibling(out_path)
    figure, axis = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    try:
        figure.patch.set_facecolor(THEME["background"])
        draw_spec(axis, spec, title=title, x_padding=x_padding)
        figure.tight_layout()
        figure.savefig(temporary, dpi=dpi, facecolor=figure.get_facecolor())
        temporary.replace(out_path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return out_path


# large individual module portraits


def _accumulate_density_lines(canvas: Any, frame: Any, datashader: Any, numpy: Any, current: Any | None) -> Any:
    aggregate = canvas.line(frame, x=["x0", "x1"], y=["y0", "y1"], axis=1, agg=datashader.sum("weight"))
    values = numpy.nan_to_num(numpy.asarray(aggregate.data), nan=0.0).astype("float32", copy=False)
    if current is None:
        return aggregate.copy(data=values)
    current.data += values
    return current


def _render_large_module_density(out_path: Path, entry: LibraryEntry) -> _LargeRenderMetadata:
    from versal.rendering_density import _render_large_module_density as render_density

    return render_density(out_path, entry)


def _temporary_sibling(out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{out_path.stem}-", suffix=".tmp.png", dir=out_path.parent)
    os.close(descriptor)
    return Path(name)


def _render_large_module_png(out_path: Path, entry: LibraryEntry) -> _LargeRenderMetadata:
    temporary = _temporary_sibling(out_path)
    try:
        metadata = _render_large_module_density(temporary, entry)
        temporary.replace(out_path)
        return metadata
    except Exception as error:
        temporary.unlink(missing_ok=True)
        from versal.utils.logging import Logger

        reason = f"{type(error).__name__}: {error}"
        Logger.get_logger().warning("large module density render failed for %s: %s", entry.key, reason)
        fallback = _temporary_sibling(out_path)
        try:
            spec = build_entry_spec(entry, node_budget=0)
            _render_spec_png(fallback, spec, f"{entry.key}  L{entry.level} {entry.entry_type}\nDatashader portrait unavailable: {reason}")
            fallback.replace(out_path)
        finally:
            fallback.unlink(missing_ok=True)
        connections = entry.payload.get("connections", [])
        enabled_count = sum(bool(connection.get("enabled", False)) for connection in connections) if isinstance(connections, list) else 0
        return _LargeRenderMetadata(
            node_count=len(entry.payload.get("nodes", [])),
            enabled_edge_count=enabled_count,
            isolated_input_count=0,
            rendered_edge_count=0,
            semantic_layout_mode="opaque-fallback",
            fallback_reason=reason,
        )


def render_network(
    directory: Path,
    genome: Genome,
    *,
    title: str,
    library: ModuleLibrary | None = None,
    max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH,
) -> Path:
    """Render a flat genome (macros expand into callouts when a library is supplied) to `net.png`."""
    resolve = library_resolver(library) if library is not None else None
    return _render_spec_png(directory / "net.png", build_genome_spec(genome, resolve=resolve, max_inline_depth=max_inline_depth), title)


def render_composition_network(
    directory: Path,
    comp: CompositionGenome,
    *,
    title: str,
    library: ModuleLibrary | None = None,
    max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH,
) -> Path:
    """Render a composition (module refs expand into callouts when a library is supplied) to `net.png`."""
    resolve = library_resolver(library) if library is not None else None
    return _render_spec_png(directory / "net.png", build_composition_spec(comp, resolve=resolve, max_inline_depth=max_inline_depth), title)


def render_entry(
    out_path: Path,
    entry: LibraryEntry,
    *,
    library: ModuleLibrary | None = None,
    node_budget: int = DEFAULT_NODE_BUDGET,
    max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH,
) -> Path:
    if entry.entry_type == MODULE and len(entry.payload.get("nodes", [])) > node_budget:
        _render_large_module_png(out_path, entry)
        return out_path
    resolve = library_resolver(library) if library is not None else None
    spec = build_entry_spec(entry, resolve=resolve, node_budget=node_budget, max_inline_depth=max_inline_depth)
    return _render_spec_png(out_path, spec, f"{entry.key}  L{entry.level} {entry.entry_type}")


# gallery


def _cell_title(summary: dict[str, Any]) -> str:
    inputs = "+".join(str(item["width"]) for item in summary["io"]["inputs"])
    label = f"{summary['key']}  L{summary['level']} {summary['entry_type'][0]}  {inputs}->{summary['io']['output']['width']}  m={summary.get('accepted_metric', 0.0):.2f}"
    return f"{label}  retired" if summary.get("retired", False) else label


def render_library_gallery(
    library: ModuleLibrary,
    out_path: Path,
    *,
    columns: int = 4,
    include_retired: bool = False,
    include_dependencies: bool = True,
    max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH,
) -> Path:
    """One contact-sheet PNG of every (selected) library entry, drawn through the same spec/draw
    pipeline as the single renders. One bad entry must never kill the sheet."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = library.summaries(include_retired=include_retired, include_dependencies=include_dependencies)
    resolve = library_resolver(library)

    if not rows:
        figure, axis = plt.subplots(figsize=(6, 4))
        figure.patch.set_facecolor(THEME["background"])
        axis.set_facecolor(THEME["background"])
        axis.text(0.5, 0.5, "library is empty", ha="center", va="center", color=THEME["label"])
        axis.axis("off")
    else:
        columns = max(1, min(columns, len(rows)))
        grid_rows = math.ceil(len(rows) / columns)
        cell_w, cell_h = 4.2, 3.2
        scale = min(1.0, 60.0 / (grid_rows * cell_h))
        figure, axes = plt.subplots(grid_rows, columns, figsize=(columns * cell_w * scale, grid_rows * cell_h * scale), squeeze=False)
        figure.patch.set_facecolor(THEME["background"])
        flat_axes = [axis for row_axes in axes for axis in row_axes]
        for axis, summary in zip(flat_axes, rows):
            try:
                # A contact sheet is an index, not the full recursive portrait: one reference level
                # preserves each entry's architecture without reducing deep compositions to slivers.
                spec = build_entry_spec(
                    library.load(summary["key"]),
                    resolve=resolve,
                    node_budget=GALLERY_NODE_BUDGET,
                    max_inline_depth=min(max_inline_depth, 1),
                )
                draw_spec(axis, spec, show_footer=False)
            except Exception:
                axis.set_facecolor(THEME["background"])
                axis.text(0.5, 0.5, f"{summary['key']}\nrender failed", ha="center", va="center", color=THEME["label"], fontsize=7)
                axis.axis("off")
            axis.set_title(_cell_title(summary), fontsize=7, color=THEME["title"])
        for axis in flat_axes[len(rows) :]:
            axis.set_facecolor(THEME["background"])
            axis.axis("off")

    temporary = _temporary_sibling(out_path)
    try:
        figure.tight_layout()
        figure.savefig(temporary, dpi=150, facecolor=figure.get_facecolor())
        temporary.replace(out_path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return out_path


# motif atlas

# Hidden-node tints for the atlas only: a tanh-gated motif should read at a glance without text.
# Identity stays neutral so the interesting activations pop.
_ACTIVATION_TINTS = {"identity": "#8b93b5", "tanh": "#7dcfff", "relu": "#9ece6a", "sigmoid": "#bb9af7", "sin": "#e0af68", "gaussian": "#f7768e"}


def build_motif_spec(node_labels: tuple[NodeLabel, ...], edges: tuple[tuple[int, int, int], ...]) -> RenderSpec:
    """One canonical motif as a tiny labeled digraph, in the shared visual vocabulary. Layered by
    longest path over forward+macro edges; recurrent edges are time-delayed so they never layer."""
    spec = RenderSpec()
    size = len(node_labels)
    if size == 0:
        return spec
    forward = [(source, target) for source, target, mask in edges if mask & (FORWARD_EDGE | MACRO_EDGE) and source != target]
    layer = {index: 0 for index in range(size)}
    for _ in range(size):
        changed = False
        for source, target in forward:
            if layer[target] < layer[source] + 1:
                layer[target] = layer[source] + 1
                changed = True
        if not changed:
            break
    else:
        layer = {index: index for index in range(size)}  # a forward cycle cannot layer; index order will do

    items = [_Item(key=index, layer=layer[index], preds=[source for source, target in forward if target == index], sort_rank=(0, index)) for index in range(size)]
    positions, spec.width, spec.height = _place_items(items)

    for index, (kind, second, aggregation, stub) in enumerate(node_labels):
        x, y = positions[index]
        if stub or kind == "module":
            color, marker, role = THEME["node_module"], "h", "module-footprint"
        elif kind == "input":
            color, marker, role = (THEME["node_bias"] if second == "bias" else THEME["node_input"]), "s", "bias" if second == "bias" else "input"
        elif kind == "bias":
            color, marker, role = THEME["node_bias"], "s", "bias"
        elif kind == "output":
            color, marker, role = THEME["node_output"], "s", "output"
        else:
            color, marker, role = _ACTIVATION_TINTS.get(second, THEME["edge_forward"]), ("D" if aggregation == "product" else "o"), "hidden"
        spec.nodes.append(SpecNode(x, y, color, size=1.3, marker=marker, role=role))

    for source, target, mask in edges:
        if source == target:
            # A recurrent self-loop (the TRM refinement motif) draws as a small arc riding the node:
            # arc3 renders nothing when both endpoints coincide.
            x, y = positions[source]
            spec.edges.append(
                SpecEdge(
                    x - 0.2,
                    y + 0.18,
                    x + 0.2,
                    y + 0.18,
                    width=1.2,
                    color=THEME["edge_recurrent"],
                    style="dashed",
                    curve=1.6,
                    alpha=0.8,
                    role="recurrent",
                )
            )
            continue
        (x0, y0), (x1, y1) = positions[source], positions[target]
        if mask & FORWARD_EDGE:
            spec.edges.append(SpecEdge(x0, y0, x1, y1, width=1.4, color=THEME["edge_positive"], alpha=0.7, role="forward-positive"))
        if mask & MACRO_EDGE:
            spec.edges.append(SpecEdge(x0, y0, x1, y1, width=1.2, color=THEME["edge_macro"], alpha=0.6, role="macro-implied"))
        if mask & RECURRENT_EDGE:
            spec.edges.append(SpecEdge(x0, y0, x1, y1, width=1.2, color=THEME["edge_recurrent"], style="dashed", curve=0.25, alpha=0.7, role="recurrent"))
    return spec


def render_motif_atlas(out_path: Path, motifs: list[MotifRecord], *, columns: int = 6, empty_note: str | None = None) -> Path:
    """A contact sheet of recurring motifs (per-cell axes, like the library gallery: the single-spec
    figure floor makes merged grids awkward). One bad motif must never kill the sheet. `empty_note`
    prints beneath the empty-state headline so a reader knows WHY the atlas is bare (small library,
    min_support too high), rather than assuming the search found nothing."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not motifs:
        figure, axis = plt.subplots(figsize=(6, 4))
        figure.patch.set_facecolor(THEME["background"])
        axis.set_facecolor(THEME["background"])
        axis.text(0.5, 0.6, "no recurring motifs yet", ha="center", va="center", color=THEME["label"])
        if empty_note:
            wrapped_note = "\n".join(textwrap.wrap(empty_note, width=64))
            axis.text(0.5, 0.32, wrapped_note, ha="center", va="center", color=THEME["label"], fontsize=7)
        axis.axis("off")
    else:
        columns = max(1, min(columns, len(motifs)))
        grid_rows = math.ceil(len(motifs) / columns)
        figure, axes = plt.subplots(grid_rows, columns, figsize=(columns * 2.6, grid_rows * 2.4), squeeze=False)
        figure.patch.set_facecolor(THEME["background"])
        flat_axes = [axis for row_axes in axes for axis in row_axes]
        for axis, record in zip(flat_axes, motifs):
            try:
                draw_spec(axis, build_motif_spec(record.graph.node_labels, record.graph.edges), show_footer=False)
            except Exception:
                axis.set_facecolor(THEME["background"])
                axis.text(0.5, 0.5, f"{record.fingerprint}\nrender failed", ha="center", va="center", color=THEME["label"], fontsize=7)
                axis.axis("off")
            axis.set_title(f"{record.diversity_class}  s={record.support} n={record.occurrences}", fontsize=7, color=THEME["title"])
        for axis in flat_axes[len(motifs) :]:
            axis.set_facecolor(THEME["background"])
            axis.axis("off")

    temporary = _temporary_sibling(out_path)
    try:
        figure.tight_layout()
        figure.savefig(temporary, dpi=150, facecolor=figure.get_facecolor())
        temporary.replace(out_path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return out_path


# overmind: the whole routed model in one frame


@dataclass(slots=True)
class OvermindVertex:
    """Read-only routed-expert data used by the renderer."""

    key: str
    label: str
    retired: bool = False
    usage: float = 0.0
    entry_share: float = 0.0
    exit_share: float = 0.0
    mean_step: float | None = None
    embedding_rank: int = 0
    stepping_stone: bool = False


@dataclass(slots=True)
class OvermindView:
    """Plain routed-model data required for an overmind render."""

    vertices: list[OvermindVertex]
    input_signatures: list[str]
    output_signatures: list[str]
    d_model: int
    top_k: int
    max_steps: int
    pathways: list[tuple[int, int, float]] = field(default_factory=list)
    traffic_observed: bool = True


def prune_overmind_view(view: OvermindView) -> OvermindView:
    from versal.rendering_overmind import prune_overmind_view as prune

    return prune(view)


_OVERMIND_COLUMNS = 8
_OVERMIND_DPI = 300
_OVERMIND_X_PADDING = 2 * _PAD


def build_overmind_spec(
    view: OvermindView,
    *,
    resolve: ResolveFn | None = None,
    node_budget: int = DEFAULT_NODE_BUDGET,
    cell_node_budget: int = 160,
    columns: int = _OVERMIND_COLUMNS,
    legend: bool = True,
    legend_mode: str = "full",
    max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH,
) -> RenderSpec:
    from versal.rendering_overmind import build_overmind_spec as build

    return build(
        view,
        resolve=resolve,
        node_budget=node_budget,
        cell_node_budget=cell_node_budget,
        columns=columns,
        legend=legend,
        legend_mode=legend_mode,
        max_inline_depth=max_inline_depth,
    )


def render_overmind(
    out_path: Path,
    view: OvermindView,
    *,
    library: ModuleLibrary | None = None,
    node_budget: int = DEFAULT_NODE_BUDGET,
    max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH,
) -> Path:
    """Render full and current-only portraits of the routed model.

    Writing ``overmind.png`` also writes ``overmind_pruned.png``.  The full portrait retains
    retired experts as opaque historical cards; the pruned portrait removes them and compacts the
    survivors into the same eight-column grid.
    """
    resolve = library_resolver(library) if library is not None else None
    spec = build_overmind_spec(view, resolve=resolve, node_budget=node_budget, max_inline_depth=max_inline_depth)
    live = sum(1 for vertex in view.vertices if not vertex.retired)
    total = len(view.vertices)
    title = f"overmind history: {live} current / {total} total, d_model={view.d_model}, top_k={view.top_k}, steps={view.max_steps}"
    rendered = _render_spec_png(out_path, spec, title, dpi=_OVERMIND_DPI, x_padding=_OVERMIND_X_PADDING)

    if not out_path.stem.endswith("_pruned"):
        pruned_path = out_path.with_name(f"{out_path.stem}_pruned{out_path.suffix}")
        pruned = prune_overmind_view(view)
        pruned_spec = build_overmind_spec(pruned, resolve=resolve, node_budget=node_budget, legend_mode="adaptive", max_inline_depth=max_inline_depth)
        pruned_title = f"overmind current: {len(pruned.vertices)} experts, d_model={view.d_model}, top_k={view.top_k}, steps={view.max_steps}"
        _render_spec_png(pruned_path, pruned_spec, pruned_title, dpi=_OVERMIND_DPI, x_padding=_OVERMIND_X_PADDING)
    return rendered
