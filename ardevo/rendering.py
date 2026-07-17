"""Artistic recursive renders of evolved networks: the full nested detail, dark, in one image.

Nested networks render as CALLOUTS: the host network keeps its natural compact layout (a macro's
output stubs and a composition's module nodes stay in place as green footprint nodes), and each
referenced network is drawn fully inside a translucent container packed into rows across the top of
the frame. A green line runs from the footprint node to a gold input anchor at the callout's
top-left. Callouts recurse (depth- and budget-guarded): a nested network's own callouts ride along
inside its box.

These renders are an artistic overview: the library JSON stays the ground truth, so every failure
mode (missing ref, cycle, over budget, undeserializable payload) degrades to a labeled opaque box
instead of raising. A render must never kill a run or a gallery.

The build/draw split keeps layout pure: builders produce a `RenderSpec` (flat primitive lists in one
shared coordinate frame; children are translated and merged into the parent at placement, never
scaled), and `draw_spec` paints any spec onto any matplotlib axis, which is what lets the gallery
reuse the exact same pipeline per cell. matplotlib is imported lazily inside the draw/render
functions (and forced onto the headless Agg backend) per project convention.
"""

import heapq
import math
import os
import tempfile
import textwrap
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, cast

from ardevo.evolution.composition import CompNodeKind, CompositionGenome, comp_from_dict, comp_topological_order
from ardevo.evolution.genome import Genome, NodeKind, genome_from_dict, macro_implied_edges, make_acyclic, topological_order
from ardevo.library import COMPOSITION, MODULE, LibraryEntry, ModuleLibrary
from ardevo.motifs import FORWARD_EDGE, MACRO_EDGE, RECURRENT_EDGE, MotifRecord, NodeLabel
from ardevo.reference_depth import DEFAULT_MAX_INLINE_DEPTH

# --- async rendering ------------------------------------------------------------------------------
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
            from ardevo.utils.logging import Logger

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
_MAX_STRAIGHT_EDGES = 60_000  # LineCollection cap; beyond it edges stride-subsample with a note
_MAX_CURVED_EDGES = 2_000  # FancyArrowPatch is one artist per edge, so its cap is much lower
_DENSITY_WIDTH = 2400
_DENSITY_HEIGHT = 1600
_DENSITY_EDGE_CHUNK = 250_000

ResolveFn = Callable[[str], LibraryEntry | None]


@dataclass(slots=True)
class SpecNode:
    x: float
    y: float
    color: str
    size: float = 1.0  # relative multiplier; draw_spec converts to point area from pixel density
    marker: str = "o"
    alpha: float = 1.0


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


# --- internal recursion ----------------------------------------------------------------------------


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
            spec.nodes.append(SpecNode(target_x, target_y, color=THEME["node_anchor"], size=0.65))
            for source_x, source_y in source_points:
                spec.edges.append(SpecEdge(source_x, source_y, target_x, target_y, width=1.0, color=THEME["edge_callout"], alpha=0.55))
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
            color, marker, alpha = THEME["node_module"], "h", 1.0
        elif node.kind is NodeKind.INPUT:
            color, marker, alpha = THEME["node_input"], "s", 1.0
        elif node.kind is NodeKind.BIAS:
            color, marker, alpha = THEME["node_bias"], "s", 0.7
        elif node.kind is NodeKind.OUTPUT:
            color, marker, alpha = THEME["node_output"], "s", 1.0
        else:
            color, marker, alpha = _layer_color(layer.get(node.id, 0), drawn_max_layer), ("D" if node.aggregation == "product" else "o"), 1.0
        size = 0.5 if isolated else min(1.0 + 0.15 * node_degree, 2.5)
        drawn = SpecNode(x, y, color, size=size, marker=marker, alpha=0.25 if isolated else alpha)
        spec.nodes.append(drawn)
        if node.kind is NodeKind.OUTPUT:
            output_nodes.append(drawn)

    for conn in genome.enabled_connections():
        source, target = positions.get(conn.in_id), positions.get(conn.out_id)
        if source is None or target is None:
            continue
        if conn.recurrent:
            spec.edges.append(
                SpecEdge(source[0], source[1], target[0], target[1], width=_edge_width(conn.weight), color=THEME["edge_recurrent"], style="dashed", curve=0.25, alpha=0.6)
            )
        else:
            spec.edges.append(SpecEdge(source[0], source[1], target[0], target[1], width=_edge_width(conn.weight), color=THEME["edge_forward"], alpha=0.35))
    for source, target in macro_implied_edges(genome):
        source_pos, target_pos = positions.get(source), positions.get(target)
        if source_pos is None or target_pos is None:
            continue
        spec.edges.append(SpecEdge(source_pos[0], source_pos[1], target_pos[0], target_pos[1], width=1.0, color=THEME["edge_macro"], alpha=0.4))

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
        elif node.kind is CompNodeKind.INPUT:
            color = THEME["node_bias"] if node.ref == "__bias__" else THEME["node_input"]
            marker = "s"
            size = 1.0 + math.log2(1 + node.out_width) / 4
        else:
            color = THEME["node_output"]
            marker = "s"
            size = 1.0 + math.log2(1 + node.in_width) / 4
        drawn = SpecNode(x, y, color, size=size, marker=marker)
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
        spec.edges.append(SpecEdge(source[0], source[1], target[0], target[1], width=_edge_width(strength), color=THEME["edge_glue"], alpha=0.5))

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


# --- public builders -------------------------------------------------------------------------------


def build_genome_spec(
    genome: Genome,
    *,
    resolve: ResolveFn | None = None,
    node_budget: int = DEFAULT_NODE_BUDGET,
    max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH,
) -> RenderSpec:
    return _build_genome(
        genome,
        resolve=resolve,
        budget=_Budget(node_budget),
        depth=0,
        reference_depth=0,
        stack=(),
        max_inline_depth=max_inline_depth,
    ).spec


def build_composition_spec(
    comp: CompositionGenome,
    *,
    resolve: ResolveFn | None = None,
    node_budget: int = DEFAULT_NODE_BUDGET,
    max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH,
) -> RenderSpec:
    return _build_comp(
        comp,
        resolve=resolve,
        budget=_Budget(node_budget),
        depth=0,
        reference_depth=0,
        stack=(),
        max_inline_depth=max_inline_depth,
    ).spec


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
        return built.spec
    except Exception:
        pass
    built = _opaque_built(f"{entry.key}  ?")
    built.spec.containers.append(SpecContainer(0.0, 0.0, built.spec.width, built.spec.height, label=built.label, depth=0, opaque=True))
    return built.spec


# --- painting --------------------------------------------------------------------------------------


def draw_spec(axis: Any, spec: RenderSpec, *, title: str | None = None, x_padding: float = _PAD) -> None:
    """Paint a spec onto a matplotlib axis (single renders and gallery cells share this path)."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colors as mcolors
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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

    # Cap what actually reaches matplotlib: a dense wide-I/O net can carry millions of edges, which
    # stalls the LineCollection (and one FancyArrowPatch PER curved edge is far worse). A stride
    # subsample keeps the drawing deterministic and the density picture honest via the note.
    straight = [edge for edge in spec.edges if edge.curve == 0.0]
    curved = [edge for edge in spec.edges if edge.curve != 0.0]
    dropped_notes = []
    if len(straight) > _MAX_STRAIGHT_EDGES:
        stride = math.ceil(len(straight) / _MAX_STRAIGHT_EDGES)
        dropped_notes.append(f"showing {len(straight[::stride]):,} of {len(straight):,} edges")
        straight = straight[::stride]
    if len(curved) > _MAX_CURVED_EDGES:
        stride = math.ceil(len(curved) / _MAX_CURVED_EDGES)
        dropped_notes.append(f"showing {len(curved[::stride]):,} of {len(curved):,} recurrent edges")
        curved = curved[::stride]
    if dropped_notes:
        axis.text(0.005, 0.005, "  |  ".join(dropped_notes), transform=axis.transAxes, fontsize=7, color=THEME["label"], ha="left", va="bottom", zorder=4)

    for style in ("solid", "dashed"):
        group = [edge for edge in straight if edge.style == style]
        if group:
            segments = [((edge.x0, edge.y0), (edge.x1, edge.y1)) for edge in group]
            rgba = [mcolors.to_rgba(edge.color, edge.alpha) for edge in group]
            axis.add_collection(LineCollection(segments, colors=rgba, linewidths=[edge.width for edge in group], linestyle=style, zorder=2))
    for edge in curved:
        if edge.curve != 0.0:
            arc = FancyArrowPatch(
                (edge.x0, edge.y0),
                (edge.x1, edge.y1),
                connectionstyle=f"arc3,rad={edge.curve}",
                color=mcolors.to_rgba(edge.color, edge.alpha),
                linewidth=edge.width,
                linestyle=edge.style,
                arrowstyle="-",
                zorder=2,
            )
            axis.add_patch(arc)

    figure = axis.figure
    fig_w, fig_h = figure.get_size_inches()
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

    axis.set_xlim(-x_padding, spec.width + x_padding)
    axis.set_ylim(-_PAD, spec.height + _PAD)
    axis.set_aspect("equal")
    axis.axis("off")
    if title is not None:
        axis.set_title(title, fontsize=11, color=THEME["title"])


def _render_spec_png(out_path: Path, spec: RenderSpec, title: str, *, dpi: int = 150, x_padding: float = _PAD) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Aspect-preserving sizing: clipping each side independently used to hand a tall-narrow spec a
    # square figure, and set_aspect("equal") filled the rest with dead background.
    scale = min(1.0, 30.0 / (0.6 * max(spec.width, spec.height, 1e-6)))
    fig_w = max(spec.width * 0.6 * scale, 4.0)
    fig_h = max(spec.height * 0.6 * scale, 4.0)
    figure, axis = plt.subplots(figsize=(fig_w, fig_h))
    figure.patch.set_facecolor(THEME["background"])
    draw_spec(axis, spec, title=title, x_padding=x_padding)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi, facecolor=figure.get_facecolor())
    plt.close(figure)
    return out_path


# --- large individual module portraits ------------------------------------------------------------


def _signature_axes(signature: object) -> tuple[str, ...]:
    if not isinstance(signature, str) or "|" not in signature:
        return ()
    _value_type, encoded_axes = signature.split("|", 1)
    return tuple(axis.strip() for axis in encoded_axes.split(",") if axis.strip())


def _format_axis_value(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _packed_bank_positions(
    bank: list[dict[str, Any]],
    positions: dict[int, tuple[float, float]],
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> None:
    if not bank:
        return
    aspect = max((x1 - x0) / max(y1 - y0, 1e-6), 0.1)
    columns = max(1, math.ceil(math.sqrt(len(bank) * aspect)))
    rows = math.ceil(len(bank) / columns)
    x_span = max(x1 - x0, 1e-6)
    y_span = max(y1 - y0, 1e-6)
    for index, node in enumerate(bank):
        row, column = divmod(index, columns)
        x = x0 + (column + 0.5) * x_span / columns
        y = y1 - (row + 0.5) * y_span / rows
        positions[int(node["id"])] = (x, y)


def _semantic_bank_positions(
    bank: list[dict[str, Any]],
    axes: tuple[str, ...],
    signature: str,
    positions: dict[int, tuple[float, float]],
    panels: list[_DensityPanel],
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> str | None:
    if "H" not in axes or "W" not in axes or axes.count("H") != 1 or axes.count("W") != 1:
        return f"{signature or '(unknown)'} has no unique H/W axes"
    if not bank:
        return None

    h_index, w_index = axes.index("H"), axes.index("W")
    coordinate_values: list[set[float]] = [set() for _ in axes]
    for node in bank:
        raw = node.get("coordinate")
        if not isinstance(raw, (list, tuple)) or len(raw) != len(axes):
            return f"{signature} coordinates do not match its axes"
        try:
            coordinate = tuple(float(value) for value in raw)
        except (TypeError, ValueError):
            return f"{signature} coordinates are not numeric"
        if not all(math.isfinite(value) for value in coordinate):
            return f"{signature} coordinates are not finite"
        for index, value in enumerate(coordinate):
            coordinate_values[index].add(value)

    # Detect malformed duplicates without retaining another set of large coordinate tuples. Axis
    # ranks turn each tuple into one integer; sparse but valid spatial banks remain valid.
    sorted_values = [sorted(values) for values in coordinate_values]
    ranks = [{value: rank for rank, value in enumerate(values)} for values in sorted_values]
    multipliers: list[int] = [1] * len(axes)
    for index in range(len(axes) - 2, -1, -1):
        multipliers[index] = multipliers[index + 1] * max(len(sorted_values[index + 1]), 1)
    flattened_ids: set[int] = set()
    combinations_set: set[tuple[float, ...]] = set()
    panel_axis_indices = tuple(index for index in range(len(axes)) if index not in (h_index, w_index))
    for node in bank:
        coordinate = tuple(float(value) for value in node["coordinate"])
        flattened_ids.add(sum(ranks[index][value] * multipliers[index] for index, value in enumerate(coordinate)))
        combinations_set.add(tuple(coordinate[index] for index in panel_axis_indices))
    if len(flattened_ids) != len(bank):
        return f"{signature} contains duplicate coordinates"

    combinations = sorted(combinations_set) or [()]
    columns = max(1, math.ceil(math.sqrt(len(combinations))))
    rows = math.ceil(len(combinations) / columns)
    horizontal_gap = 0.008
    vertical_gap = 0.018
    header = min(0.025, max((y1 - y0) * 0.12, 0.012))
    panel_width = (x1 - x0 - horizontal_gap * (columns - 1)) / columns
    panel_height = (y1 - y0 - header - vertical_gap * (rows - 1)) / rows
    panel_rects: dict[tuple[float, ...], _DensityPanel] = {}
    for index, combination in enumerate(combinations):
        row, column = divmod(index, columns)
        panel_x0 = x0 + column * (panel_width + horizontal_gap)
        panel_y1 = y1 - header - row * (panel_height + vertical_gap)
        panel_y0 = panel_y1 - panel_height
        labels = [f"{axes[axis_index]}={_format_axis_value(value)}" for axis_index, value in zip(panel_axis_indices, combination)]
        label = " · ".join(labels) if labels else "H×W"
        panel = _DensityPanel(panel_x0, panel_y0, panel_x0 + panel_width, panel_y1, label)
        panels.append(panel)
        panel_rects[combination] = panel

    h_values, w_values = sorted_values[h_index], sorted_values[w_index]
    h_min, h_span = h_values[0], max(h_values[-1] - h_values[0], 1.0)
    w_min, w_span = w_values[0], max(w_values[-1] - w_values[0], 1.0)
    for node in bank:
        coordinate = tuple(float(value) for value in node["coordinate"])
        combination = tuple(coordinate[index] for index in panel_axis_indices)
        panel = panel_rects[combination]
        panel_x0, panel_y0, panel_x1, panel_y1 = panel.x0, panel.y0, panel.x1, panel.y1
        inset_x = min((panel_x1 - panel_x0) * 0.04, 0.006)
        inset_y = min((panel_y1 - panel_y0) * 0.04, 0.006)
        x = panel_x0 + inset_x + (coordinate[w_index] - w_min) / w_span * max(panel_x1 - panel_x0 - 2 * inset_x, 1e-6)
        # Raw row zero belongs at the visual top, regardless of where H occurs in the signature.
        y = panel_y1 - inset_y - (coordinate[h_index] - h_min) / h_span * max(panel_y1 - panel_y0 - 2 * inset_y, 1e-6)
        node_id = int(node["id"])
        positions[node_id] = (x, y)
        panel.node_ids.append(node_id)
    return None


def _macro_implied_payload_edges(payload: dict[str, Any]) -> Iterator[tuple[int, int]]:
    for macro in payload.get("macros", []):
        inputs = [int(node_id) for node_id in macro.get("inputs", [])]
        outputs = [int(node_id) for node_id in macro.get("outputs", [])]
        for source in inputs:
            for target in outputs:
                yield source, target


def _computed_node_positions(
    nodes: list[dict[str, Any]],
    connections: list[dict[str, Any]],
    macro_edges: Iterable[tuple[int, int]],
    positions: dict[int, tuple[float, float]],
) -> tuple[list[int], str | None]:
    computed = sorted((node for node in nodes if node.get("kind") != NodeKind.INPUT.value), key=lambda node: int(node["id"]))
    computed_ids = [int(node["id"]) for node in computed]
    computed_set = set(computed_ids)
    adjacency: dict[int, list[int]] = {}
    incoming = {node_id: 0 for node_id in computed_ids}

    def add_candidate(source: int, target: int) -> None:
        if source in computed_set and target in computed_set:
            adjacency.setdefault(source, []).append(target)
            incoming[target] += 1

    for connection in connections:
        if bool(connection.get("enabled", False)) and not bool(connection.get("recurrent", False)):
            add_candidate(int(connection["in"]), int(connection["out"]))
    for source, target in macro_edges:
        add_candidate(source, target)

    ready = [node_id for node_id, count in incoming.items() if count == 0]
    heapq.heapify(ready)
    layer = {node_id: 1 for node_id in ready}
    visited: list[int] = []
    while ready:
        source = heapq.heappop(ready)
        visited.append(source)
        for target in sorted(adjacency.get(source, [])):
            layer[target] = max(layer.get(target, 1), layer.get(source, 1) + 1)
            incoming[target] -= 1
            if incoming[target] == 0:
                heapq.heappush(ready, target)
    fallback_reason = None
    if len(visited) != len(computed_ids):
        fallback_reason = "forward computed-node cycle; cyclic nodes packed in a final column"
        last_layer = max(layer.values(), default=0) + 1
        for node_id in computed_ids:
            layer.setdefault(node_id, last_layer)

    grouped: dict[int, list[int]] = {}
    for node_id in computed_ids:
        grouped.setdefault(layer.get(node_id, 1), []).append(node_id)
    ordered_layers = sorted(grouped)
    slot_width = 0.22 / max(len(ordered_layers), 1)
    for column_index, layer_id in enumerate(ordered_layers):
        group = grouped[layer_id]
        sub_columns = max(1, math.ceil(len(group) / 64))
        rows = math.ceil(len(group) / sub_columns)
        slot_x0 = 0.75 + column_index * slot_width
        for index, node_id in enumerate(group):
            sub_column, row = divmod(index, rows)
            x = slot_x0 + (sub_column + 0.5) * slot_width / sub_columns
            y = 0.87 - (row + 0.5) * 0.72 / rows
            positions[node_id] = (x, y)
    return computed_ids, fallback_reason


def _density_layout(entry: LibraryEntry) -> _DensityLayout:
    raw_nodes_value = entry.payload.get("nodes", [])
    raw_connections_value = entry.payload.get("connections", [])
    if not isinstance(raw_nodes_value, list) or not isinstance(raw_connections_value, list):
        raise ValueError("module payload nodes/connections must be lists")
    raw_nodes = cast(list[dict[str, Any]], raw_nodes_value)
    raw_connections = cast(list[dict[str, Any]], raw_connections_value)
    nodes = sorted(raw_nodes, key=lambda node: int(node["id"]))
    seen_node_ids: set[int] = set()
    for node in nodes:
        node_id = int(node["id"])
        if node_id in seen_node_ids:
            raise ValueError("module payload contains duplicate node ids")
        seen_node_ids.add(node_id)
    del seen_node_ids
    input_nodes = [node for node in nodes if node.get("kind") == NodeKind.INPUT.value]
    positions: dict[int, tuple[float, float]] = {}
    panels: list[_DensityPanel] = []
    fallback_reasons: list[str] = []

    descriptors = entry.io.get("inputs", []) if isinstance(entry.io, dict) else []
    if not isinstance(descriptors, list) or not descriptors:
        descriptors = [{"signature": "(untyped)", "width": len(input_nodes)}]
    banks: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    cursor = 0
    for descriptor in descriptors:
        try:
            width = max(0, int(descriptor.get("width", 0)))
        except (AttributeError, TypeError, ValueError):
            width = 0
        bank = input_nodes[cursor : cursor + width]
        banks.append((descriptor if isinstance(descriptor, dict) else {}, bank))
        cursor += width
    if cursor != len(input_nodes):
        fallback_reasons.append(f"I/O widths describe {cursor:,} of {len(input_nodes):,} inputs")
        if cursor < len(input_nodes):
            banks.append(({"signature": "(unmapped)", "width": len(input_nodes) - cursor}, input_nodes[cursor:]))

    spatial_banks = 0
    packed_banks = 0
    bank_count = max(len(banks), 1)
    available_height = 0.74
    bank_gap = min(0.025, available_height / max(bank_count * 8, 1))
    bank_height = (available_height - bank_gap * (bank_count - 1)) / bank_count
    for bank_index, (descriptor, bank) in enumerate(banks):
        bank_y1 = 0.90 - bank_index * (bank_height + bank_gap)
        bank_y0 = bank_y1 - bank_height
        signature = str(descriptor.get("signature", "(unknown)"))
        axes = _signature_axes(signature)
        reason = _semantic_bank_positions(bank, axes, signature, positions, panels, x0=0.04, y0=bank_y0, x1=0.69, y1=bank_y1)
        if reason is None and "H" in axes and "W" in axes:
            spatial_banks += 1
        else:
            packed_banks += 1
            if reason is not None:
                fallback_reasons.append(reason)
            _packed_bank_positions(bank, positions, x0=0.04, y0=bank_y0, x1=0.69, y1=bank_y1 - 0.02)
            panels.append(_DensityPanel(0.04, bank_y0, 0.69, bank_y1 - 0.02, signature, [int(node["id"]) for node in bank]))

    macro_edges = _macro_implied_payload_edges(entry.payload)
    computed_ids, computed_reason = _computed_node_positions(nodes, raw_connections, macro_edges, positions)
    if computed_reason:
        fallback_reasons.append(computed_reason)
    mode = "semantic-spatial" if spatial_banks and not packed_banks else ("mixed semantic/packed" if spatial_banks else "packed-grid")
    return _DensityLayout(
        positions=positions,
        input_ids=[int(node["id"]) for node in input_nodes],
        computed_ids=computed_ids,
        panels=panels,
        mode=mode,
        fallback_reason="; ".join(fallback_reasons) or None,
    )


def _accumulate_density_lines(canvas: Any, frame: Any, datashader: Any, numpy: Any, current: Any | None) -> Any:
    aggregate = canvas.line(frame, x=["x0", "x1"], y=["y0", "y1"], axis=1, agg=datashader.sum("weight"))
    values = numpy.nan_to_num(numpy.asarray(aggregate.data), nan=0.0).astype("float32", copy=False)
    if current is None:
        return aggregate.copy(data=values)
    current.data += values
    return current


def _rasterized_edge_layers(
    canvas: Any,
    positions: dict[int, tuple[float, float]],
    connections: list[dict[str, Any]],
    macro_edges: Iterable[tuple[int, int]],
    datashader: Any,
    pandas: Any,
    numpy: Any,
) -> tuple[dict[str, Any], int, int]:
    layers: dict[str, Any] = {"positive": None, "negative": None, "recurrent": None, "macro": None}
    enabled_count = 0
    rendered_count = 0

    def consume(rows: list[tuple[float, float, float, float, float, str]]) -> None:
        grouped: dict[str, list[tuple[float, float, float, float, float]]] = {}
        for x0, y0, x1, y1, weight, category in rows:
            grouped.setdefault(category, []).append((x0, y0, x1, y1, weight))
        for category, category_rows in grouped.items():
            frame = pandas.DataFrame(category_rows, columns=["x0", "y0", "x1", "y1", "weight"])
            layers[category] = _accumulate_density_lines(canvas, frame, datashader, numpy, layers[category])

    chunk: list[tuple[float, float, float, float, float, str]] = []
    for connection in connections:
        if not bool(connection.get("enabled", False)):
            continue
        enabled_count += 1
        source_id, target_id = int(connection["in"]), int(connection["out"])
        if source_id not in positions or target_id not in positions:
            raise ValueError(f"enabled edge {source_id}->{target_id} names a missing node")
        source, target = positions[source_id], positions[target_id]
        weight = float(connection.get("weight", 0.0))
        if not math.isfinite(weight):
            raise ValueError(f"enabled edge {source_id}->{target_id} has a non-finite weight")
        category = "recurrent" if bool(connection.get("recurrent", False)) else ("positive" if weight >= 0 else "negative")
        chunk.append((source[0], source[1], target[0], target[1], abs(weight), category))
        rendered_count += 1
        if len(chunk) == _DENSITY_EDGE_CHUNK:
            consume(chunk)
            chunk = []
    if chunk:
        consume(chunk)

    macro_chunk: list[tuple[float, float, float, float, float, str]] = []
    for source_id, target_id in macro_edges:
        if source_id not in positions or target_id not in positions:
            raise ValueError(f"macro-implied edge {source_id}->{target_id} names a missing node")
        source, target = positions[source_id], positions[target_id]
        macro_chunk.append((source[0], source[1], target[0], target[1], 1.0, "macro"))
        if len(macro_chunk) == _DENSITY_EDGE_CHUNK:
            consume(macro_chunk)
            macro_chunk = []
    if macro_chunk:
        consume(macro_chunk)
    return layers, enabled_count, rendered_count


def _panel_influence_aggregate(
    panel: _DensityPanel,
    positions: dict[int, tuple[float, float]],
    magnitude: dict[int, float],
    signed: dict[int, float],
    datashader: Any,
    pandas: Any,
    numpy: Any,
) -> tuple[Any, Any]:
    """Rasterize one semantic panel at its native coordinate resolution, so grid cells fill the
    panel instead of appearing as widely separated point markers."""
    if not panel.node_ids:
        return numpy.zeros((1, 1), dtype=numpy.float32), numpy.zeros((1, 1), dtype=numpy.float32)
    x_values = sorted({positions[node_id][0] for node_id in panel.node_ids})
    y_values = sorted({positions[node_id][1] for node_id in panel.node_ids})

    def bounds(values: list[float]) -> tuple[float, float]:
        if len(values) == 1:
            return values[0] - 0.5, values[0] + 0.5
        low_step = max(values[1] - values[0], 1e-9)
        high_step = max(values[-1] - values[-2], 1e-9)
        return values[0] - low_step / 2, values[-1] + high_step / 2

    frame = pandas.DataFrame(
        {
            "x": numpy.fromiter((positions[node_id][0] for node_id in panel.node_ids), dtype=numpy.float64, count=len(panel.node_ids)),
            "y": numpy.fromiter((positions[node_id][1] for node_id in panel.node_ids), dtype=numpy.float64, count=len(panel.node_ids)),
            "magnitude": numpy.fromiter((magnitude.get(node_id, 0.0) for node_id in panel.node_ids), dtype=numpy.float64, count=len(panel.node_ids)),
            "signed": numpy.fromiter((signed.get(node_id, 0.0) for node_id in panel.node_ids), dtype=numpy.float64, count=len(panel.node_ids)),
        }
    )
    canvas = datashader.Canvas(plot_width=max(len(x_values), 1), plot_height=max(len(y_values), 1), x_range=bounds(x_values), y_range=bounds(y_values))
    magnitude_aggregate = canvas.points(frame, "x", "y", agg=datashader.sum("magnitude"))
    signed_aggregate = canvas.points(frame, "x", "y", agg=datashader.sum("signed"))
    return (
        numpy.nan_to_num(numpy.asarray(magnitude_aggregate.data), nan=0.0).astype(numpy.float32, copy=False),
        numpy.nan_to_num(numpy.asarray(signed_aggregate.data), nan=0.0).astype(numpy.float32, copy=False),
    )


def _signed_influence_rgba(magnitude: Any, signed: Any, span: float, numpy: Any, colors: Any) -> Any:
    """Bivariate field: brightness is accumulated |weight| and hue is signed balance."""
    safe_span = max(float(span), 1e-12)
    level = numpy.clip(numpy.log1p(magnitude) / math.log1p(safe_span), 0.0, 1.0)
    balance = numpy.divide(signed, magnitude, out=numpy.zeros_like(signed, dtype=numpy.float32), where=magnitude > 0)
    balance = numpy.clip(balance, -1.0, 1.0)
    positive = numpy.asarray(colors.to_rgb(THEME["edge_glue"]), dtype=numpy.float32)
    negative = numpy.asarray(colors.to_rgb(THEME["edge_exit"]), dtype=numpy.float32)
    neutral = numpy.asarray(colors.to_rgb("#7d86a3"), dtype=numpy.float32)
    background = numpy.asarray(colors.to_rgb("#172033"), dtype=numpy.float32)
    positive_mix = numpy.clip(balance, 0.0, 1.0)[..., None]
    negative_mix = numpy.clip(-balance, 0.0, 1.0)[..., None]
    hue = neutral + positive_mix * (positive - neutral) + negative_mix * (negative - neutral)
    visibility = (0.12 + 0.88 * level)[..., None]
    rgb = background * (1.0 - visibility) + hue * visibility
    alpha = numpy.ones((*magnitude.shape, 1), dtype=numpy.float32)
    return numpy.concatenate((numpy.clip(rgb, 0.0, 1.0), alpha), axis=2)


def _flow_color(stats: _FlowStats, colors: Any) -> tuple[float, float, float, float]:
    if stats.magnitude <= 0:
        return colors.to_rgba(THEME["label"], 0.35)
    balance = max(-1.0, min(1.0, stats.signed / stats.magnitude))
    neutral = colors.to_rgb("#7d86a3")
    target = colors.to_rgb(THEME["edge_glue"] if balance >= 0 else THEME["edge_exit"])
    amount = abs(balance)
    return (*(neutral[index] + amount * (target[index] - neutral[index]) for index in range(3)), 0.72)


def _flow_width(stats: _FlowStats, peak: float) -> float:
    if stats.magnitude <= 0:
        return 0.7
    return 0.8 + 6.2 * math.log1p(stats.magnitude) / math.log1p(max(peak, stats.magnitude, 1e-12))


def _render_large_module_density_raw(out_path: Path, entry: LibraryEntry) -> _LargeRenderMetadata:
    """Render one oversized module directly from its serialized payload, with no Genome or graph construction."""
    # Deliberately lazy: normal startup, small portraits, galleries, task renders, and overmind cards
    # never import Datashader (or its pandas/xarray/numba dependency chain).
    import datashader as ds
    import matplotlib
    import numpy as np
    import pandas as pd
    from datashader import transfer_functions as tf

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    layout = _density_layout(entry)
    connections = entry.payload.get("connections", [])
    if not isinstance(connections, list):
        raise ValueError("module payload connections must be a list")
    macro_edges = _macro_implied_payload_edges(entry.payload)
    canvas = ds.Canvas(plot_width=_DENSITY_WIDTH, plot_height=_DENSITY_HEIGHT, x_range=(0.0, 1.0), y_range=(0.0, 1.0))

    input_id_set = set(layout.input_ids)
    outgoing_strength: dict[int, float] = {}
    active_input_ids: set[int] = set()
    for connection in connections:
        if not bool(connection.get("enabled", False)):
            continue
        source_id = int(connection["in"])
        if source_id in input_id_set:
            weight = float(connection.get("weight", 0.0))
            if not math.isfinite(weight):
                raise ValueError(f"input edge from {source_id} has a non-finite weight")
            outgoing_strength[source_id] = outgoing_strength.get(source_id, 0.0) + abs(weight)
            active_input_ids.add(source_id)
    isolated_count = len(layout.input_ids) - len(active_input_ids)
    del active_input_ids, input_id_set

    input_frame = pd.DataFrame(
        {
            "x": np.fromiter((layout.positions[node_id][0] for node_id in layout.input_ids), dtype=np.float64, count=len(layout.input_ids)),
            "y": np.fromiter((layout.positions[node_id][1] for node_id in layout.input_ids), dtype=np.float64, count=len(layout.input_ids)),
            "strength": np.fromiter((outgoing_strength.get(node_id, 0.0) for node_id in layout.input_ids), dtype=np.float64, count=len(layout.input_ids)),
        }
    )
    del outgoing_strength
    images: list[Any] = []
    edge_layers, enabled_count, rendered_count = _rasterized_edge_layers(canvas, layout.positions, connections, macro_edges, ds, pd, np)
    edge_colors = {
        "positive": THEME["edge_glue"],
        "negative": THEME["edge_exit"],
        "recurrent": THEME["edge_recurrent"],
        "macro": THEME["edge_macro"],
    }
    for category in ("positive", "negative", "recurrent", "macro"):
        aggregate = edge_layers[category]
        if aggregate is not None and bool(np.any(aggregate.data > 0)):
            visible = aggregate.where(aggregate > 0)
            images.append(tf.shade(visible, cmap=[edge_colors[category], edge_colors[category]], how="log", alpha=185, min_alpha=28))
    del edge_layers

    if layout.input_ids:
        base = canvas.points(input_frame, "x", "y", agg=ds.count())
        active = canvas.points(input_frame, "x", "y", agg=ds.sum("strength"))
        images.append(tf.spread(tf.shade(base, cmap=[THEME["node_input"], THEME["node_input"]], how="linear", alpha=105, min_alpha=105), px=1))
        if bool(np.any(np.nan_to_num(active.data, nan=0.0) > 0)):
            images.append(tf.spread(tf.shade(active.where(active > 0), cmap=["#315f9e", THEME["edge_glue"]], how="log", alpha=245, min_alpha=75), px=1))
        del active, base, input_frame

    if images:
        raster = np.asarray(tf.stack(*images, how="over").to_pil(origin="upper"))
    else:
        raster = np.zeros((_DENSITY_HEIGHT, _DENSITY_WIDTH, 4), dtype=np.uint8)
    del images

    figure = plt.figure(figsize=(_DENSITY_WIDTH / 200, _DENSITY_HEIGHT / 200), dpi=200)
    figure.patch.set_facecolor(THEME["background"])
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.set_facecolor(THEME["background"])
    axis.imshow(raster, extent=(0.0, 1.0, 0.0, 1.0), origin="lower", interpolation="nearest", zorder=1)
    axis.set_aspect("auto")
    for panel in layout.panels:
        axis.add_patch(
            Rectangle((panel.x0, panel.y0), panel.x1 - panel.x0, panel.y1 - panel.y0, fill=False, edgecolor=THEME["container_edge"], linewidth=0.55, alpha=0.8, zorder=2)
        )
        axis.text((panel.x0 + panel.x1) / 2, panel.y1 + 0.003, panel.label, color=THEME["label"], fontsize=5.5, ha="center", va="bottom", zorder=4)

    raw_nodes = entry.payload.get("nodes", [])
    computed_set = set(layout.computed_ids)
    nodes_by_id = {int(node["id"]): node for node in raw_nodes if int(node["id"]) in computed_set}
    macro_outputs = {int(node_id) for macro in entry.payload.get("macros", []) for node_id in macro.get("outputs", [])}
    marker_groups: dict[tuple[str, str], list[int]] = {}
    for node_id in layout.computed_ids:
        node = nodes_by_id[node_id]
        kind = str(node.get("kind", "hidden"))
        if node_id in macro_outputs:
            marker, color = "h", THEME["node_module"]
        elif kind == NodeKind.BIAS.value:
            marker, color = "s", THEME["node_bias"]
        elif kind == NodeKind.OUTPUT.value:
            marker, color = "s", THEME["node_output"]
        else:
            marker = "D" if node.get("aggregation", "sum") == "product" else "o"
            color = "#73d055"
        marker_groups.setdefault((marker, color), []).append(node_id)
    for (marker, color), node_ids in marker_groups.items():
        axis.scatter(
            [layout.positions[node_id][0] for node_id in node_ids],
            [layout.positions[node_id][1] for node_id in node_ids],
            s=22 if len(layout.computed_ids) <= 64 else 12,
            c=color,
            marker=marker,
            linewidths=0.35,
            edgecolors=THEME["background"],
            zorder=5,
        )
    if len(layout.computed_ids) <= 64:
        for node_id in layout.computed_ids:
            x, y = layout.positions[node_id]
            axis.text(x + 0.004, y, str(node_id), color=THEME["label"], fontsize=5, ha="left", va="center", zorder=6)

    kind_counts: dict[str, int] = {}
    for node in raw_nodes:
        kind = str(node.get("kind", "?"))
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    macro_refs = sorted({str(macro.get("ref", "?")) for macro in entry.payload.get("macros", [])})
    inputs = " + ".join(f"{item.get('signature', '?')}:{item.get('width', '?')}" for item in entry.io.get("inputs", []))
    output = entry.io.get("output", {})
    output_label = f"{output.get('signature', '?')}:{output.get('width', '?')}"
    axis.text(0.025, 0.975, f"{entry.key}  L{entry.level} module", color=THEME["title"], fontsize=15, fontweight="bold", ha="left", va="top", zorder=7)
    axis.text(0.025, 0.948, f"{inputs}  →  {output_label}", color=THEME["label"], fontsize=8, ha="left", va="top", zorder=7)
    axis.text(0.735, 0.975, "computed topology", color=THEME["title"], fontsize=10, ha="left", va="top", zorder=7)
    count_text = " · ".join(f"{kind_counts.get(kind, 0):,} {kind}" for kind in ("input", "bias", "hidden", "output") if kind_counts.get(kind, 0))
    axis.text(0.735, 0.945, count_text, color=THEME["label"], fontsize=7, ha="left", va="top", zorder=7)
    axis.text(0.735, 0.925, f"{enabled_count:,} enabled edges · {isolated_count:,} isolated inputs", color=THEME["label"], fontsize=7, ha="left", va="top", zorder=7)
    axis.text(0.735, 0.905, f"renderer: Datashader · {layout.mode} · {_DENSITY_WIDTH}×{_DENSITY_HEIGHT}", color=THEME["label"], fontsize=7, ha="left", va="top", zorder=7)
    if macro_refs:
        refs = ", ".join(ref.removeprefix("library:") for ref in macro_refs)
        axis.text(0.735, 0.885, f"macro refs (not expanded): {refs}", color=THEME["node_module"], fontsize=6.5, ha="left", va="top", wrap=True, zorder=7)
    if layout.fallback_reason:
        axis.text(0.025, 0.115, f"layout note: {layout.fallback_reason}", color=THEME["label"], fontsize=6, ha="left", va="top", zorder=7)

    legend = (
        (THEME["edge_glue"], "positive forward density"),
        (THEME["edge_exit"], "negative forward density"),
        (THEME["edge_recurrent"], "recurrent density"),
        (THEME["edge_macro"], "macro-implied flow"),
        (THEME["node_input"], "input location / outgoing |weight|"),
    )
    for index, (color, label) in enumerate(legend):
        x = 0.04 + index * 0.185
        axis.plot((x, x + 0.022), (0.075, 0.075), color=color, linewidth=2.2, zorder=7)
        axis.text(x + 0.027, 0.075, label, color=THEME["label"], fontsize=6, ha="left", va="center", zorder=7)
    axis.text(
        0.5,
        0.025,
        f"Aggregate density portrait · all {rendered_count:,} enabled edges included · intensity accumulates absolute weight",
        color=THEME["label"],
        fontsize=7,
        ha="center",
        va="center",
        zorder=7,
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=200, facecolor=figure.get_facecolor())
    plt.close(figure)
    return _LargeRenderMetadata(
        node_count=len(raw_nodes),
        enabled_edge_count=enabled_count,
        isolated_input_count=isolated_count,
        rendered_edge_count=rendered_count,
        semantic_layout_mode=layout.mode,
        fallback_reason=layout.fallback_reason,
    )


def _render_large_module_density(out_path: Path, entry: LibraryEntry) -> _LargeRenderMetadata:
    """A conceptual potential-influence portrait: spatial fields, transform cards, and output matrix."""
    import datashader as ds
    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    layout = _density_layout(entry)
    connections = entry.payload.get("connections", [])
    raw_nodes = entry.payload.get("nodes", [])
    if not isinstance(connections, list) or not isinstance(raw_nodes, list):
        raise ValueError("module payload nodes/connections must be lists")

    # Compress the field slightly to make a clean transform-card column. No semantic coordinates
    # change relative to one another; only their final screen extent changes.
    input_left, input_right = 0.04, 0.59
    old_input_right = max((panel.x1 for panel in layout.panels), default=0.69)
    if old_input_right > input_left:
        scale = (input_right - input_left) / (old_input_right - input_left)
        for node_id in layout.input_ids:
            x, y = layout.positions[node_id]
            layout.positions[node_id] = (input_left + (x - input_left) * scale, y)
        for panel in layout.panels:
            panel.x0 = input_left + (panel.x0 - input_left) * scale
            panel.x1 = input_left + (panel.x1 - input_left) * scale

    input_ids = set(layout.input_ids)
    computed_ids = set(layout.computed_ids)
    nodes_by_id = {int(node["id"]): node for node in raw_nodes if int(node["id"]) in computed_ids}
    output_ids = sorted(node_id for node_id in layout.computed_ids if nodes_by_id[node_id].get("kind") == NodeKind.OUTPUT.value)
    output_set = set(output_ids)
    transform_ids = [node_id for node_id in layout.computed_ids if node_id not in output_set]
    macro_outputs = {int(node_id) for macro in entry.payload.get("macros", []) for node_id in macro.get("outputs", [])}
    hidden_ids = [node_id for node_id in transform_ids if nodes_by_id[node_id].get("kind") == NodeKind.HIDDEN.value]
    card_ids = hidden_ids if len(hidden_ids) <= 6 else []
    card_set = set(card_ids)

    # Output-matrix columns are concrete transform nodes while that remains readable. Large
    # transform banks collapse by kind, but every edge still accumulates into its group cell.
    source_labels = ["input field"]
    source_group_for_id: dict[int, int] = {}
    if len(transform_ids) <= 11:
        for node_id in transform_ids:
            source_group_for_id[node_id] = len(source_labels)
            kind = "macro" if node_id in macro_outputs else str(nodes_by_id[node_id].get("kind", "node"))
            source_labels.append(f"{kind} {node_id}")
    else:
        grouped: dict[str, int] = {}
        for node_id in transform_ids:
            kind = "macro" if node_id in macro_outputs else str(nodes_by_id[node_id].get("kind", "node"))
            if kind not in grouped:
                grouped[kind] = len(source_labels)
                source_labels.append(f"{kind} bank")
            source_group_for_id[node_id] = grouped[kind]

    input_magnitude: dict[int, float] = {}
    input_signed: dict[int, float] = {}
    target_input_magnitude: dict[int, dict[int, float]] = {node_id: {} for node_id in card_ids}
    target_input_signed: dict[int, dict[int, float]] = {node_id: {} for node_id in card_ids}
    input_target_flow: dict[int, _FlowStats] = {}
    computed_flow: dict[tuple[int, int, bool], _FlowStats] = {}
    source_output_flow: dict[int, _FlowStats] = {}
    output_cells: dict[tuple[int, int], _FlowStats] = {}
    active_inputs: set[int] = set()
    enabled_count = 0

    for connection in connections:
        if not bool(connection.get("enabled", False)):
            continue
        enabled_count += 1
        source_id, target_id = int(connection["in"]), int(connection["out"])
        if source_id not in layout.positions or target_id not in layout.positions:
            raise ValueError(f"enabled edge {source_id}->{target_id} names a missing node")
        weight = float(connection.get("weight", 0.0))
        if not math.isfinite(weight):
            raise ValueError(f"enabled edge {source_id}->{target_id} has a non-finite weight")
        recurrent = bool(connection.get("recurrent", False))
        if source_id in input_ids:
            input_magnitude[source_id] = input_magnitude.get(source_id, 0.0) + abs(weight)
            input_signed[source_id] = input_signed.get(source_id, 0.0) + weight
            active_inputs.add(source_id)
            source_group = 0
            if target_id not in output_set:
                input_target_flow.setdefault(target_id, _FlowStats()).add(weight)
            if target_id in card_set:
                magnitude = target_input_magnitude[target_id]
                signed = target_input_signed[target_id]
                magnitude[source_id] = magnitude.get(source_id, 0.0) + abs(weight)
                signed[source_id] = signed.get(source_id, 0.0) + weight
        else:
            source_group = source_group_for_id.get(source_id, -1)
        if target_id in output_set:
            if source_group < 0:
                source_group = len(source_labels)
                source_group_for_id[source_id] = source_group
                source_labels.append(f"output feedback {source_id}")
            output_cells.setdefault((target_id, source_group), _FlowStats()).add(weight)
            source_output_flow.setdefault(source_group, _FlowStats()).add(weight)
        elif source_id not in input_ids:
            computed_flow.setdefault((source_id, target_id, recurrent), _FlowStats()).add(weight)

    macro_flow: dict[tuple[int, int], _FlowStats] = {}
    for source_id, target_id in _macro_implied_payload_edges(entry.payload):
        if source_id not in layout.positions or target_id not in layout.positions:
            raise ValueError(f"macro-implied edge {source_id}->{target_id} names a missing node")
        macro_flow.setdefault((source_id, target_id), _FlowStats()).add(1.0)
    isolated_count = len(layout.input_ids) - len(active_inputs)

    main_fields: list[tuple[_DensityPanel, Any, Any]] = []
    main_spans: list[float] = []
    for panel in layout.panels:
        magnitude, signed = _panel_influence_aggregate(panel, layout.positions, input_magnitude, input_signed, ds, pd, np)
        main_fields.append((panel, magnitude, signed))
        nonzero = magnitude[magnitude > 0]
        if nonzero.size:
            main_spans.append(float(np.quantile(nonzero, 0.99)))
    main_span = max(main_spans, default=1.0)

    figure = plt.figure(figsize=(_DENSITY_WIDTH / 200, _DENSITY_HEIGHT / 200), dpi=200)
    figure.patch.set_facecolor(THEME["background"])
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.set_facecolor(THEME["background"])
    axis.set_aspect("auto")

    # Filled semantic fields: no raw edge crosses this region.
    for panel, magnitude, signed in main_fields:
        rgba = _signed_influence_rgba(magnitude, signed, main_span, np, mcolors)
        axis.imshow(rgba, extent=(panel.x0, panel.x1, panel.y0, panel.y1), origin="lower", interpolation="nearest", aspect="auto", zorder=1)
        axis.add_patch(Rectangle((panel.x0, panel.y0), panel.x1 - panel.x0, panel.y1 - panel.y0, fill=False, edgecolor=THEME["container_edge"], linewidth=0.7, zorder=2))
        axis.text((panel.x0 + panel.x1) / 2, panel.y1 + 0.003, panel.label, color=THEME["label"], fontsize=5.5, ha="center", va="bottom", zorder=4)

    # Hidden-node cards carry locally normalized receptive-field thumbnails.
    transform_positions = dict(layout.positions)
    card_rects: dict[int, tuple[float, float, float, float]] = {}
    if card_ids:
        card_x0, card_x1 = 0.625, 0.805
        centers = np.linspace(0.79, 0.27, len(card_ids)) if len(card_ids) > 1 else np.asarray([0.54])
        card_height = min(0.15, 0.48 / max(len(card_ids), 1))
        panel_y_min = min((panel.y0 for panel in layout.panels), default=0.16)
        panel_y_max = max((panel.y1 for panel in layout.panels), default=0.90)
        panel_y_span = max(panel_y_max - panel_y_min, 1e-6)
        for node_id, center_value in zip(card_ids, centers):
            center_y = float(center_value)
            y0, y1 = center_y - card_height / 2, center_y + card_height / 2
            card_rects[node_id] = (card_x0, y0, card_x1, y1)
            transform_positions[node_id] = (card_x1, center_y)
            axis.add_patch(
                FancyBboxPatch(
                    (card_x0, y0),
                    card_x1 - card_x0,
                    y1 - y0,
                    boxstyle="round,pad=0.002,rounding_size=0.006",
                    facecolor=THEME["panel_even"],
                    edgecolor=THEME["container_edge"],
                    linewidth=0.7,
                    zorder=3,
                )
            )
            target_fields: list[tuple[_DensityPanel, Any, Any]] = []
            target_spans: list[float] = []
            for panel in layout.panels:
                magnitude, signed = _panel_influence_aggregate(panel, layout.positions, target_input_magnitude[node_id], target_input_signed[node_id], ds, pd, np)
                target_fields.append((panel, magnitude, signed))
                nonzero = magnitude[magnitude > 0]
                if nonzero.size:
                    target_spans.append(float(np.quantile(nonzero, 0.99)))
            target_span = max(target_spans, default=1.0)
            map_x0, map_x1 = card_x0 + 0.004, card_x0 + 0.102
            map_y0, map_y1 = y0 + 0.008, y1 - 0.008
            for panel, magnitude, signed in target_fields:
                px0 = map_x0 + (panel.x0 - input_left) / max(input_right - input_left, 1e-6) * (map_x1 - map_x0)
                px1 = map_x0 + (panel.x1 - input_left) / max(input_right - input_left, 1e-6) * (map_x1 - map_x0)
                py0 = map_y0 + (panel.y0 - panel_y_min) / panel_y_span * (map_y1 - map_y0)
                py1 = map_y0 + (panel.y1 - panel_y_min) / panel_y_span * (map_y1 - map_y0)
                rgba = _signed_influence_rgba(magnitude, signed, target_span, np, mcolors)
                axis.imshow(rgba, extent=(px0, px1, py0, py1), origin="lower", interpolation="nearest", aspect="auto", zorder=4)
            stats = input_target_flow.get(node_id, _FlowStats())
            kind = "macro" if node_id in macro_outputs else "hidden"
            axis.text(card_x0 + 0.108, y1 - 0.018, f"{kind} {node_id}", color=THEME["title"], fontsize=6.2, ha="left", va="top", zorder=6)
            axis.text(card_x0 + 0.108, y1 - 0.041, f"{stats.count:,} edges", color=THEME["label"], fontsize=5.2, ha="left", va="top", zorder=6)
            axis.text(card_x0 + 0.108, y1 - 0.062, f"Σ|w| {stats.magnitude:,.2g}", color=THEME["label"], fontsize=5.2, ha="left", va="top", zorder=6)

    for index, node_id in enumerate(node_id for node_id in transform_ids if nodes_by_id[node_id].get("kind") == NodeKind.BIAS.value):
        transform_positions[node_id] = (0.805, 0.145 - index * 0.025)

    # The output bank is a signed influence matrix. For 200 outputs this replaces an unreadable
    # marker/edge comb; for small banks it remains concrete and receives row labels.
    matrix_x0, matrix_x1, matrix_y0, matrix_y1 = 0.855, 0.975, 0.17, 0.84
    output_row = {node_id: index for index, node_id in enumerate(output_ids)}
    matrix_magnitude = np.zeros((max(len(output_ids), 1), max(len(source_labels), 1)), dtype=np.float32)
    matrix_signed = np.zeros_like(matrix_magnitude)
    for (output_id, source_group), stats in output_cells.items():
        matrix_magnitude[output_row[output_id], source_group] = stats.magnitude
        matrix_signed[output_row[output_id], source_group] = stats.signed
    matrix_nonzero = matrix_magnitude[matrix_magnitude > 0]
    matrix_span = float(np.quantile(matrix_nonzero, 0.99)) if matrix_nonzero.size else 1.0
    matrix_rgba = _signed_influence_rgba(matrix_magnitude, matrix_signed, matrix_span, np, mcolors)
    axis.imshow(matrix_rgba, extent=(matrix_x0, matrix_x1, matrix_y0, matrix_y1), origin="lower", interpolation="nearest", aspect="auto", zorder=3)
    axis.add_patch(Rectangle((matrix_x0, matrix_y0), matrix_x1 - matrix_x0, matrix_y1 - matrix_y0, fill=False, edgecolor=THEME["container_edge"], linewidth=0.8, zorder=4))
    axis.text(
        (matrix_x0 + matrix_x1) / 2,
        matrix_y1 + 0.028,
        f"output influence matrix · {len(output_ids):,} outputs",
        color=THEME["title"],
        fontsize=7,
        ha="center",
        va="bottom",
        zorder=6,
    )
    for index, label in enumerate(source_labels):
        x = matrix_x0 + (index + 0.5) * (matrix_x1 - matrix_x0) / max(len(source_labels), 1)
        axis.text(x, matrix_y1 - 0.006, label, color=THEME["title"], fontsize=4.3, rotation=90, ha="center", va="top", zorder=6)
    if len(output_ids) <= 16:
        for index, output_id in enumerate(output_ids):
            y = matrix_y0 + (index + 0.5) * (matrix_y1 - matrix_y0) / max(len(output_ids), 1)
            axis.text(matrix_x1 + 0.003, y, str(output_id), color=THEME["label"], fontsize=5, ha="left", va="center", zorder=6)

    all_stats = list(input_target_flow.values()) + list(computed_flow.values()) + list(source_output_flow.values()) + list(macro_flow.values())
    flow_peak = max((stats.magnitude for stats in all_stats), default=1.0)

    def draw_flow(
        start: tuple[float, float],
        end: tuple[float, float],
        stats: _FlowStats,
        *,
        curve: float = 0.0,
        color: str | None = None,
        dashed: bool = False,
        zorder: int = 5,
    ) -> None:
        rgba = mcolors.to_rgba(color, 0.75) if color else _flow_color(stats, mcolors)
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                connectionstyle=f"arc3,rad={curve}",
                arrowstyle="-|>",
                mutation_scale=7.0,
                linewidth=_flow_width(stats, flow_peak),
                linestyle="dashed" if dashed else "solid",
                color=rgba,
                capstyle="round",
                joinstyle="round",
                zorder=zorder,
            )
        )

    for target_id, stats in input_target_flow.items():
        if target_id in transform_positions:
            target = transform_positions[target_id]
            end_x = card_rects[target_id][0] if target_id in card_rects else target[0]
            draw_flow((input_right + 0.006, target[1]), (end_x, target[1]), stats, zorder=2)
    for (source_id, target_id, recurrent), stats in computed_flow.items():
        source = transform_positions.get(source_id)
        if source is None and source_id in output_row:
            source = (matrix_x1, matrix_y0 + (output_row[source_id] + 0.5) * (matrix_y1 - matrix_y0) / max(len(output_ids), 1))
        if source is not None and target_id in transform_positions:
            draw_flow(
                source,
                transform_positions[target_id],
                stats,
                curve=0.32 if recurrent else 0.12,
                color=THEME["edge_recurrent"] if recurrent else None,
                dashed=recurrent,
            )
    for (source_id, target_id), stats in macro_flow.items():
        source = transform_positions.get(source_id, layout.positions[source_id])
        target = transform_positions.get(target_id, layout.positions[target_id])
        draw_flow(source, target, stats, curve=-0.15, color=THEME["edge_macro"], dashed=True)
    for source_group, stats in source_output_flow.items():
        column_x = matrix_x0 + (source_group + 0.5) * (matrix_x1 - matrix_x0) / max(len(source_labels), 1)
        if source_group == 0:
            rgba = _flow_color(stats, mcolors)
            axis.plot((input_right + 0.006, matrix_x0 - 0.018), (0.125, 0.125), color=rgba, linewidth=_flow_width(stats, flow_peak), alpha=0.72, zorder=2)
            draw_flow((matrix_x0 - 0.018, 0.125), (column_x, matrix_y0), stats, curve=0.16, zorder=2)
            axis.text(0.70, 0.108, "direct input bypass", color=THEME["label"], fontsize=5.5, ha="center", va="top", zorder=6)
        else:
            group_ids = [node_id for node_id, group in source_group_for_id.items() if group == source_group]
            points = [transform_positions[node_id] for node_id in group_ids if node_id in transform_positions]
            if points:
                start = (max(point[0] for point in points), sum(point[1] for point in points) / len(points))
                draw_flow(start, (column_x, matrix_y1), stats, curve=-0.14, zorder=2)

    marker_groups: dict[tuple[str, str], list[int]] = {}
    for node_id in transform_ids:
        node = nodes_by_id[node_id]
        kind = str(node.get("kind", "hidden"))
        if node_id in macro_outputs:
            marker, color = "h", THEME["node_module"]
        elif kind == NodeKind.BIAS.value:
            marker, color = "s", THEME["node_bias"]
        else:
            marker = "D" if node.get("aggregation", "sum") == "product" else "o"
            color = "#73d055"
        marker_groups.setdefault((marker, color), []).append(node_id)
    for (marker, color), node_ids in marker_groups.items():
        axis.scatter(
            [transform_positions[node_id][0] for node_id in node_ids],
            [transform_positions[node_id][1] for node_id in node_ids],
            s=27,
            c=color,
            marker=marker,
            linewidths=0.45,
            edgecolors=THEME["background"],
            zorder=7,
        )
    for node_id in transform_ids:
        if node_id not in card_set:
            x, y = transform_positions[node_id]
            axis.text(x + 0.005, y, str(node_id), color=THEME["label"], fontsize=5, ha="left", va="center", zorder=7)

    kind_counts: dict[str, int] = {}
    for node in raw_nodes:
        kind = str(node.get("kind", "?"))
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    macro_refs = sorted({str(macro.get("ref", "?")) for macro in entry.payload.get("macros", [])})
    inputs = " + ".join(f"{item.get('signature', '?')}:{item.get('width', '?')}" for item in entry.io.get("inputs", []))
    output = entry.io.get("output", {})
    output_label = f"{output.get('signature', '?')}:{output.get('width', '?')}"
    axis.text(0.025, 0.975, f"{entry.key}  L{entry.level} module", color=THEME["title"], fontsize=15, fontweight="bold", ha="left", va="top", zorder=8)
    axis.text(0.025, 0.948, f"{inputs}  →  {output_label}", color=THEME["label"], fontsize=8, ha="left", va="top", zorder=8)
    axis.text(0.04, 0.922, "spatial potential influence · filled semantic H×W cells", color=THEME["label"], fontsize=6.3, ha="left", va="top", zorder=8)
    axis.text(0.625, 0.975, "potential influence flow", color=THEME["title"], fontsize=11, fontweight="bold", ha="left", va="top", zorder=8)
    axis.text(0.625, 0.949, "conceptual weight flow · not activation analysis", color=THEME["label"], fontsize=6.8, ha="left", va="top", zorder=8)
    count_text = " · ".join(f"{kind_counts.get(kind, 0):,} {kind}" for kind in ("input", "bias", "hidden", "output") if kind_counts.get(kind, 0))
    axis.text(0.625, 0.928, count_text, color=THEME["label"], fontsize=6.5, ha="left", va="top", zorder=8)
    axis.text(0.625, 0.908, f"{enabled_count:,} enabled edges · {isolated_count:,} isolated inputs", color=THEME["label"], fontsize=6.5, ha="left", va="top", zorder=8)
    axis.text(0.625, 0.888, f"renderer: Datashader · {layout.mode} · {_DENSITY_WIDTH}×{_DENSITY_HEIGHT}", color=THEME["label"], fontsize=5.8, ha="left", va="top", zorder=8)
    if macro_refs:
        refs = ", ".join(ref.removeprefix("library:") for ref in macro_refs)
        axis.text(0.625, 0.870, f"macro refs (not expanded): {refs}", color=THEME["node_module"], fontsize=6, ha="left", va="top", wrap=True, zorder=8)
    if layout.fallback_reason:
        axis.text(0.025, 0.112, f"layout note: {layout.fallback_reason}", color=THEME["label"], fontsize=5.5, ha="left", va="top", zorder=8)

    legend = (
        (THEME["edge_glue"], "positive influence"),
        (THEME["edge_exit"], "negative influence"),
        ("#7d86a3", "mixed sign"),
        (THEME["edge_recurrent"], "recurrent"),
        (THEME["edge_macro"], "macro-implied"),
    )
    for index, (color, label) in enumerate(legend):
        x = 0.04 + index * 0.185
        axis.plot((x, x + 0.022), (0.072, 0.072), color=color, linewidth=2.2, zorder=8)
        axis.text(x + 0.027, 0.072, label, color=THEME["label"], fontsize=6, ha="left", va="center", zorder=8)
    axis.text(
        0.04,
        0.048,
        "field brightness = log accumulated |weight| · field hue = signed balance · hidden cards are locally scaled",
        color=THEME["label"],
        fontsize=5.8,
        ha="left",
        va="center",
        zorder=8,
    )
    axis.text(
        0.5,
        0.022,
        f"Potential influence flow (weights, not activations) · all {enabled_count:,} enabled edges included in fields, ribbons, or matrices",
        color=THEME["label"],
        fontsize=7,
        ha="center",
        va="center",
        zorder=8,
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(out_path, dpi=200, facecolor=figure.get_facecolor())
    finally:
        plt.close(figure)
    return _LargeRenderMetadata(
        node_count=len(raw_nodes),
        enabled_edge_count=enabled_count,
        isolated_input_count=isolated_count,
        rendered_edge_count=enabled_count,
        semantic_layout_mode=layout.mode,
        fallback_reason=layout.fallback_reason,
    )


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
        from ardevo.utils.logging import Logger

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


# --- gallery ---------------------------------------------------------------------------------------


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
                spec = build_entry_spec(library.load(summary["key"]), resolve=resolve, node_budget=GALLERY_NODE_BUDGET, max_inline_depth=max_inline_depth)
                draw_spec(axis, spec)
            except Exception:
                axis.set_facecolor(THEME["background"])
                axis.text(0.5, 0.5, f"{summary['key']}\nrender failed", ha="center", va="center", color=THEME["label"], fontsize=7)
                axis.axis("off")
            axis.set_title(_cell_title(summary), fontsize=7, color=THEME["title"])
        for axis in flat_axes[len(rows) :]:
            axis.set_facecolor(THEME["background"])
            axis.axis("off")

    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)
    return out_path


# --- motif atlas -----------------------------------------------------------------------------------

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
            color, marker = THEME["node_module"], "h"
        elif kind == "input":
            color, marker = (THEME["node_bias"] if second == "bias" else THEME["node_input"]), "s"
        elif kind == "bias":
            color, marker = THEME["node_bias"], "s"
        elif kind == "output":
            color, marker = THEME["node_output"], "s"
        else:
            color, marker = _ACTIVATION_TINTS.get(second, THEME["edge_forward"]), ("D" if aggregation == "product" else "o")
        spec.nodes.append(SpecNode(x, y, color, size=1.3, marker=marker))

    for source, target, mask in edges:
        if source == target:
            # A recurrent self-loop (the TRM refinement motif) draws as a small arc riding the node:
            # arc3 renders nothing when both endpoints coincide.
            x, y = positions[source]
            spec.edges.append(SpecEdge(x - 0.2, y + 0.18, x + 0.2, y + 0.18, width=1.2, color=THEME["edge_recurrent"], style="dashed", curve=1.6, alpha=0.8))
            continue
        (x0, y0), (x1, y1) = positions[source], positions[target]
        if mask & FORWARD_EDGE:
            spec.edges.append(SpecEdge(x0, y0, x1, y1, width=1.4, color=THEME["edge_forward"], alpha=0.7))
        if mask & MACRO_EDGE:
            spec.edges.append(SpecEdge(x0, y0, x1, y1, width=1.2, color=THEME["edge_macro"], alpha=0.6))
        if mask & RECURRENT_EDGE:
            spec.edges.append(SpecEdge(x0, y0, x1, y1, width=1.2, color=THEME["edge_recurrent"], style="dashed", curve=0.25, alpha=0.7))
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
                draw_spec(axis, build_motif_spec(record.graph.node_labels, record.graph.edges))
            except Exception:
                axis.set_facecolor(THEME["background"])
                axis.text(0.5, 0.5, f"{record.fingerprint}\nrender failed", ha="center", va="center", color=THEME["label"], fontsize=7)
                axis.axis("off")
            axis.set_title(f"{record.diversity_class}  s={record.support} n={record.occurrences}", fontsize=7, color=THEME["title"])
        for axis in flat_axes[len(motifs) :]:
            axis.set_facecolor(THEME["background"])
            axis.axis("off")

    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)
    return out_path


# --- overmind: the whole routed model in one frame -------------------------------------------------


@dataclass(slots=True)
class OvermindVertex:
    """A read-only view of one routed-model expert, so rendering stays decoupled from `routing.py`."""

    key: str  # the library key (or "" for a synthetic vertex); resolved to its embedded network
    label: str  # display label (usage share, stone/retired markers, ...)
    retired: bool = False
    usage: float = 0.0  # lifetime gate-mass share in [0, 1]
    entry_share: float = 0.0  # step-0 gate mass, peak-normalized; widths the input feed edge
    exit_share: float = 0.0  # final-step gate mass, peak-normalized; widths the output feed edge
    mean_step: float | None = None  # mass-weighted mean firing step; None = never trafficked
    embedding_rank: int = 0  # 1D latent-projection rank; orders cells WITHIN a grid row
    stepping_stone: bool = False  # wall-ledger below-bar admission (immature circuit)


@dataclass(slots=True)
class OvermindView:
    """Everything the overmind render needs about a routed model, as plain data. `routing.py` builds
    this from a live `RoutedNet`; the renderer never imports the router. Vertices arrive in FINAL
    ROW ORDER (traffic-first: early-firing experts lead); the renderer only chunks them."""

    vertices: list[OvermindVertex]
    input_signatures: list[str]  # one per input adapter (e.g. "BINARY|K:2")
    output_signatures: list[str]  # one per output head
    d_model: int
    top_k: int
    max_steps: int
    # REAL routing paths: directed (source_index, target_index, weight in [0, 1]) between vertices,
    # from observed step-to-step traffic (or the edge_bias prior as a fallback). Drawn as curved
    # weighted edges from concrete source outputs to target network-input anchors.
    pathways: list[tuple[int, int, float]] = field(default_factory=list)


def prune_overmind_view(view: OvermindView) -> OvermindView:
    """Remove retired cards and compact the remaining portrait without changing its grid width.

    Vertex order is preserved, so :func:`build_overmind_spec` repacks the survivors left-to-right
    into the same eight-column rows.  Pathway indices are remapped and edges touching a hidden
    vertex disappear.
    """

    kept = [index for index, vertex in enumerate(view.vertices) if not vertex.retired]
    remap = {old: new for new, old in enumerate(kept)}
    pathways = [(remap[source], remap[target], weight) for source, target, weight in view.pathways if source in remap and target in remap]
    return OvermindView(
        vertices=[view.vertices[index] for index in kept],
        input_signatures=list(view.input_signatures),
        output_signatures=list(view.output_signatures),
        d_model=view.d_model,
        top_k=view.top_k,
        max_steps=view.max_steps,
        pathways=pathways,
    )


_ROW_GAP = 2.4  # vertical gap between grid rows; leaves room for the container label above a box
_BAND_GAP = 4  # clearance between the input/output bands and the grid
_BAND_H = 1.6  # band strip: node row plus its signature label
_LEGEND_ROW_STEP = 1.8
_LEGEND_WIDTH = 24.0
_OVERMIND_COLUMNS = 8
_OVERMIND_DPI = 300
_OVERMIND_X_PADDING = 2 * _PAD


def _overmind_legend_entries() -> list[tuple[str, dict[str, Any], str]]:
    """Every marker/edge class the overmind canvas can show, as (swatch kind, params, label) rows."""
    early, deep = _layer_color(0, 3), _layer_color(3, 3)
    return [
        ("node", {"color": THEME["node_input"], "marker": "s"}, "input"),
        ("node", {"color": THEME["node_bias"], "marker": "s", "alpha": 0.7}, "bias"),
        ("node", {"color": THEME["node_output"], "marker": "s"}, "output"),
        ("node", {"color": early, "marker": "o"}, "hidden (early layer)"),
        ("node", {"color": deep, "marker": "o"}, "hidden (deep layer)"),
        ("node", {"color": deep, "marker": "D"}, "product gate"),
        ("node", {"color": THEME["node_module"], "marker": "h"}, "module ref / macro footprint"),
        ("node", {"color": THEME["node_anchor"], "marker": "o", "size": 0.65}, "network input anchor"),
        ("node", {"color": deep, "marker": "o", "alpha": 0.25, "size": 0.5}, "isolated (unused)"),
        ("box", {}, "retired or unexpanded network"),
        ("edge", {"color": THEME["edge_forward"]}, "forward connection"),
        ("edge", {"color": THEME["edge_recurrent"], "style": "dashed", "curve": 0.25}, "recurrent (time-delayed)"),
        ("edge", {"color": THEME["edge_macro"]}, "macro implied wiring"),
        ("edge", {"color": THEME["edge_glue"]}, "composition glue"),
        ("edge", {"color": THEME["edge_callout"]}, "nested-network flow"),
        ("edge", {"color": THEME["edge_pathway"], "curve": 0.25}, "routing traffic (observed)"),
        ("edge", {"color": THEME["edge_entry"]}, "input feed (step-0 gate mass)"),
        ("edge", {"color": THEME["edge_exit"]}, "output feed (final-step gate mass)"),
    ]


def _overmind_legend(spec: RenderSpec, x0: float, y_top: float, entries: list[tuple[str, dict[str, Any], str]]) -> float:
    """Append the key panel at the right margin; returns the panel width."""
    spec.texts.append(SpecText(x0 + 0.6, y_top - 0.6, "key", size=8.0, color=THEME["title"]))
    y = y_top - 0.6
    for kind, params, label in entries:
        y -= _LEGEND_ROW_STEP
        if kind == "node":
            spec.nodes.append(SpecNode(x0 + 1.0, y, color=params["color"], size=params.get("size", 1.2), marker=params.get("marker", "o"), alpha=params.get("alpha", 1.0)))
        elif kind == "edge":
            spec.edges.append(SpecEdge(x0 + 0.4, y, x0 + 1.8, y, width=1.6, color=params["color"], style=params.get("style", "solid"), curve=params.get("curve", 0.0), alpha=0.9))
        else:
            spec.containers.append(SpecContainer(x0 + 0.3, y - 0.35, x0 + 1.9, y + 0.35, label="", depth=1, opaque=True))
        spec.texts.append(SpecText(x0 + 2.4, y, label, size=6.5))
    spec.containers.append(SpecContainer(x0, y - 0.8, x0 + _LEGEND_WIDTH, y_top, label="", depth=0))
    return _LEGEND_WIDTH


def build_overmind_spec(
    view: OvermindView,
    *,
    resolve: ResolveFn | None = None,
    node_budget: int = DEFAULT_NODE_BUDGET,
    cell_node_budget: int = 160,
    columns: int = _OVERMIND_COLUMNS,
    legend: bool = True,
    max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH,
) -> RenderSpec:
    """The whole routed model as a top-down flow portrait: input adapters band across the TOP, every
    expert a fully-embedded cell in a `columns`-wide grid (row order = observed firing order, so an
    input can be traced downward through the paths it actually takes), output heads across the
    BOTTOM. There is no gate-hub node: the gate IS the edge fabric (feed widths and pathway edges
    carry the learned routing). This whole-graph portrait is for the current scale; at thousands of
    vertices it becomes a density-map problem, not more edges.

    Never raises: an unresolvable/oversized expert degrades to a labeled opaque box, like everywhere.

    `cell_node_budget` is the PER-EXPERT detail cap, separate from the shared `node_budget`: one
    image-scale entry (the 798-node MNIST stepping stone, 2026-07-05) otherwise fits the shared
    budget, embeds as a ~784-node input column, and its cell height degenerates the whole portrait
    into a tall-narrow bar while starving every other cell's budget. Above the cap an expert reads
    as its labeled opaque footprint; its own full-detail portrait still lives in images/<key>.png."""
    spec = RenderSpec()
    inputs = view.input_signatures or ["(no input adapter yet)"]
    outputs = view.output_signatures or ["(no output head yet)"]
    columns = max(1, columns)

    budget = _Budget(node_budget)
    children: list[_Built] = []
    for vertex in view.vertices:
        entry: LibraryEntry | None = None
        if vertex.key:
            allowance = min(cell_node_budget, budget.remaining)
            cell_budget = _Budget(allowance)
            entry = resolve(vertex.key) if resolve is not None else None
            try:
                # An overmind cell is a root payload, just like render_entry: selecting it for the
                # grid does not consume a reference level. Seed its key only for cycle detection.
                child = (
                    _build_entry(
                        entry,
                        resolve=resolve,
                        budget=cell_budget,
                        depth=1,
                        reference_depth=0,
                        stack=(vertex.key,),
                        max_inline_depth=max_inline_depth,
                    )
                    if entry is not None
                    else _opaque_built(vertex.key)
                )
            except Exception:
                child = _opaque_built(f"{vertex.key}  ?")
            budget.remaining -= allowance - cell_budget.remaining
        else:
            child = _opaque_built(vertex.label)
        child.label = f"{vertex.label}\n{_entry_size_label(entry)}" if child.opaque and entry is not None else vertex.label
        child.opaque = child.opaque or vertex.retired  # retired experts read as opaque footprints
        children.append(child)
    boxes = [(child.spec.width + 2 * _PAD, child.spec.height + 2 * _PAD) for child in children]

    # Rows of `columns` in arrival (traffic) order; WITHIN a row, latent order keeps similar experts
    # adjacent on the x axis. Rows are independently centered flow rows, never rigid column slots:
    # one oversized cell would otherwise blow a whole column wide and smear the rest.
    rows = [list(range(start, min(start + columns, len(children)))) for start in range(0, len(children), columns)]
    for row in rows:
        row.sort(key=lambda index: (view.vertices[index].embedding_rank, index))
    row_heights = [max(boxes[i][1] for i in row) for row in rows]
    row_widths = [sum(boxes[i][0] for i in row) + _H_GAP * (len(row) - 1) for row in rows]
    grid_width = max([*row_widths, 8.0])
    grid_height = sum(row_heights) + _ROW_GAP * max(len(rows) - 1, 0)

    legend_entries = _overmind_legend_entries() if legend else []
    legend_height = 1.4 + _LEGEND_ROW_STEP * len(legend_entries) + 1.0
    total_height = max(2 * (_BAND_H + _BAND_GAP) + grid_height, legend_height)

    bottoms: list[tuple[float, float]] = [(0.0, 0.0)] * len(children)
    anchors: list[tuple[float, float]] = [(0.0, 0.0)] * len(children)
    y_cursor = total_height - _BAND_H - _BAND_GAP  # y-up frame: the top rail of the first row
    for row, row_height, row_width in zip(rows, row_heights, row_widths):
        x_cursor = (grid_width - row_width) / 2
        for i in row:
            box_w, box_h = boxes[i]
            cx, cy = x_cursor + box_w / 2, y_cursor - box_h / 2  # top-aligned: a clean rail for input feeds
            _place_child(spec, children[i], (cx, cy), depth=1)
            bottoms[i] = (cx, y_cursor - box_h)
            anchors[i] = (cx - box_w / 2 + _NETWORK_ANCHOR_INSET, y_cursor - _NETWORK_ANCHOR_INSET)
            spec.nodes.append(SpecNode(*anchors[i], color=THEME["node_anchor"], size=0.65))
            if not children[i].opaque and box_w < 2.5:
                # draw_spec skips container labels on narrow boxes; every cell still deserves a name
                spec.texts.append(SpecText(cx - box_w / 2 + 0.15, y_cursor + 0.08, children[i].label, size=5.0, va="bottom"))
            x_cursor += box_w + _H_GAP
        y_cursor -= row_height + _ROW_GAP

    def _band(signatures: list[str], y: float, color: str) -> list[tuple[float, float]]:
        placed: list[tuple[float, float]] = []
        for index, signature in enumerate(signatures):
            x = grid_width * (index + 1) / (len(signatures) + 1)
            spec.nodes.append(SpecNode(x, y, color=color, size=1.4, marker="s"))
            spec.texts.append(SpecText(x, y - 0.45, signature, size=6.0, ha="center", va="top"))
            placed.append((x, y))
        return placed

    input_positions = _band(inputs, total_height - 0.5, THEME["node_input"])
    output_positions = _band(outputs, 0.5, THEME["node_output"])

    def _outputs(index: int) -> list[tuple[float, float]]:
        rendered = children[index].output_nodes
        return [(node.x, node.y) for node in rendered] if rendered else [bottoms[index]]

    # Traffic feeds. Full bipartite is honest: the bus injects every adapter into every selected
    # expert, and per-adapter attribution does not exist in the ledgers. A fresh library (no traffic
    # yet) draws uniform thin feeds so the flow story exists on day one.
    live = [index for index, vertex in enumerate(view.vertices) if not vertex.retired]
    uniform = all(view.vertices[index].entry_share <= 0.0 and view.vertices[index].exit_share <= 0.0 for index in live)
    for index in live:
        vertex = view.vertices[index]
        entry_width = 0.8 if uniform else (_edge_width(vertex.entry_share * 3) if vertex.entry_share > 0.0 else 0.0)
        exit_width = 0.8 if uniform else (_edge_width(vertex.exit_share * 3) if vertex.exit_share > 0.0 else 0.0)
        alpha = 0.2 if uniform else 0.35
        if entry_width > 0.0:
            for x, y in input_positions:
                spec.edges.append(SpecEdge(x, y, anchors[index][0], anchors[index][1], width=entry_width, color=THEME["edge_entry"], alpha=alpha))
        if exit_width > 0.0:
            for source_x, source_y in _outputs(index):
                for x, y in output_positions:
                    spec.edges.append(SpecEdge(source_x, source_y, x, y, width=exit_width, color=THEME["edge_exit"], alpha=alpha))

    # THE ROUTING PATHS: output node -> target input anchor, from observed traffic (or the edge-bias
    # prior). An observed self-transition is recurrent expert use and loops back to that card's own
    # anchor. Retired vertices carry no pathways.
    for source, target, weight in view.pathways:
        if not (0 <= source < len(children) and 0 <= target < len(children)):
            continue
        if view.vertices[source].retired or view.vertices[target].retired:
            continue
        x1, y1 = anchors[target]
        for x0, y0 in _outputs(source):
            spec.edges.append(SpecEdge(x0, y0, x1, y1, width=_edge_width(weight * 3), color=THEME["edge_pathway"], curve=0.25, alpha=0.25 + 0.6 * min(weight, 1.0)))

    spec.width = grid_width
    if legend:
        spec.width = grid_width + 2 * _H_GAP + _overmind_legend(spec, grid_width + 2 * _H_GAP, total_height - 0.5, legend_entries)
    spec.height = total_height
    return spec


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
        pruned_spec = build_overmind_spec(pruned, resolve=resolve, node_budget=node_budget, max_inline_depth=max_inline_depth)
        pruned_title = f"overmind current: {len(pruned.vertices)} experts, d_model={view.d_model}, top_k={view.top_k}, steps={view.max_steps}"
        _render_spec_png(pruned_path, pruned_spec, pruned_title, dpi=_OVERMIND_DPI, x_padding=_OVERMIND_X_PADDING)
    return rendered
