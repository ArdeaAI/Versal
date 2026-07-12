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

import math
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ardevo.evolution.composition import CompNodeKind, CompositionGenome, comp_from_dict, comp_topological_order
from ardevo.evolution.genome import Genome, NodeKind, genome_from_dict, macro_implied_edges, make_acyclic, topological_order
from ardevo.library import COMPOSITION, MODULE, LibraryEntry, ModuleLibrary
from ardevo.motifs import FORWARD_EDGE, MACRO_EDGE, RECURRENT_EDGE, MotifRecord, NodeLabel

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

RENDER_MAX_DEPTH = 4  # matches substrate._MAX_MACRO_DEPTH: deeper refs exist only pathologically
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


def _build_ref(ref: str, *, resolve: ResolveFn | None, budget: _Budget, depth: int, stack: tuple[str, ...]) -> _Built:
    if not ref.startswith("library:"):
        return _opaque_built(ref)  # live refs only exist mid-run; renders happen on admitted entries
    key = ref.removeprefix("library:")
    if key in stack or depth >= RENDER_MAX_DEPTH or resolve is None:
        return _opaque_built(key)
    entry = resolve(key)
    if entry is None:
        return _opaque_built(key)
    label = f"{entry.key}  L{entry.level}"
    try:
        if len(entry.payload["nodes"]) > budget.remaining:
            return _opaque_built(label)
        if entry.entry_type == MODULE:
            built = _build_genome(genome_from_dict(entry.payload), resolve=resolve, budget=budget, depth=depth + 1, stack=stack + (key,))
        elif entry.entry_type == COMPOSITION:
            built = _build_comp(comp_from_dict(entry.payload), resolve=resolve, budget=budget, depth=depth + 1, stack=stack + (key,))
        else:
            return _opaque_built(f"{label}  ?")
    except Exception:
        return _opaque_built(f"{label}  ?")
    built.label = label
    return built


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


def _build_genome(genome: Genome, *, resolve: ResolveFn | None, budget: _Budget, depth: int, stack: tuple[str, ...]) -> _Built:
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
        child = _build_ref(macro.ref, resolve=resolve, budget=budget, depth=depth, stack=stack)
        anchors = [positions[stub_id] for stub_id in macro.output_node_ids if stub_id in positions]
        callouts.append((child, anchors))
    spec.width, spec.height, _centers = _attach_callouts(spec, callouts, host_width, host_height, depth + 1)
    return _Built(spec=spec, output_nodes=output_nodes)


def _build_comp(comp: CompositionGenome, *, resolve: ResolveFn | None, budget: _Budget, depth: int, stack: tuple[str, ...]) -> _Built:
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
            child = _build_ref(node.ref, resolve=resolve, budget=budget, depth=depth, stack=stack)
            callouts.append((child, [positions[node.id]]))
    spec.width, spec.height, _centers = _attach_callouts(spec, callouts, host_width, host_height, depth + 1)
    return _Built(spec=spec, output_nodes=output_nodes)


# --- public builders -------------------------------------------------------------------------------


def build_genome_spec(genome: Genome, *, resolve: ResolveFn | None = None, node_budget: int = DEFAULT_NODE_BUDGET) -> RenderSpec:
    return _build_genome(genome, resolve=resolve, budget=_Budget(node_budget), depth=0, stack=()).spec


def build_composition_spec(comp: CompositionGenome, *, resolve: ResolveFn | None = None, node_budget: int = DEFAULT_NODE_BUDGET) -> RenderSpec:
    return _build_comp(comp, resolve=resolve, budget=_Budget(node_budget), depth=0, stack=()).spec


def build_entry_spec(entry: LibraryEntry, *, resolve: ResolveFn | None = None, node_budget: int = DEFAULT_NODE_BUDGET) -> RenderSpec:
    try:
        if entry.entry_type == MODULE:
            return build_genome_spec(genome_from_dict(entry.payload), resolve=resolve, node_budget=node_budget)
        if entry.entry_type == COMPOSITION:
            return build_composition_spec(comp_from_dict(entry.payload), resolve=resolve, node_budget=node_budget)
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


def render_network(directory: Path, genome: Genome, *, title: str, library: ModuleLibrary | None = None) -> Path:
    """Render a flat genome (macros expand into callouts when a library is supplied) to `net.png`."""
    resolve = library_resolver(library) if library is not None else None
    return _render_spec_png(directory / "net.png", build_genome_spec(genome, resolve=resolve), title)


def render_composition_network(directory: Path, comp: CompositionGenome, *, title: str, library: ModuleLibrary | None = None) -> Path:
    """Render a composition (module refs expand into callouts when a library is supplied) to `net.png`."""
    resolve = library_resolver(library) if library is not None else None
    return _render_spec_png(directory / "net.png", build_composition_spec(comp, resolve=resolve), title)


def render_entry(out_path: Path, entry: LibraryEntry, *, library: ModuleLibrary | None = None, node_budget: int = DEFAULT_NODE_BUDGET) -> Path:
    resolve = library_resolver(library) if library is not None else None
    spec = build_entry_spec(entry, resolve=resolve, node_budget=node_budget)
    return _render_spec_png(out_path, spec, f"{entry.key}  L{entry.level} {entry.entry_type}")


# --- gallery ---------------------------------------------------------------------------------------


def _cell_title(summary: dict[str, Any]) -> str:
    inputs = "+".join(str(item["width"]) for item in summary["io"]["inputs"])
    label = f"{summary['key']}  L{summary['level']} {summary['entry_type'][0]}  {inputs}->{summary['io']['output']['width']}  m={summary.get('accepted_metric', 0.0):.2f}"
    return f"{label}  retired" if summary.get("retired", False) else label


def render_library_gallery(library: ModuleLibrary, out_path: Path, *, columns: int = 4, include_retired: bool = False, include_dependencies: bool = True) -> Path:
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
                spec = build_entry_spec(library.load(summary["key"]), resolve=resolve, node_budget=GALLERY_NODE_BUDGET)
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


_ROW_GAP = 1.6  # vertical gap between grid rows; leaves room for the container label above a box
_BAND_GAP = 2.5  # clearance between the input/output bands and the grid
_BAND_H = 1.6  # band strip: node row plus its signature label
_LEGEND_ROW_STEP = 1.1
_LEGEND_WIDTH = 16.0
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
        if vertex.key:
            allowance = min(cell_node_budget, budget.remaining)
            cell_budget = _Budget(allowance)
            child = _build_ref(f"library:{vertex.key}", resolve=resolve, budget=cell_budget, depth=0, stack=())
            budget.remaining -= allowance - cell_budget.remaining
        else:
            child = _opaque_built(vertex.label)
        child.label = vertex.label
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
    # prior). A self-transition is recurrent expert use and loops back to that card's own anchor.
    # Retired vertices carry no pathways.
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


def render_overmind(out_path: Path, view: OvermindView, *, library: ModuleLibrary | None = None, node_budget: int = DEFAULT_NODE_BUDGET) -> Path:
    """Render the entire routed model (every expert embedded, wired to the shared bus) to one PNG.
    Intended for `<library_dir>/images/overmind.png`, refreshed whenever the model grows."""
    resolve = library_resolver(library) if library is not None else None
    spec = build_overmind_spec(view, resolve=resolve, node_budget=node_budget)
    live = sum(1 for vertex in view.vertices if not vertex.retired)
    title = f"overmind: {live} experts, d_model={view.d_model}, top_k={view.top_k}, steps={view.max_steps}"
    return _render_spec_png(out_path, spec, title, dpi=_OVERMIND_DPI, x_padding=_OVERMIND_X_PADDING)
