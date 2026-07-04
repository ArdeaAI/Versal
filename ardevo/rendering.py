"""Artistic recursive renders of evolved networks: the full nested detail, dark, in one image.

Nested networks render as CALLOUTS: the host network keeps its natural compact layout (a macro's
output stubs and a composition's module nodes stay in place as green footprint nodes), and each
referenced network is drawn fully inside a translucent container packed into rows across the top of
the frame, with a green line from the box to the footprint nodes it occupies. Callouts recurse
(depth- and budget-guarded): a nested network's own callouts ride along inside its box.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ardevo.evolution.composition import CompNodeKind, CompositionGenome, comp_from_dict, comp_topological_order
from ardevo.evolution.genome import Genome, NodeKind, genome_from_dict, macro_implied_edges, make_acyclic, topological_order
from ardevo.library import COMPOSITION, MODULE, LibraryEntry, ModuleLibrary

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
    "node_input": "#7aa2f7",
    "node_bias": "#566190",
    "node_output": "#f7768e",
    "node_module": "#6fd08c",
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
class RenderSpec:
    nodes: list[SpecNode] = field(default_factory=list)
    edges: list[SpecEdge] = field(default_factory=list)
    containers: list[SpecContainer] = field(default_factory=list)
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
    everything at placement; nothing crosses the boundary except the green callout line."""

    spec: RenderSpec
    label: str = ""
    opaque: bool = False

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


@dataclass(slots=True)
class _Item:
    """One placeable node cell in the layered host layout."""

    key: int
    layer: int
    preds: list[int]
    sort_rank: tuple[int, int]


def _place_items(items: list[_Item]) -> tuple[dict[int, tuple[float, float]], float, float]:
    """Layered layout: x = layer column, y = stacked with barycenter ordering to reduce crossings.

    The frame is [0, width] x [0, height] with every column vertically centered. Returns
    (center positions, width, height)."""
    by_layer: dict[int, list[_Item]] = {}
    for item in items:
        by_layer.setdefault(item.layer, []).append(item)
    layers = sorted(by_layer)
    if not layers:
        return {}, 1.0, 1.0

    col_x = {k: index * (1.0 + _H_GAP) + 0.5 for index, k in enumerate(layers)}
    height = max(len(by_layer[k]) + _V_GAP * (len(by_layer[k]) - 1) for k in layers)
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
        total = len(column) + _V_GAP * (len(column) - 1)
        cursor = height / 2 + total / 2
        for item in column:
            centers[item.key] = (col_x[k], cursor - 0.5)
            cursor -= 1.0 + _V_GAP

    width = col_x[layers[-1]] + 0.5
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
    spec.containers.append(SpecContainer(cx - half_w - _PAD, cy - half_h - _PAD, cx + half_w + _PAD, cy + half_h + _PAD, label=child.label, depth=depth, opaque=child.opaque))


def _attach_callouts(spec: RenderSpec, callouts: list[tuple[_Built, list[tuple[float, float]]]], host_width: float, host_height: float, depth: int) -> tuple[float, float]:
    """Pack the expanded child boxes into rows across the TOP of the host frame and draw a green
    line from each box to the footprint nodes it occupies down in the host network. Returns the
    combined (width, height) of host + callout band."""
    if not callouts:
        return host_width, host_height
    boxes = [(child.spec.width + 2 * _PAD, child.spec.height + 2 * _PAD) for child, _ in callouts]
    available = max(host_width, max(box_w for box_w, _ in boxes))

    # Center the host under a wider callout band so the green lines hang symmetrically.
    host_shift = (available - host_width) / 2
    if host_shift > 0:
        _Built(spec=spec).translate(host_shift, 0.0)
        callouts = [(child, [(x + host_shift, y) for x, y in anchors]) for child, anchors in callouts]

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

    row_base = host_height + _CALLOUT_GAP
    for row in rows:
        row_height = max(boxes[i][1] for i in row)
        row_width = sum(boxes[i][0] for i in row) + _H_GAP * (len(row) - 1)
        x_cursor = (available - row_width) / 2  # center each row over the host
        for i in row:
            box_w, box_h = boxes[i]
            child, anchors = callouts[i]
            cx, cy = x_cursor + box_w / 2, row_base + box_h / 2
            _place_child(spec, child, (cx, cy), depth)
            for anchor_x, anchor_y in anchors:
                spec.edges.append(SpecEdge(cx, row_base, anchor_x, anchor_y, width=1.0, color=THEME["edge_callout"], alpha=0.55))
            x_cursor += box_w + _H_GAP
        row_base += row_height + _V_GAP

    return available, row_base - _V_GAP


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
        spec.nodes.append(SpecNode(x, y, color, size=size, marker=marker, alpha=0.25 if isolated else alpha))

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
    spec.width, spec.height = _attach_callouts(spec, callouts, host_width, host_height, depth + 1)
    return _Built(spec=spec)


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
        spec.nodes.append(SpecNode(x, y, color, size=size, marker=marker))

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
    spec.width, spec.height = _attach_callouts(spec, callouts, host_width, host_height, depth + 1)
    return _Built(spec=spec)


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


def draw_spec(axis: Any, spec: RenderSpec, *, title: str | None = None) -> None:
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

    for style in ("solid", "dashed"):
        group = [edge for edge in spec.edges if edge.curve == 0.0 and edge.style == style]
        if group:
            segments = [((edge.x0, edge.y0), (edge.x1, edge.y1)) for edge in group]
            rgba = [mcolors.to_rgba(edge.color, edge.alpha) for edge in group]
            axis.add_collection(LineCollection(segments, colors=rgba, linewidths=[edge.width for edge in group], linestyle=style, zorder=2))
    for edge in spec.edges:
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

    axis.set_xlim(-_PAD, spec.width + _PAD)
    axis.set_ylim(-_PAD, spec.height + _PAD)
    axis.set_aspect("equal")
    axis.axis("off")
    if title is not None:
        axis.set_title(title, fontsize=11, color=THEME["title"])


def _render_spec_png(out_path: Path, spec: RenderSpec, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_w = min(max(spec.width * 0.6, 6.0), 30.0)
    fig_h = min(max(spec.height * 0.6, 4.5), 30.0)
    figure, axis = plt.subplots(figsize=(fig_w, fig_h))
    figure.patch.set_facecolor(THEME["background"])
    draw_spec(axis, spec, title=title)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, facecolor=figure.get_facecolor())
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


# --- overmind: the whole routed model in one frame -------------------------------------------------


@dataclass(slots=True)
class OvermindVertex:
    """A read-only view of one routed-model expert, so rendering stays decoupled from `routing.py`."""

    key: str  # the library key (or "" for a synthetic vertex); resolved to its embedded network
    label: str  # display label (usage share, retired marker, ...)
    retired: bool = False


@dataclass(slots=True)
class OvermindView:
    """Everything the overmind render needs about a routed model, as plain data. `routing.py` builds
    this from a live `RoutedNet`; the renderer never imports the router."""

    vertices: list[OvermindVertex]
    input_signatures: list[str]  # one per input adapter (e.g. "BINARY|K:2")
    output_signatures: list[str]  # one per output head
    d_model: int
    top_k: int
    max_steps: int


def build_overmind_spec(view: OvermindView, *, resolve: ResolveFn | None = None, node_budget: int = DEFAULT_NODE_BUDGET) -> RenderSpec:
    """The whole routed model in one spec: the shared bus (input adapters -> gate -> output heads)
    is the HOST network, and every expert is a fully-embedded CALLOUT wired to the gate, reusing the
    exact recursion that draws a composition's referenced modules. Growth just adds callouts.

    Never raises: an unresolvable/oversized expert degrades to a labeled opaque box, like everywhere."""
    spec = RenderSpec()
    # The bus host: input-adapter nodes (left), the gate hub (center), output-head nodes (right),
    # laid out as three columns in the shared frame, wired input -> gate -> output.
    inputs = view.input_signatures or ["(no input adapter yet)"]
    outputs = view.output_signatures or ["(no output head yet)"]
    column_gap = 3.0 + _H_GAP
    host_height = max(len(inputs), len(outputs), 1) * (1.0 + _V_GAP)
    gate_pos = (column_gap, host_height / 2)

    def _column(signatures: list[str], x: float, color: str, marker: str) -> list[tuple[float, float]]:
        total = len(signatures) + _V_GAP * (len(signatures) - 1)
        cursor = host_height / 2 + total / 2 - 0.5
        placed: list[tuple[float, float]] = []
        for _signature in signatures:
            spec.nodes.append(SpecNode(x, cursor, color=color, size=1.4, marker=marker))
            placed.append((x, cursor))
            cursor -= 1.0 + _V_GAP
        return placed

    input_positions = _column(inputs, 0.0, THEME["node_input"], "s")
    output_positions = _column(outputs, 2 * column_gap, THEME["node_output"], "s")
    spec.nodes.append(SpecNode(gate_pos[0], gate_pos[1], color=THEME["node_module"], size=2.2, marker="D"))  # the gate hub
    for position in input_positions:
        spec.edges.append(SpecEdge(position[0], position[1], gate_pos[0], gate_pos[1], width=1.4, color=THEME["edge_glue"], alpha=0.6))
    for position in output_positions:
        spec.edges.append(SpecEdge(gate_pos[0], gate_pos[1], position[0], position[1], width=1.4, color=THEME["edge_glue"], alpha=0.6))
    host_width = 2 * column_gap

    budget = _Budget(node_budget)
    callouts: list[tuple[_Built, list[tuple[float, float]]]] = []
    for vertex in view.vertices:
        child = _build_ref(f"library:{vertex.key}", resolve=resolve, budget=budget, depth=0, stack=()) if vertex.key else _opaque_built(vertex.label)
        child.label = vertex.label
        child.opaque = child.opaque or vertex.retired  # retired experts read as opaque footprints
        callouts.append((child, [gate_pos]))
    spec.width, spec.height = _attach_callouts(spec, callouts, host_width, host_height, depth=1)
    return spec


def render_overmind(out_path: Path, view: OvermindView, *, library: ModuleLibrary | None = None, node_budget: int = DEFAULT_NODE_BUDGET) -> Path:
    """Render the entire routed model (every expert embedded, wired to the shared bus) to one PNG.
    Intended for `<library_dir>/images/overmind.png`, refreshed whenever the model grows."""
    resolve = library_resolver(library) if library is not None else None
    spec = build_overmind_spec(view, resolve=resolve, node_budget=node_budget)
    live = sum(1 for vertex in view.vertices if not vertex.retired)
    title = f"overmind: {live} experts, d_model={view.d_model}, top_k={view.top_k}, steps={view.max_steps}"
    return _render_spec_png(out_path, spec, title)
