"""Routed overmind layout and pruning."""

import math
from typing import Any

from versal.library import LibraryEntry
from versal.reference_depth import DEFAULT_MAX_INLINE_DEPTH
from versal.rendering import (
    _H_GAP,
    _NETWORK_ANCHOR_INSET,
    _PAD,
    DEFAULT_NODE_BUDGET,
    THEME,
    OvermindView,
    RenderSpec,
    ResolveFn,
    SpecContainer,
    SpecEdge,
    SpecNode,
    SpecText,
    _Budget,
    _build_entry,
    _Built,
    _edge_width,
    _entry_size_label,
    _layer_color,
    _opaque_built,
    _place_child,
)

_ROW_GAP = 2.4
_BAND_GAP = 4
_BAND_H = 1.6
_LEGEND_COLUMNS = 2
_LEGEND_ROW_STEP = 1.45
_LEGEND_WIDTH = 22.0
_OVERMIND_COLUMNS = 8


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
        traffic_observed=view.traffic_observed,
    )


def _overmind_legend_entries(*, traffic_observed: bool = True) -> list[tuple[str, dict[str, Any], str]]:
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
        (
            "edge",
            {"color": THEME["edge_pathway"], "curve": 0.25},
            "routing traffic (observed)" if traffic_observed else "routing potential (cold structural view)",
        ),
        ("edge", {"color": THEME["edge_entry"], "alpha": 0.6}, "input feed (step-0 gate mass)"),
        ("edge", {"color": THEME["edge_exit"], "alpha": 0.6}, "output feed (final-step gate mass)"),
    ]


def _adaptive_overmind_legend_entries(
    children: list[_Built],
    view: OvermindView,
    *,
    traffic_observed: bool,
) -> list[tuple[str, dict[str, Any], str]]:
    """Return only legend rows represented by the focused current-only portrait."""
    nodes = [node for child in children for node in child.spec.nodes]
    edges = [edge for child in children for edge in child.spec.edges]
    node_roles = {node.role for node in nodes}
    edge_roles = {edge.role for edge in edges}
    hidden_colors = {node.color for node in nodes if node.role == "hidden"}
    labels = {"input", "output"}
    if "bias" in node_roles:
        labels.add("bias")
    if hidden_colors:
        labels.add("hidden (early layer)")
    if len(hidden_colors) > 1:
        labels.add("hidden (deep layer)")
    if any(node.marker == "D" for node in nodes):
        labels.add("product gate")
    if node_roles & {"macro-footprint", "module-footprint"}:
        labels.add("module ref / macro footprint")
    if view.vertices:
        labels.add("network input anchor")
        labels.update({"input feed (step-0 gate mass)", "output feed (final-step gate mass)"})
    if "isolated" in node_roles:
        labels.add("isolated (unused)")
    if any(child.opaque for child in children):
        labels.add("retired or unexpanded network")
    if any(role.startswith("forward") for role in edge_roles):
        labels.add("forward connection")
    if "recurrent" in edge_roles:
        labels.add("recurrent (time-delayed)")
    if "macro-implied" in edge_roles:
        labels.add("macro implied wiring")
    if any(role.startswith("composition-glue") for role in edge_roles):
        labels.add("composition glue")
    if "nested-network" in edge_roles:
        labels.add("nested-network flow")
    if view.pathways:
        labels.add("routing traffic (observed)" if traffic_observed else "routing potential (cold structural view)")
    return [entry for entry in _overmind_legend_entries(traffic_observed=traffic_observed) if entry[2] in labels]


def _overmind_legend(spec: RenderSpec, x0: float, y_top: float, entries: list[tuple[str, dict[str, Any], str]]) -> float:
    """Append a compact two-column key panel at the right margin; returns its width."""
    spec.texts.append(SpecText(x0 + 0.6, y_top - 0.6, "key", size=8.0, color=THEME["title"]))
    item_width = _LEGEND_WIDTH / _LEGEND_COLUMNS
    for index, (kind, params, label) in enumerate(entries):
        column, row = divmod(index, math.ceil(len(entries) / _LEGEND_COLUMNS))
        item_x = x0 + column * item_width
        y = y_top - 1.4 - row * _LEGEND_ROW_STEP
        if kind == "node":
            spec.nodes.append(SpecNode(item_x + 1.0, y, color=params["color"], size=params.get("size", 1.2), marker=params.get("marker", "o"), alpha=params.get("alpha", 1.0)))
        elif kind == "edge":
            spec.edges.append(
                SpecEdge(
                    item_x + 0.4,
                    y,
                    item_x + 1.8,
                    y,
                    width=1.6,
                    color=params["color"],
                    style=params.get("style", "solid"),
                    curve=params.get("curve", 0.0),
                    alpha=params.get("alpha", 0.9),
                    role="legend",
                )
            )
        else:
            spec.containers.append(SpecContainer(item_x + 0.3, y - 0.35, item_x + 1.9, y + 0.35, label="", depth=1, opaque=True))
        spec.texts.append(SpecText(item_x + 2.4, y, label, size=5.7))
    rows = math.ceil(len(entries) / _LEGEND_COLUMNS)
    bottom = y_top - 1.4 - max(rows - 1, 0) * _LEGEND_ROW_STEP - 0.8
    spec.containers.append(SpecContainer(x0, bottom, x0 + _LEGEND_WIDTH, y_top, label="", depth=0))
    return _LEGEND_WIDTH


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
    """Build a top-down adapter, expert-grid, and output-head portrait.

    ``cell_node_budget`` prevents one large expert from distorting the entire grid.
    """
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

    legend_entries = (
        _adaptive_overmind_legend_entries(children, view, traffic_observed=view.traffic_observed)
        if legend and legend_mode == "adaptive"
        else (_overmind_legend_entries(traffic_observed=view.traffic_observed) if legend else [])
    )
    legend_height = 2.2 + _LEGEND_ROW_STEP * math.ceil(len(legend_entries) / _LEGEND_COLUMNS)
    core_height = 2 * (_BAND_H + _BAND_GAP) + grid_height
    legend_below = bool(legend_entries) and legend_mode == "adaptive" and grid_width < _LEGEND_WIDTH
    content_y_shift = legend_height + _ROW_GAP if legend_below else 0.0
    content_x_shift = (_LEGEND_WIDTH - grid_width) / 2 if legend_below else 0.0
    total_height = core_height + content_y_shift if legend_below else max(core_height, legend_height)

    bottoms: list[tuple[float, float]] = [(0.0, 0.0)] * len(children)
    anchors: list[tuple[float, float]] = [(0.0, 0.0)] * len(children)
    content_top = content_y_shift + core_height
    y_cursor = content_top - _BAND_H - _BAND_GAP  # y-up frame: the top rail of the first row
    for row, row_height, row_width in zip(rows, row_heights, row_widths):
        x_cursor = content_x_shift + (grid_width - row_width) / 2
        for i in row:
            box_w, box_h = boxes[i]
            cx, cy = x_cursor + box_w / 2, y_cursor - box_h / 2  # top-aligned: a clean rail for input feeds
            _place_child(spec, children[i], (cx, cy), depth=1)
            bottoms[i] = (cx, y_cursor - box_h)
            anchors[i] = (cx - box_w / 2 + _NETWORK_ANCHOR_INSET, y_cursor - _NETWORK_ANCHOR_INSET)
            spec.nodes.append(SpecNode(*anchors[i], color=THEME["node_anchor"], size=0.65, role="network-input-anchor"))
            if not children[i].opaque and box_w < 2.5:
                # draw_spec skips container labels on narrow boxes; every cell still deserves a name
                spec.texts.append(SpecText(cx - box_w / 2 + 0.15, y_cursor + 0.08, children[i].label, size=5.0, va="bottom"))
            x_cursor += box_w + _H_GAP
        y_cursor -= row_height + _ROW_GAP

    def _band(signatures: list[str], y: float, color: str) -> list[tuple[float, float]]:
        placed: list[tuple[float, float]] = []
        for index, signature in enumerate(signatures):
            x = content_x_shift + grid_width * (index + 1) / (len(signatures) + 1)
            spec.nodes.append(SpecNode(x, y, color=color, size=1.4, marker="s", role="input-adapter" if color == THEME["node_input"] else "output-head"))
            spec.texts.append(SpecText(x, y - 0.45, signature, size=6.0, ha="center", va="top"))
            placed.append((x, y))
        return placed

    input_positions = _band(inputs, content_top - 0.5, THEME["node_input"])
    output_positions = _band(outputs, content_y_shift + 0.5, THEME["node_output"])

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
        alpha = 0.6
        if entry_width > 0.0:
            for x, y in input_positions:
                spec.edges.append(
                    SpecEdge(
                        x,
                        y,
                        anchors[index][0],
                        anchors[index][1],
                        width=entry_width,
                        color=THEME["edge_entry"],
                        alpha=alpha,
                        role="routing-entry",
                        magnitude=max(vertex.entry_share, 0.05),
                    )
                )
        if exit_width > 0.0:
            for source_x, source_y in _outputs(index):
                for x, y in output_positions:
                    spec.edges.append(
                        SpecEdge(
                            source_x,
                            source_y,
                            x,
                            y,
                            width=exit_width,
                            color=THEME["edge_exit"],
                            alpha=alpha,
                            role="routing-exit",
                            magnitude=max(vertex.exit_share, 0.05),
                        )
                    )

    # Pathways connect expert outputs to target anchors; self-transitions loop to the same card.
    for source, target, weight in view.pathways:
        if not (0 <= source < len(children) and 0 <= target < len(children)):
            continue
        if view.vertices[source].retired or view.vertices[target].retired:
            continue
        x1, y1 = anchors[target]
        for x0, y0 in _outputs(source):
            spec.edges.append(
                SpecEdge(
                    x0,
                    y0,
                    x1,
                    y1,
                    width=_edge_width(weight * 3),
                    color=THEME["edge_pathway"],
                    curve=0.25,
                    alpha=0.25 + 0.6 * min(weight, 1.0),
                    role="routing-observed" if view.traffic_observed else "routing-potential",
                    magnitude=weight,
                )
            )

    spec.width = max(grid_width, _LEGEND_WIDTH) if legend_below else grid_width
    if legend_below:
        _overmind_legend(spec, 0.0, legend_height - 0.4, legend_entries)
    elif legend:
        spec.width = grid_width + 2 * _H_GAP + _overmind_legend(spec, grid_width + 2 * _H_GAP, total_height - 0.5, legend_entries)
    spec.height = total_height
    spec.flow_label = (
        "observed routing flow · gate-mass traffic, not activation analysis"
        if view.traffic_observed
        else "routing potential · cold structural view, not observed traffic or activations"
    )
    return spec
