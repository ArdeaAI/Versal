"""Density portraits for large module payloads."""

import heapq
import math
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast

from versal.evolution.genome import NodeKind
from versal.library import LibraryEntry
from versal.rendering import (
    _DENSITY_HEIGHT,
    _DENSITY_WIDTH,
    THEME,
    _DensityLayout,
    _DensityPanel,
    _FlowStats,
    _LargeRenderMetadata,
)


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
