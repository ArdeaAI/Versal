"""Recursive render specs: nested networks expand inline, every failure degrades to an opaque box."""

import random
from itertools import product
from pathlib import Path

import pytest

from versal.evolution.composition import minimal_composition
from versal.evolution.genome import ConnectionGene, Genome, InnovationTracker, MacroGene, NodeGene, NodeKind, genome_to_dict
from versal.library import COMPOSITION, MODULE, LibraryEntry, ModuleLibrary
from versal.rendering import (
    RENDER_MAX_DEPTH,
    THEME,
    build_entry_spec,
    build_genome_spec,
    library_resolver,
    render_composition_network,
    render_entry,
    render_library_gallery,
    render_network,
)

_IO = {"inputs": [{"signature": "BINARY|K", "width": 2}], "output": {"signature": "BINARY|K", "width": 1}}
_FIXTURE_LIBRARY = Path(__file__).parent / "fixtures" / "library_v1"


def _macro_host(key: str) -> Genome:
    """2 inputs + bias -> macro(ref) -> output; node 4 is the macro's output stub."""
    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity"),
        1: NodeGene(1, NodeKind.INPUT, "identity"),
        2: NodeGene(2, NodeKind.BIAS, "identity"),
        3: NodeGene(3, NodeKind.OUTPUT, "identity"),
        4: NodeGene(4, NodeKind.HIDDEN, "identity"),
    }
    connections = [ConnectionGene(4, 3, 1.0, True, 0)]
    macros = [MacroGene(ref=f"library:{key}", input_node_ids=(0, 1), output_node_ids=(4,), innovation=100)]
    return Genome(nodes=nodes, connections=connections, macros=macros)


def _entry(key: str, payload: dict) -> LibraryEntry:
    return LibraryEntry(key=key, entry_type=MODULE, level=1, io=_IO, payload=payload, weights_frozen=True, provenance={})


def _spatial_entry(key: str, descriptors: list[tuple[str, tuple[int, ...]]]) -> LibraryEntry:
    nodes: list[dict] = []
    io_inputs: list[dict] = []
    node_id = 0
    for signature, shape in descriptors:
        width = 1
        for size in shape:
            width *= size
        io_inputs.append({"signature": signature, "width": width})
        for coordinate in product(*(range(size) for size in shape)):
            nodes.append({"id": node_id, "kind": "input", "activation": "identity", "coordinate": list(coordinate), "aggregation": "sum"})
            node_id += 1
    nodes.extend(
        [
            {"id": node_id, "kind": "bias", "activation": "identity", "coordinate": None, "aggregation": "sum"},
            {"id": node_id + 1, "kind": "hidden", "activation": "tanh", "coordinate": None, "aggregation": "product"},
            {"id": node_id + 2, "kind": "output", "activation": "identity", "coordinate": None, "aggregation": "sum"},
        ]
    )
    connections = [
        {"in": 0, "out": node_id + 1, "weight": 1.0, "enabled": True, "innovation": 0, "recurrent": False},
        {"in": 1, "out": node_id + 1, "weight": -0.5, "enabled": True, "innovation": 1, "recurrent": False},
        {"in": node_id, "out": node_id + 1, "weight": 0.25, "enabled": True, "innovation": 2, "recurrent": False},
        {"in": node_id + 1, "out": node_id + 2, "weight": 0.75, "enabled": True, "innovation": 3, "recurrent": False},
        {"in": node_id + 2, "out": node_id + 1, "weight": 0.2, "enabled": True, "innovation": 4, "recurrent": True},
    ]
    return LibraryEntry(
        key=key,
        entry_type=MODULE,
        level=1,
        io={"inputs": io_inputs, "output": {"signature": "CONTINUOUS|C", "width": 1}},
        payload={"nodes": nodes, "connections": connections, "macros": []},
        weights_frozen=True,
        provenance={},
    )


def _render_chain() -> dict[str, LibraryEntry]:
    leaf = Genome(
        nodes={0: NodeGene(0, NodeKind.INPUT, "identity"), 1: NodeGene(1, NodeKind.OUTPUT, "identity")},
        connections=[ConnectionGene(0, 1, 1.0, True, 0)],
    )
    chain: dict[str, LibraryEntry] = {}
    payload = genome_to_dict(leaf)
    for index in reversed(range(6)):  # d0 embeds d1 embeds ... embeds d5 (the leaf)
        chain[f"d{index}"] = _entry(f"d{index}", payload)
        payload = genome_to_dict(_macro_host(f"d{index}"))
    return chain


# --- spec builders ---------------------------------------------------------------------------------


def test_genome_spec_basic(solving_genome: Genome) -> None:
    spec = build_genome_spec(solving_genome)
    assert spec.node_count == len(solving_genome.nodes)
    assert spec.containers == []
    assert len(spec.edges) == len(solving_genome.enabled_connections())
    assert spec.width > 0 and spec.height > 0
    assert spec.flow_label == "potential influence flow · weights/topology, not activations"


def test_genome_spec_preserves_signed_and_recurrent_influence_semantics() -> None:
    genome = Genome(
        nodes={
            0: NodeGene(0, NodeKind.INPUT, "identity"),
            1: NodeGene(1, NodeKind.HIDDEN, "tanh"),
            2: NodeGene(2, NodeKind.OUTPUT, "identity"),
        },
        connections=[
            ConnectionGene(0, 1, 0.75, True, 0),
            ConnectionGene(0, 2, -0.5, True, 1),
            ConnectionGene(2, 1, 0.25, True, 2, recurrent=True),
        ],
    )

    spec = build_genome_spec(genome)
    by_role = {edge.role: edge for edge in spec.edges}

    assert by_role["forward-positive"].color == THEME["edge_positive"]
    assert by_role["forward-positive"].signed_weight == 0.75
    assert by_role["forward-negative"].color == THEME["edge_negative"]
    assert by_role["forward-negative"].signed_weight == -0.5
    assert by_role["recurrent"].color == THEME["edge_recurrent"]
    assert by_role["recurrent"].style == "dashed"


def test_genome_spec_expands_macro_as_callout(tmp_path: Path, solving_genome: Genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    key = library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=_IO, provenance={"accepted_metric": 1.0})
    spec = build_genome_spec(_macro_host(key), resolve=library_resolver(library))

    assert len(spec.containers) == 1
    container = spec.containers[0]
    assert not container.opaque
    assert key in container.label and "L1" in container.label
    # All 5 host nodes (the stub stays, as the macro's green footprint), the inner nodes, and the
    # gold input anchor on the nested network card.
    assert spec.node_count == 5 + len(solving_genome.nodes) + 1
    # Inner edges + the host readout (stub -> output) + 2 implied macro edges + 1 green callout line.
    assert len(spec.edges) == len(solving_genome.enabled_connections()) + 1 + 2 + 1
    assert any(edge.color == THEME["edge_macro"] for edge in spec.edges)
    # The callout box sits ABOVE the host network, green-lined to the footprint stub.
    callout_lines = [edge for edge in spec.edges if edge.color == THEME["edge_callout"]]
    footprints = [node for node in spec.nodes if node.color == THEME["node_module"]]
    anchors = [node for node in spec.nodes if node.color == THEME["node_anchor"]]
    assert len(callout_lines) == 1 and len(footprints) == 1
    assert len(anchors) == 1
    assert (anchors[0].x, anchors[0].y) == pytest.approx((container.x0 + 0.3, container.y1 - 0.3))
    assert (callout_lines[0].x0, callout_lines[0].y0) == pytest.approx((footprints[0].x, footprints[0].y))
    assert (callout_lines[0].x1, callout_lines[0].y1) == pytest.approx((anchors[0].x, anchors[0].y))
    assert container.y0 > footprints[0].y


def test_macro_missing_ref_is_opaque_box(solving_genome: Genome) -> None:
    spec = build_genome_spec(_macro_host("m1_gone"), resolve=lambda key: None)
    assert len(spec.containers) == 1 and spec.containers[0].opaque
    assert spec.node_count == 6  # host incl. footprint stub + the opaque nested card's input anchor
    assert len(spec.edges) == 1 + 2 + 1  # readout + implied macro edges + the green callout line


def test_cycle_guard_stops_expansion() -> None:
    loop_genome = Genome(
        nodes={0: NodeGene(0, NodeKind.INPUT, "identity"), 1: NodeGene(1, NodeKind.OUTPUT, "identity"), 2: NodeGene(2, NodeKind.HIDDEN, "identity")},
        connections=[ConnectionGene(2, 1, 1.0, True, 0)],
        macros=[MacroGene(ref="library:loop", input_node_ids=(0,), output_node_ids=(2,), innovation=7)],
    )
    payload = genome_to_dict(loop_genome)
    spec = build_genome_spec(_macro_host("loop"), resolve=lambda key: _entry("loop", payload) if key == "loop" else None)
    assert any(container.opaque for container in spec.containers)
    assert len(spec.containers) <= RENDER_MAX_DEPTH + 1


def test_depth_guard_stops_expansion() -> None:
    chain = _render_chain()

    def resolve(key: str) -> LibraryEntry | None:
        return chain.get(key)

    spec = build_genome_spec(_macro_host("d0"), resolve=lambda key: chain.get(key))
    assert any(container.opaque for container in spec.containers)
    assert sum(1 for container in spec.containers if not container.opaque) == RENDER_MAX_DEPTH
    shallow = build_genome_spec(_macro_host("d0"), resolve=resolve, max_inline_depth=2)
    assert sum(1 for container in shallow.containers if not container.opaque) == 2
    deep = build_genome_spec(_macro_host("d0"), resolve=resolve, max_inline_depth=6)
    assert sum(1 for container in deep.containers if not container.opaque) == 6
    assert not any(container.opaque for container in deep.containers)


def test_overmind_card_root_does_not_consume_reference_depth() -> None:
    from versal.rendering import OvermindVertex, OvermindView, build_overmind_spec

    chain = _render_chain()

    def resolve(key: str) -> LibraryEntry | None:
        return chain.get(key)

    entry_spec = build_entry_spec(chain["d0"], resolve=resolve, max_inline_depth=5)
    assert sum(1 for container in entry_spec.containers if not container.opaque) == 5
    assert not any(container.opaque for container in entry_spec.containers)

    view = OvermindView(vertices=[OvermindVertex(key="d0", label="root")], input_signatures=[], output_signatures=[], d_model=8, top_k=1, max_steps=1)
    overmind = build_overmind_spec(view, resolve=resolve, legend=False, max_inline_depth=5)
    assert sum(1 for container in overmind.containers if not container.opaque) == 6  # root card plus five followed refs
    assert not any(container.opaque for container in overmind.containers)


def test_budget_falls_back_opaque(tmp_path: Path, solving_genome: Genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    key = library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=_IO, provenance={})
    spec = build_genome_spec(_macro_host(key), resolve=library_resolver(library), node_budget=5)
    assert len(spec.containers) == 1 and spec.containers[0].opaque


def test_root_budget_fallback_is_a_labeled_opaque_summary(solving_genome: Genome) -> None:
    entry = _entry("wide", genome_to_dict(solving_genome))
    spec = build_entry_spec(entry, node_budget=1)

    assert spec.node_count == 0
    assert len(spec.containers) == 1 and spec.containers[0].opaque
    assert "wide  L1" in spec.containers[0].label
    assert f"{len(solving_genome.nodes):,} nodes" in spec.containers[0].label
    assert f"{len(solving_genome.connections):,} edges" in spec.containers[0].label


def test_garbage_payload_is_opaque() -> None:
    spec = build_genome_spec(_macro_host("bad"), resolve=lambda key: _entry("bad", {"nodes": "garbage"}))
    assert len(spec.containers) == 1 and spec.containers[0].opaque
    assert spec.containers[0].label.endswith("?")


def test_composition_spec_expands_nested_chain() -> None:
    library = ModuleLibrary(_FIXTURE_LIBRARY)
    entry = library.load("c3_87a6e67226b1")
    spec = build_entry_spec(entry, resolve=library_resolver(library))
    assert len(spec.containers) >= 2  # c3 -> c2 -> m1 chain
    assert sum(1 for container in spec.containers if not container.opaque) >= 2
    assert spec.node_count > len(entry.payload["nodes"])
    # Every module footprint node gets a green line up to its callout box.
    assert any(edge.color == THEME["edge_callout"] for edge in spec.edges)
    assert any(node.color == THEME["node_module"] for node in spec.nodes)


def test_recurrent_edge_is_dashed_curved() -> None:
    genome = Genome(
        nodes={0: NodeGene(0, NodeKind.INPUT, "identity"), 1: NodeGene(1, NodeKind.OUTPUT, "tanh")},
        connections=[ConnectionGene(0, 1, 1.0, True, 0), ConnectionGene(1, 0, 0.5, True, 1, recurrent=True)],
    )
    spec = build_genome_spec(genome)
    recurrent_edges = [edge for edge in spec.edges if edge.style == "dashed"]
    assert len(recurrent_edges) == 1
    assert recurrent_edges[0].curve != 0.0
    assert recurrent_edges[0].color == THEME["edge_recurrent"]


def test_product_node_uses_diamond_marker() -> None:
    genome = Genome(
        nodes={
            0: NodeGene(0, NodeKind.INPUT, "identity"),
            1: NodeGene(1, NodeKind.INPUT, "identity"),
            2: NodeGene(2, NodeKind.HIDDEN, "tanh", aggregation="product"),
            3: NodeGene(3, NodeKind.OUTPUT, "identity"),
        },
        connections=[ConnectionGene(0, 2, 1.0, True, 0), ConnectionGene(1, 2, 1.0, True, 1), ConnectionGene(2, 3, 1.0, True, 2)],
    )
    spec = build_genome_spec(genome)
    assert any(node.marker == "D" for node in spec.nodes)


def test_isolated_node_is_faint() -> None:
    genome = Genome(
        nodes={
            0: NodeGene(0, NodeKind.INPUT, "identity"),
            1: NodeGene(1, NodeKind.OUTPUT, "identity"),
            2: NodeGene(2, NodeKind.HIDDEN, "tanh"),  # no enabled edges touch it
        },
        connections=[ConnectionGene(0, 1, 1.0, True, 0)],
    )
    spec = build_genome_spec(genome)
    faint = [node for node in spec.nodes if node.alpha < 1.0 and node.marker == "o"]
    assert len(faint) == 1 and faint[0].size == 0.5


def test_small_genome_layout_pinned() -> None:
    # Pins the pre-wrap column layout: layers below _MAX_COLUMN_NODES must place exactly as before
    # the block-wrap change (fixed 2.6-unit column pitch, vertically centered stacks).
    genome = Genome(
        nodes={
            0: NodeGene(0, NodeKind.INPUT, "identity"),
            1: NodeGene(1, NodeKind.INPUT, "identity"),
            2: NodeGene(2, NodeKind.BIAS, "identity"),
            3: NodeGene(3, NodeKind.OUTPUT, "identity"),
            4: NodeGene(4, NodeKind.HIDDEN, "tanh"),
        },
        connections=[ConnectionGene(0, 4, 1.0, True, 0), ConnectionGene(4, 3, 1.0, True, 1), ConnectionGene(1, 3, 1.0, True, 2)],
    )
    spec = build_genome_spec(genome)
    positions = {(round(node.x, 6), round(node.y, 6)) for node in spec.nodes}
    assert positions == {(0.5, 3.9), (0.5, 2.2), (0.5, 0.5), (3.1, 2.2), (5.7, 2.2)}
    assert (round(spec.width, 6), round(spec.height, 6)) == (6.2, 4.4)


def test_wide_layer_wraps_into_block() -> None:
    # The ARC-scale failure: a 2,000-node output layer must wrap into a near-square block instead of
    # one 3,400-unit column that set_aspect("equal") crushes into a one-pixel vertical line.
    n_in, n_out = 200, 2000
    nodes = {i: NodeGene(i, NodeKind.INPUT, "identity") for i in range(n_in)}
    for j in range(n_out):
        nodes[n_in + j] = NodeGene(n_in + j, NodeKind.OUTPUT, "identity")
    connections = [ConnectionGene(j % n_in, n_in + j, 0.1, True, j) for j in range(n_out)]
    spec = build_genome_spec(Genome(nodes=nodes, connections=connections))

    assert spec.node_count == n_in + n_out
    aspect = max(spec.width, spec.height) / min(spec.width, spec.height)
    assert aspect < 5.0
    output_columns = {round(node.x, 6) for node in spec.nodes if node.color == THEME["node_output"]}
    assert len(output_columns) == 45  # ceil(sqrt(2000)) rows -> 45 sub-columns


def test_draw_spec_rasterizes_every_edge_without_matplotlib_sampling() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    from versal.rendering import _MAX_STRAIGHT_EDGES, RenderSpec, SpecEdge, draw_spec

    count = _MAX_STRAIGHT_EDGES + 10_000
    edges = [SpecEdge(0.0, float(i % 100), 1.0, float(i % 97), width=0.6, color=THEME["edge_forward"]) for i in range(count)]
    figure, axis = plt.subplots()
    draw_spec(axis, RenderSpec(edges=edges, width=2.0, height=100.0, flow_label="potential influence flow · test"))
    drawn = sum(len(artist.get_segments()) for artist in axis.collections if isinstance(artist, LineCollection))
    assert drawn == 0  # dense edges live in one raster image, not per-edge Matplotlib artists
    assert any(f"all {count:,} scene edges included" in text.get_text() for text in axis.texts)
    assert axis.images
    plt.close(figure)


def test_shared_datashader_edge_chunks_include_every_scene_edge(monkeypatch) -> None:
    import numpy as np

    import versal.rendering as rendering

    monkeypatch.setattr(rendering, "_DENSITY_EDGE_CHUNK", 2)
    spec = rendering.RenderSpec(
        edges=[
            rendering.SpecEdge(0.0, 0.0, 1.0, 1.0, 1.0, THEME["edge_positive"], magnitude=1.0),
            rendering.SpecEdge(0.0, 1.0, 1.0, 0.0, 1.0, THEME["edge_negative"], magnitude=2.0),
            rendering.SpecEdge(0.5, 0.0, 0.5, 1.0, 1.0, THEME["edge_recurrent"], curve=0.25, magnitude=0.5),
        ],
        width=1.0,
        height=1.0,
    )

    image, rendered = rendering._rasterized_spec_edges(spec, pixel_width=64, pixel_height=48, x_range=(-0.1, 1.1), y_range=(-0.1, 1.1))

    assert rendered == 3
    assert image.shape == (48, 64, 4)
    assert np.count_nonzero(image[:, :, 3]) > 0


def test_datashader_edge_layer_caps_overlap_at_declared_opacity() -> None:
    import numpy as np

    from versal.rendering import RenderSpec, SpecEdge, _rasterized_spec_edges

    repeated = SpecEdge(0.0, 0.0, 1.0, 1.0, 1.0, THEME["edge_entry"], alpha=0.6, role="routing-entry")
    image, rendered = _rasterized_spec_edges(
        RenderSpec(edges=[repeated] * 100),
        pixel_width=64,
        pixel_height=48,
        x_range=(-0.1, 1.1),
        y_range=(-0.1, 1.1),
    )

    assert rendered == 100
    assert int(np.max(image[:, :, 3])) <= round(0.6 * 255)


# --- png renders -----------------------------------------------------------------------------------


def test_render_network_writes_png(tmp_path: Path, solving_genome: Genome) -> None:
    image_path = render_network(tmp_path, solving_genome, title="test")
    assert image_path.name == "net.png"
    assert image_path.exists() and image_path.stat().st_size > 0


def test_render_network_expands_macro_with_library(tmp_path: Path, solving_genome: Genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    key = library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=_IO, provenance={})
    image_path = render_network(tmp_path, _macro_host(key), title="macro host", library=library)
    assert image_path.exists() and image_path.stat().st_size > 0


def test_render_composition_network_writes_png(tmp_path: Path) -> None:
    comp = minimal_composition([("BINARY|K", 2)], "head", 1, InnovationTracker(_next_node_id=0), random.Random(0))
    image_path = render_composition_network(tmp_path, comp, title="composition")
    assert image_path.exists() and image_path.stat().st_size > 0


def test_render_entry_density_dispatch_threshold_and_compositions(monkeypatch, tmp_path: Path) -> None:
    import versal.rendering as rendering

    density_calls: list[str] = []
    detail_calls: list[str] = []

    def density(path, entry):
        density_calls.append(entry.key)
        return rendering._LargeRenderMetadata(len(entry.payload["nodes"]), 0, 0, 0, "test")

    def detail(entry, **_kwargs):
        detail_calls.append(entry.key)
        return rendering.RenderSpec()

    monkeypatch.setattr(rendering, "_render_large_module_png", density)
    monkeypatch.setattr(rendering, "build_entry_spec", detail)
    monkeypatch.setattr(rendering, "_render_spec_png", lambda path, *_args, **_kwargs: path)

    def bare(key: str, count: int, entry_type: str = MODULE) -> LibraryEntry:
        return LibraryEntry(key, entry_type, 1, _IO, {"nodes": [{}] * count}, True, {})

    rendering.render_entry(tmp_path / "at.png", bare("at", 1500))
    rendering.render_entry(tmp_path / "over.png", bare("over", 1501))
    rendering.render_entry(tmp_path / "custom-at.png", bare("custom-at", 7), node_budget=7)
    rendering.render_entry(tmp_path / "custom-over.png", bare("custom-over", 8), node_budget=7)
    rendering.render_entry(tmp_path / "composition.png", bare("composition", 2000, COMPOSITION))

    assert density_calls == ["over", "custom-over"]
    assert detail_calls == ["at", "custom-at", "composition"]


def test_density_renderer_is_not_used_by_task_gallery_nested_or_overmind_paths(monkeypatch, tmp_path: Path, solving_genome: Genome) -> None:
    import versal.rendering as rendering

    def forbidden(*_args, **_kwargs):
        raise AssertionError("individual-entry density renderer was invoked")

    monkeypatch.setattr(rendering, "_render_large_module_png", forbidden)
    monkeypatch.setattr(rendering, "_render_spec_png", lambda path, *_args, **_kwargs: path)
    rendering.render_network(tmp_path, solving_genome, title="task artifact")

    composition = minimal_composition([("BINARY|K", 2)], "head", 1, InnovationTracker(_next_node_id=0), random.Random(0))
    rendering.render_composition_network(tmp_path, composition, title="composition artifact")

    wide_nodes = {node_id: NodeGene(node_id, NodeKind.INPUT, "identity") for node_id in range(1501)}
    wide = Genome(nodes=wide_nodes, connections=[])
    library = ModuleLibrary(tmp_path / "library")
    wide_key = library.add(entry_type=MODULE, payload=genome_to_dict(wide), io={"inputs": [{"signature": "ANY", "width": 1501}], "output": _IO["output"]}, provenance={})
    gallery = rendering.render_library_gallery(library, tmp_path / "gallery.png")
    assert gallery.exists()

    nested = rendering.build_genome_spec(_macro_host(wide_key), resolve=library_resolver(library))
    assert any(box.opaque and wide_key in box.label for box in nested.containers)

    view = rendering.OvermindView(
        vertices=[rendering.OvermindVertex(key=wide_key, label="wide")],
        input_signatures=["ANY:1501"],
        output_signatures=["BINARY|K:1"],
        d_model=8,
        top_k=1,
        max_steps=1,
    )
    overmind = rendering.build_overmind_spec(view, resolve=library_resolver(library), legend=False)
    assert any(box.opaque and "wide" in box.label for box in overmind.containers)


@pytest.mark.parametrize(
    ("descriptors", "expected_panels"),
    [
        ([("CONTINUOUS|C,H,W", (2, 3, 4))], 2),
        ([("CONTINUOUS|E,C,H,W", (4, 1, 3, 4))], 4),
        ([("CONTINUOUS|W,C,H", (4, 2, 3))], 2),
        ([("CONTINUOUS|C,H,W", (1, 2, 3)), ("CONTINUOUS|E,H,W", (2, 2, 2))], 3),
    ],
)
def test_density_semantic_layouts_are_deterministic(descriptors, expected_panels) -> None:
    from versal.rendering import _density_layout

    entry = _spatial_entry("spatial", descriptors)
    first = _density_layout(entry)
    second = _density_layout(entry)

    assert first.mode == "semantic-spatial"
    assert first.fallback_reason is None
    assert len(first.panels) == expected_panels
    assert sorted(node_id for panel in first.panels for node_id in panel.node_ids) == sorted(first.input_ids)
    assert first.positions == second.positions
    assert all(0.0 <= coordinate <= 1.0 for position in first.positions.values() for coordinate in position)


@pytest.mark.parametrize(
    ("signature", "coordinate", "reason"),
    [
        ("CONTINUOUS|C", [0.0], "no unique H/W"),
        ("CONTINUOUS|C,H,W", None, "do not match"),
        ("CONTINUOUS|C,H,W", [0.0, "bad", 0.0], "not numeric"),
    ],
)
def test_density_malformed_or_nonspatial_inputs_use_packed_grid(signature, coordinate, reason) -> None:
    from versal.rendering import _density_layout

    entry = _spatial_entry("fallback", [("CONTINUOUS|C,H,W", (1, 2, 2))])
    entry.io["inputs"][0]["signature"] = signature
    for node in entry.payload["nodes"]:
        if node["kind"] == "input":
            node["coordinate"] = coordinate
    layout = _density_layout(entry)

    assert layout.mode == "packed-grid"
    assert reason in (layout.fallback_reason or "")
    assert len({layout.positions[node_id] for node_id in layout.input_ids}) == len(layout.input_ids)


def test_density_duplicate_coordinates_use_packed_grid() -> None:
    from versal.rendering import _density_layout

    entry = _spatial_entry("duplicate", [("CONTINUOUS|C,H,W", (1, 2, 2))])
    entry.payload["nodes"][1]["coordinate"] = entry.payload["nodes"][0]["coordinate"]
    layout = _density_layout(entry)

    assert layout.mode == "packed-grid"
    assert "duplicate coordinates" in (layout.fallback_reason or "")


def test_density_rasterizes_every_edge_category_and_counts_all_enabled() -> None:
    import datashader as ds
    import numpy as np
    import pandas as pd

    from versal.rendering import _density_layout, _macro_implied_payload_edges, _rasterized_edge_layers

    entry = _spatial_entry("layers", [("CONTINUOUS|C,H,W", (1, 2, 2))])
    hidden_id = next(node["id"] for node in entry.payload["nodes"] if node["kind"] == "hidden")
    entry.payload["macros"] = [{"ref": "library:m1_inner", "inputs": [2], "outputs": [hidden_id], "innovation": 9}]
    layout = _density_layout(entry)
    canvas = ds.Canvas(plot_width=64, plot_height=48, x_range=(0.0, 1.0), y_range=(0.0, 1.0))
    layers, enabled, rendered = _rasterized_edge_layers(
        canvas,
        layout.positions,
        entry.payload["connections"],
        _macro_implied_payload_edges(entry.payload),
        ds,
        pd,
        np,
    )

    assert enabled == rendered == 5
    assert all(layers[name] is not None for name in ("positive", "negative", "recurrent", "macro"))
    assert all(float(np.asarray(layers[name].data).sum()) > 0 for name in layers)


def test_density_edge_chunk_boundary_includes_every_enabled_edge() -> None:
    import datashader as ds
    import numpy as np
    import pandas as pd

    import versal.rendering as rendering

    positions = {0: (0.1, 0.1), 1: (0.9, 0.9)}
    connection = {"in": 0, "out": 1, "weight": 1.0, "enabled": True, "recurrent": False}
    connections = [connection] * (rendering._DENSITY_EDGE_CHUNK + 1)
    canvas = ds.Canvas(plot_width=16, plot_height=16, x_range=(0.0, 1.0), y_range=(0.0, 1.0))
    layers, enabled, rendered = rendering._rasterized_edge_layers(canvas, positions, connections, [], ds, pd, np)

    assert enabled == rendered == rendering._DENSITY_EDGE_CHUNK + 1
    assert layers["positive"] is not None


def test_large_density_png_is_fixed_size_nonblank_and_deterministic(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.image as mpimg
    import numpy as np
    from matplotlib.axes import Axes

    from versal.rendering import _render_large_module_density

    entry = _spatial_entry("portrait", [("CONTINUOUS|E,C,H,W", (4, 1, 3, 4))])
    labels: list[str] = []
    original_text = Axes.text

    def capture_text(self, _x, _y, value, *args, **kwargs):
        labels.append(str(value))
        return original_text(self, _x, _y, value, *args, **kwargs)

    monkeypatch.setattr(Axes, "text", capture_text)
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    first = _render_large_module_density(first_path, entry)
    _render_large_module_density(second_path, entry)
    first_pixels = mpimg.imread(first_path)
    second_pixels = mpimg.imread(second_path)

    assert first.canvas_width == 4800 and first.canvas_height == 3200
    assert first.enabled_edge_count == first.rendered_edge_count == 5
    assert first.isolated_input_count == 46
    assert first.semantic_layout_mode == "semantic-spatial"
    assert first_pixels.shape[:2] == (3200, 4800)
    assert first_pixels.std() > 0.01
    assert np.array_equal(first_pixels, second_pixels)
    assert "potential influence flow" in labels
    assert any("weights, not activations" in label for label in labels)


def test_large_density_failure_atomically_replaces_with_opaque_summary(monkeypatch, tmp_path: Path, solving_genome: Genome) -> None:
    import versal.rendering as rendering

    target = tmp_path / "portrait.png"
    target.write_bytes(b"old portrait")
    entry = _entry("wide", genome_to_dict(solving_genome))

    def fail(path: Path, _entry: LibraryEntry):
        path.write_bytes(b"partial density image")
        raise ImportError("datashader unavailable")

    def summary(path: Path, *_args, **_kwargs):
        path.write_bytes(b"opaque summary")
        return path

    monkeypatch.setattr(rendering, "_render_large_module_density", fail)
    monkeypatch.setattr(rendering, "_render_spec_png", summary)
    metadata = rendering._render_large_module_png(target, entry)

    assert target.read_bytes() == b"opaque summary"
    assert metadata.semantic_layout_mode == "opaque-fallback"
    assert "ImportError: datashader unavailable" == metadata.fallback_reason
    assert not list(tmp_path.glob(".*.tmp.png"))


def test_render_entry_writes_png_for_all_fixture_entries(tmp_path: Path) -> None:
    library = ModuleLibrary(_FIXTURE_LIBRARY)
    for key in library.keys():
        image_path = render_entry(tmp_path / f"{key}.png", library.load(key), library=library)
        assert image_path.exists() and image_path.stat().st_size > 0


def test_render_library_gallery_smoke(tmp_path: Path) -> None:
    library = ModuleLibrary(_FIXTURE_LIBRARY)
    out_path = render_library_gallery(library, tmp_path / "gallery.png", columns=2)
    assert out_path.exists() and out_path.stat().st_size > 0


def test_gallery_limits_nested_detail_to_one_reference_level(monkeypatch, tmp_path: Path) -> None:
    import versal.rendering as rendering

    library = ModuleLibrary(_FIXTURE_LIBRARY)
    observed_depths: list[int] = []
    original = rendering.build_entry_spec

    def capture(*args, **kwargs):
        observed_depths.append(kwargs["max_inline_depth"])
        return original(*args, **kwargs)

    monkeypatch.setattr(rendering, "build_entry_spec", capture)
    rendering.render_library_gallery(library, tmp_path / "gallery.png", columns=2, max_inline_depth=5)

    assert observed_depths and set(observed_depths) == {1}


def test_gallery_empty_library_still_writes(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "empty")
    out_path = render_library_gallery(library, tmp_path / "gallery.png")
    assert out_path.exists() and out_path.stat().st_size > 0


def test_composition_spec_without_library_uses_opaque_boxes() -> None:
    library = ModuleLibrary(_FIXTURE_LIBRARY)
    entry = library.load("c2_88ff427f98da")
    spec = build_entry_spec(entry, resolve=None)
    assert len(spec.containers) == 1 and spec.containers[0].opaque


# --- overmind flow grid ----------------------------------------------------------------------------


def _grid_view(count: int):
    from versal.rendering import OvermindVertex, OvermindView

    vertices = [OvermindVertex(key="", label=f"v{index}", embedding_rank=index) for index in range(count)]
    return OvermindView(vertices=vertices, input_signatures=["IN_2"], output_signatures=["OUT_1"], d_model=8, top_k=1, max_steps=2)


def test_overmind_grid_bands_and_rows() -> None:
    from versal.rendering import THEME, build_overmind_spec

    spec = build_overmind_spec(_grid_view(5), columns=4, legend=False)
    input_nodes = [node for node in spec.nodes if node.color == THEME["node_input"]]
    output_nodes = [node for node in spec.nodes if node.color == THEME["node_output"]]
    anchors = [node for node in spec.nodes if node.color == THEME["node_anchor"]]
    assert len(input_nodes) == 1 and len(output_nodes) == 1
    assert len(anchors) == 5
    assert input_nodes[0].y == max(node.y for node in spec.nodes)  # input band at the TOP (y-up frame)
    assert output_nodes[0].y == min(node.y for node in spec.nodes)  # output band at the BOTTOM
    cells = [box for box in spec.containers if box.depth == 1]
    assert len(cells) == 5
    actual_anchors = sorted((node.x, node.y) for node in anchors)
    expected_anchors = sorted((box.x0 + 0.3, box.y1 - 0.3) for box in cells)
    assert all(actual == pytest.approx(expected) for actual, expected in zip(actual_anchors, expected_anchors))
    assert all(box.y1 < input_nodes[0].y and box.y0 > output_nodes[0].y for box in cells)  # grid strictly between the bands
    row_tops = sorted({round(box.y1, 6) for box in cells}, reverse=True)
    assert len(row_tops) == 2  # 5 cells at columns=4 -> a 4-row and a 1-row
    assert sum(1 for box in cells if round(box.y1, 6) == row_tops[0]) == 4
    assert not any(node.marker == "D" for node in spec.nodes)  # the gate hub is gone
    band_labels = [text.text for text in spec.texts]
    assert "IN_2" in band_labels and "OUT_1" in band_labels


def test_overmind_default_grid_keeps_order_at_eight_across() -> None:
    from versal.rendering import build_overmind_spec

    spec = build_overmind_spec(_grid_view(9), legend=False)
    cells = [box for box in spec.containers if box.depth == 1]
    row_tops = sorted({round(box.y1, 6) for box in cells}, reverse=True)
    first_row = sorted((box for box in cells if round(box.y1, 6) == row_tops[0]), key=lambda box: box.x0)
    second_row = sorted((box for box in cells if round(box.y1, 6) == row_tops[1]), key=lambda box: box.x0)
    assert [box.label for box in first_row] == [f"v{index}" for index in range(8)]
    assert [box.label for box in second_row] == ["v8"]


def test_pruned_overmind_compacts_survivors_to_eight_across_and_remaps_paths() -> None:
    from versal.rendering import build_overmind_spec, prune_overmind_view

    view = _grid_view(10)
    view.vertices[1].retired = True
    view.vertices[4].retired = True
    view.pathways = [(0, 2, 0.8), (1, 2, 0.7), (8, 9, 0.6)]

    pruned = prune_overmind_view(view)

    assert [vertex.label for vertex in pruned.vertices] == ["v0", "v2", "v3", "v5", "v6", "v7", "v8", "v9"]
    assert pruned.pathways == [(0, 1, 0.8), (6, 7, 0.6)]
    spec = build_overmind_spec(pruned, legend=False)
    cells = [box for box in spec.containers if box.depth == 1]
    assert len({round(box.y1, 6) for box in cells}) == 1
    assert len(cells) == 8


def test_overmind_render_uses_double_resolution_and_wider_sides(monkeypatch, tmp_path: Path) -> None:
    import versal.rendering as rendering

    captured: dict[str, float | int] = {}
    paths: list[Path] = []

    def capture(path, _spec, _title, **kwargs):
        paths.append(path)
        captured.update(kwargs)
        return path

    monkeypatch.setattr(rendering, "_render_spec_png", capture)
    out_path = tmp_path / "overmind.png"
    assert rendering.render_overmind(out_path, _grid_view(1)) == out_path
    assert paths == [out_path, tmp_path / "overmind_pruned.png"]
    assert captured == {"dpi": 300, "x_padding": 1.8}


def test_overmind_fresh_library_draws_uniform_feeds() -> None:
    from versal.rendering import THEME, build_overmind_spec

    spec = build_overmind_spec(_grid_view(3), legend=False)  # every share is 0.0: no traffic yet
    entry_edges = [edge for edge in spec.edges if edge.color == THEME["edge_entry"]]
    exit_edges = [edge for edge in spec.edges if edge.color == THEME["edge_exit"]]
    assert len(entry_edges) == 3 and len(exit_edges) == 3
    assert all(edge.width == 0.8 for edge in entry_edges + exit_edges)  # uniform thin: the flow story on day one
    assert all(edge.alpha == 0.6 for edge in entry_edges + exit_edges)
    anchors = {(node.x, node.y) for node in spec.nodes if node.color == THEME["node_anchor"]}
    assert {(edge.x1, edge.y1) for edge in entry_edges} == anchors


def test_overmind_pathway_flows_from_card_output_to_input_anchor() -> None:
    from versal.rendering import THEME, build_overmind_spec

    view = _grid_view(2)
    view.pathways = [(0, 1, 0.75), (1, 1, 0.5)]  # repeated selection is recurrent flow through one expert
    spec = build_overmind_spec(view, legend=False)
    cells = [box for box in spec.containers if box.depth == 1]
    pathways = [edge for edge in spec.edges if edge.color == THEME["edge_pathway"]]
    assert len(pathways) == 2
    assert (pathways[0].x0, pathways[0].y0) == pytest.approx(((cells[0].x0 + cells[0].x1) / 2, cells[0].y0))  # opaque fallback has no rendered output node
    assert (pathways[0].x1, pathways[0].y1) == pytest.approx((cells[1].x0 + 0.3, cells[1].y1 - 0.3))
    assert (pathways[1].x0, pathways[1].y0) == pytest.approx(((cells[1].x0 + cells[1].x1) / 2, cells[1].y0))
    assert (pathways[1].x1, pathways[1].y1) == pytest.approx((cells[1].x0 + 0.3, cells[1].y1 - 0.3))


def test_overmind_nested_flow_runs_from_footprint_to_callout_anchor(tmp_path: Path, solving_genome: Genome) -> None:
    from versal.rendering import THEME, OvermindVertex, OvermindView, build_overmind_spec

    library = ModuleLibrary(tmp_path / "lib")
    inner_key = library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=_IO, provenance={"accepted_metric": 1.0})
    host_key = library.add(entry_type=MODULE, payload=genome_to_dict(_macro_host(inner_key)), io=_IO, provenance={"accepted_metric": 1.0})
    view = OvermindView(
        vertices=[OvermindVertex(key=host_key, label="host"), OvermindVertex(key=inner_key, label="inner", embedding_rank=1)],
        input_signatures=["IN_2"],
        output_signatures=["OUT_1"],
        d_model=8,
        top_k=1,
        max_steps=1,
        pathways=[(0, 1, 0.75)],
    )
    spec = build_overmind_spec(view, resolve=library_resolver(library), legend=False)
    assert not any(edge.color == THEME["edge_macro"] and edge.style == "dashed" for edge in spec.edges)
    nested_box = next(box for box in spec.containers if box.depth > 1 and inner_key in box.label)
    nested_anchor = next(node for node in spec.nodes if node.color == THEME["node_anchor"] and (node.x, node.y) == pytest.approx((nested_box.x0 + 0.3, nested_box.y1 - 0.3)))
    flow = next(edge for edge in spec.edges if edge.color == THEME["edge_callout"])
    footprint = next(node for node in spec.nodes if node.color == THEME["node_module"])
    assert (flow.x0, flow.y0) == pytest.approx((footprint.x, footprint.y))
    assert (flow.x1, flow.y1) == pytest.approx((nested_anchor.x, nested_anchor.y))

    host_box = next(box for box in spec.containers if box.depth == 1 and box.label == "host")
    inner_box = next(box for box in spec.containers if box.depth == 1 and box.label == "inner")
    host_output = next(
        node
        for node in spec.nodes
        if node.color == THEME["node_output"]
        and host_box.x0 < node.x < host_box.x1
        and host_box.y0 < node.y < host_box.y1
        and not (nested_box.x0 < node.x < nested_box.x1 and nested_box.y0 < node.y < nested_box.y1)
    )
    pathway = next(edge for edge in spec.edges if edge.color == THEME["edge_pathway"])
    assert (pathway.x0, pathway.y0) == pytest.approx((host_output.x, host_output.y))
    assert (pathway.x1, pathway.y1) == pytest.approx((inner_box.x0 + 0.3, inner_box.y1 - 0.3))


def test_overmind_legend_populates_texts_and_widens() -> None:
    from versal.rendering import build_overmind_spec

    bare = build_overmind_spec(_grid_view(2), legend=False)
    keyed = build_overmind_spec(_grid_view(2), legend=True)
    assert keyed.width > bare.width
    labels = [text.text for text in keyed.texts]
    assert "key" in labels
    expected_labels = ("routing traffic (observed)", "input feed (step-0 gate mass)", "output feed (final-step gate mass)", "recurrent (time-delayed)")
    for expected in expected_labels + ("network input anchor", "nested-network flow"):
        assert expected in labels
    assert "built from (structural ref)" not in labels
    feed_swatches = [edge for edge in keyed.edges if edge.role == "legend" and edge.color in {THEME["edge_entry"], THEME["edge_exit"]}]
    assert len(feed_swatches) == 2 and all(edge.alpha == 0.6 for edge in feed_swatches)


def test_pruned_adaptive_legend_only_shows_present_symbols_and_moves_below_narrow_grid() -> None:
    from versal.rendering import build_overmind_spec

    full = build_overmind_spec(_grid_view(1), legend=True)
    adaptive = build_overmind_spec(_grid_view(1), legend=True, legend_mode="adaptive")
    labels = {text.text for text in adaptive.texts}

    assert "input feed (step-0 gate mass)" in labels
    assert "output feed (final-step gate mass)" in labels
    assert "retired or unexpanded network" in labels
    assert "macro implied wiring" not in labels
    assert "recurrent (time-delayed)" not in labels
    assert adaptive.width < full.width
    expert = next(box for box in adaptive.containers if box.depth == 1 and box.label == "v0")
    legend_panel = next(box for box in adaptive.containers if box.depth == 0)
    assert legend_panel.y1 < expert.y0


def test_cold_overmind_labels_routing_potential_not_observed_traffic() -> None:
    from versal.rendering import build_overmind_spec

    view = _grid_view(2)
    view.traffic_observed = False
    spec = build_overmind_spec(view, legend=True)
    labels = [text.text for text in spec.texts]

    assert spec.flow_label == "routing potential · cold structural view, not observed traffic or activations"
    assert "routing potential (cold structural view)" in labels
    assert "routing traffic (observed)" not in labels


def test_spec_text_draws() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from versal.rendering import RenderSpec, SpecText, draw_spec

    spec = RenderSpec(texts=[SpecText(1.0, 2.0, "hello")], width=4.0, height=4.0)
    figure, axis = plt.subplots()
    draw_spec(axis, spec)
    assert any(text.get_text() == "hello" for text in axis.texts)
    plt.close(figure)


def test_render_spec_png_preserves_aspect(tmp_path: Path) -> None:
    import matplotlib.image as mpimg

    from versal.rendering import RenderSpec, _render_spec_png

    tall = _render_spec_png(tmp_path / "tall.png", RenderSpec(width=10.0, height=80.0), "tall")
    image = mpimg.imread(tall)
    assert image.shape[0] > image.shape[1]  # a tall spec renders taller than wide, not a padded square


def test_default_render_size_is_double_density_and_memory_bounded() -> None:
    from versal.rendering import _MAX_RENDER_PIXELS, _RENDER_DPI, RenderSpec, _render_figure_size

    square_width, square_height = _render_figure_size(RenderSpec(width=4.0, height=4.0), _RENDER_DPI)
    assert (round(square_width * _RENDER_DPI), round(square_height * _RENDER_DPI)) == (1200, 1200)

    huge_width, huge_height = _render_figure_size(RenderSpec(width=100.0, height=100.0), _RENDER_DPI)
    assert huge_width * huge_height * _RENDER_DPI * _RENDER_DPI == pytest.approx(_MAX_RENDER_PIXELS)


def test_draw_spec_orders_exact_alpha_feeds_beneath_network_and_routing(monkeypatch) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import versal.rendering as rendering

    calls: list[tuple[list[str], list[float], float]] = []

    def capture(_axis, edges, *, directional, zorder=2.2):
        del directional
        calls.append(([edge.role for edge in edges], [edge.alpha for edge in edges], zorder))

    monkeypatch.setattr(rendering, "_draw_classic_edges", capture)
    spec = rendering.RenderSpec(
        edges=[
            rendering.SpecEdge(0, 0, 1, 1, 1, THEME["edge_positive"], alpha=0.4, role="forward-positive"),
            rendering.SpecEdge(0, 1, 1, 0, 1, THEME["edge_entry"], alpha=0.6, role="routing-entry"),
            rendering.SpecEdge(1, 0, 0, 1, 1, THEME["edge_pathway"], alpha=0.7, role="routing-observed"),
        ],
        width=1,
        height=1,
    )
    figure, axis = plt.subplots()
    rendering.draw_spec(axis, spec)
    plt.close(figure)

    assert calls == [
        (["routing-entry"], [0.6], 1.8),
        (["forward-positive"], [0.4], 2.2),
        (["routing-observed"], [0.7], 2.4),
    ]


def test_render_spec_png_preserves_destination_when_save_fails(monkeypatch, tmp_path: Path) -> None:
    from matplotlib.figure import Figure

    from versal.rendering import RenderSpec, _render_spec_png

    target = tmp_path / "portrait.png"
    target.write_bytes(b"old portrait")

    def fail_savefig(self, path, *args, **kwargs):
        Path(path).write_bytes(b"partial portrait")
        raise RuntimeError("save failed")

    monkeypatch.setattr(Figure, "savefig", fail_savefig)
    with pytest.raises(RuntimeError, match="save failed"):
        _render_spec_png(target, RenderSpec(width=2.0, height=1.0), "test")

    assert target.read_bytes() == b"old portrait"
    assert not list(tmp_path.glob(".*.tmp.png"))


def test_draw_spec_falls_back_when_datashader_layer_fails(monkeypatch) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import versal.rendering as rendering

    def fail(*_args, **_kwargs):
        raise ImportError("Datashader unavailable")

    monkeypatch.setattr(rendering, "_rasterized_spec_edges", fail)
    monkeypatch.setattr(rendering, "_EXPLICIT_EDGE_LIMIT", 0)
    monkeypatch.setattr(rendering, "_MAX_STRAIGHT_EDGES", 1)
    spec = rendering.RenderSpec(
        edges=[
            rendering.SpecEdge(0.0, 0.0, 1.0, 1.0, 1.0, THEME["edge_positive"]),
            rendering.SpecEdge(0.0, 1.0, 1.0, 0.0, 1.0, THEME["edge_negative"]),
        ],
        width=1.0,
        height=1.0,
        flow_label="potential influence flow · test",
    )
    figure, axis = plt.subplots()
    rendering.draw_spec(axis, spec)

    assert any("classic fallback" in text.get_text() for text in axis.texts)
    assert axis.collections  # classic line collection remains visible
    plt.close(figure)


def test_overmind_caps_per_cell_detail_for_wide_experts(tmp_path: Path, solving_genome: Genome) -> None:
    """An image-scale expert (the 798-node MNIST stepping stone class) must degrade to its opaque
    footprint instead of embedding a ~784-node input column that degenerates the whole portrait
    into a tall-narrow bar and starves every other cell's budget."""
    from versal.evolution.genome import ConnectionGene, Genome, NodeGene, NodeKind
    from versal.rendering import OvermindVertex, OvermindView, build_overmind_spec

    nodes = {i: NodeGene(i, NodeKind.INPUT, "identity") for i in range(784)}
    nodes[784] = NodeGene(784, NodeKind.OUTPUT, "identity")
    wide = Genome(nodes=nodes, connections=[ConnectionGene(i, 784, 0.1, True, i) for i in range(784)])

    library = ModuleLibrary(tmp_path / "lib")
    wide_key = library.add(entry_type=MODULE, payload=genome_to_dict(wide), io=_IO, provenance={"accepted_metric": 0.7})
    small_key = library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=_IO, provenance={"accepted_metric": 1.0})
    view = OvermindView(
        vertices=[OvermindVertex(key=wide_key, label="wide stone"), OvermindVertex(key=small_key, label="small", embedding_rank=1)],
        input_signatures=["IN_784"],
        output_signatures=["OUT_10"],
        d_model=8,
        top_k=1,
        max_steps=1,
    )
    spec = build_overmind_spec(view, resolve=library_resolver(library), legend=False)
    wide_boxes = [box for box in spec.containers if "wide stone" in (box.label or "")]
    assert wide_boxes and all(box.opaque for box in wide_boxes)  # capped: footprint, not a 784-node bar
    assert "785 nodes · 784 edges" in wide_boxes[0].label
    small_boxes = [box for box in spec.containers if (box.label or "").startswith("small")]
    assert small_boxes and not all(box.opaque for box in small_boxes)  # small experts keep full detail
    assert max(spec.width, spec.height) / max(min(spec.width, spec.height), 1e-6) < 6.0  # sane aspect
