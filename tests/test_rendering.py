"""Recursive render specs: nested networks expand inline, every failure degrades to an opaque box."""

import random
from pathlib import Path

from ardevo.evolution.composition import minimal_composition
from ardevo.evolution.genome import ConnectionGene, Genome, InnovationTracker, MacroGene, NodeGene, NodeKind, genome_to_dict
from ardevo.library import MODULE, LibraryEntry, ModuleLibrary
from ardevo.rendering import (
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


# --- spec builders ---------------------------------------------------------------------------------


def test_genome_spec_basic(solving_genome: Genome) -> None:
    spec = build_genome_spec(solving_genome)
    assert spec.node_count == len(solving_genome.nodes)
    assert spec.containers == []
    assert len(spec.edges) == len(solving_genome.enabled_connections())
    assert spec.width > 0 and spec.height > 0


def test_genome_spec_expands_macro_as_callout(tmp_path: Path, solving_genome: Genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    key = library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=_IO, provenance={"accepted_metric": 1.0})
    spec = build_genome_spec(_macro_host(key), resolve=library_resolver(library))

    assert len(spec.containers) == 1
    container = spec.containers[0]
    assert not container.opaque
    assert key in container.label and "L1" in container.label
    # All 5 host nodes (the stub stays, as the macro's green footprint) + the 6 inner nodes.
    assert spec.node_count == 5 + len(solving_genome.nodes)
    # Inner edges + the host readout (stub -> output) + 2 implied macro edges + 1 green callout line.
    assert len(spec.edges) == len(solving_genome.enabled_connections()) + 1 + 2 + 1
    assert any(edge.color == THEME["edge_macro"] for edge in spec.edges)
    # The callout box sits ABOVE the host network, green-lined to the footprint stub.
    callout_lines = [edge for edge in spec.edges if edge.color == THEME["edge_callout"]]
    footprints = [node for node in spec.nodes if node.color == THEME["node_module"]]
    assert len(callout_lines) == 1 and len(footprints) == 1
    assert container.y0 > footprints[0].y


def test_macro_missing_ref_is_opaque_box(solving_genome: Genome) -> None:
    spec = build_genome_spec(_macro_host("m1_gone"), resolve=lambda key: None)
    assert len(spec.containers) == 1 and spec.containers[0].opaque
    assert spec.node_count == 5  # host only (incl the footprint stub), nothing expanded
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
    leaf = Genome(nodes={0: NodeGene(0, NodeKind.INPUT, "identity"), 1: NodeGene(1, NodeKind.OUTPUT, "identity")}, connections=[ConnectionGene(0, 1, 1.0, True, 0)])
    chain: dict[str, LibraryEntry] = {}
    payload = genome_to_dict(leaf)
    for index in reversed(range(6)):  # d0 embeds d1 embeds ... embeds d5 (the leaf)
        chain[f"d{index}"] = _entry(f"d{index}", payload)
        payload = genome_to_dict(_macro_host(f"d{index}"))
    spec = build_genome_spec(_macro_host("d0"), resolve=lambda key: chain.get(key))
    assert any(container.opaque for container in spec.containers)
    assert sum(1 for container in spec.containers if not container.opaque) == RENDER_MAX_DEPTH


def test_budget_falls_back_opaque(tmp_path: Path, solving_genome: Genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    key = library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=_IO, provenance={})
    spec = build_genome_spec(_macro_host(key), resolve=library_resolver(library), node_budget=5)
    assert len(spec.containers) == 1 and spec.containers[0].opaque


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


def test_render_entry_writes_png_for_all_fixture_entries(tmp_path: Path) -> None:
    library = ModuleLibrary(_FIXTURE_LIBRARY)
    for key in library.keys():
        image_path = render_entry(tmp_path / f"{key}.png", library.load(key), library=library)
        assert image_path.exists() and image_path.stat().st_size > 0


def test_render_library_gallery_smoke(tmp_path: Path) -> None:
    library = ModuleLibrary(_FIXTURE_LIBRARY)
    out_path = render_library_gallery(library, tmp_path / "gallery.png", columns=2)
    assert out_path.exists() and out_path.stat().st_size > 0


def test_gallery_empty_library_still_writes(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "empty")
    out_path = render_library_gallery(library, tmp_path / "gallery.png")
    assert out_path.exists() and out_path.stat().st_size > 0


def test_composition_spec_without_library_uses_opaque_boxes() -> None:
    library = ModuleLibrary(_FIXTURE_LIBRARY)
    entry = library.load("c2_88ff427f98da")
    spec = build_entry_spec(entry, resolve=None)
    assert len(spec.containers) == 1 and spec.containers[0].opaque
