"""Motif census: extraction rules, exact canonicalization, ESU enumeration, planted-motif support
counting, diversity grouping, the reuse census, and the CLI core / atlas smokes."""

import itertools
import json
import random
from pathlib import Path
from typing import Any

from versal.library import COMPOSITION, MODULE, ModuleLibrary
from versal.motifs import (
    FORWARD_EDGE,
    MACRO_EDGE,
    RECURRENT_EDGE,
    MotifRecord,
    NodeLabel,
    _skeleton,
    canonical_form,
    composition_motif_graph,
    diversity_class,
    enumerate_connected_subgraphs,
    module_motif_graph,
    motif_census,
    motif_fingerprint,
    report_to_dict,
    reuse_census,
)

_IO = {"inputs": [{"signature": "BINARY|K", "width": 2}], "output": {"signature": "BINARY|K", "width": 1}}
_FIXTURE_LIBRARY = Path(__file__).parent / "fixtures" / "library_v1"

_IN = ("input", "identity", "sum", "")
_OUT = ("output", "identity", "sum", "")
_HID_ID = ("hidden", "identity", "sum", "")
_HID_TANH = ("hidden", "tanh", "sum", "")
_HID_GATE = ("hidden", "tanh", "product", "")


# canonicalization


def test_canonical_form_is_permutation_invariant() -> None:
    labels: list[NodeLabel] = [_IN, _HID_TANH, _OUT]
    edges = [(0, 1, FORWARD_EDGE), (1, 2, FORWARD_EDGE), (0, 2, FORWARD_EDGE)]
    base = motif_fingerprint(canonical_form(labels, edges))
    for permutation in itertools.permutations(range(3)):
        relabeled: list[NodeLabel] = [_IN, _IN, _IN]
        for old_index, new_index in enumerate(permutation):
            relabeled[new_index] = labels[old_index]
        remapped = [(permutation[source], permutation[target], mask) for source, target, mask in edges]
        assert motif_fingerprint(canonical_form(relabeled, remapped)) == base


def test_canonical_form_distinguishes_direction_and_edge_kind() -> None:
    labels = [_HID_ID, _HID_ID, _HID_ID]
    chain = motif_fingerprint(canonical_form(labels, [(0, 1, FORWARD_EDGE), (1, 2, FORWARD_EDGE)]))
    fan_in = motif_fingerprint(canonical_form(labels, [(0, 1, FORWARD_EDGE), (2, 1, FORWARD_EDGE)]))
    recurrent_link = motif_fingerprint(canonical_form(labels, [(0, 1, RECURRENT_EDGE), (1, 2, FORWARD_EDGE)]))
    assert len({chain, fan_in, recurrent_link}) == 3


# ESU enumeration


def _brute_force(adjacency: dict[int, set[int]], k: int) -> set[frozenset[int]]:
    found: set[frozenset[int]] = set()
    for subset in itertools.combinations(sorted(adjacency), k):
        members = set(subset)
        frontier = [subset[0]]
        seen = {subset[0]}
        while frontier:
            for neighbor in adjacency[frontier.pop()] & members:
                if neighbor not in seen:
                    seen.add(neighbor)
                    frontier.append(neighbor)
        if seen == members:
            found.add(frozenset(subset))
    return found


def test_esu_counts_on_known_graphs() -> None:
    triangle = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
    assert len(enumerate_connected_subgraphs(triangle, 3, 10_000)[0]) == 1
    star = {0: {1, 2, 3}, 1: {0}, 2: {0}, 3: {0}}
    assert len(enumerate_connected_subgraphs(star, 3, 10_000)[0]) == 3
    path = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2, 4}, 4: {3}}
    assert len(enumerate_connected_subgraphs(path, 3, 10_000)[0]) == 3
    complete4 = {node: set(range(4)) - {node} for node in range(4)}
    assert len(enumerate_connected_subgraphs(complete4, 3, 10_000)[0]) == 4
    assert len(enumerate_connected_subgraphs(complete4, 4, 10_000)[0]) == 1


def test_esu_matches_brute_force_on_random_graph() -> None:
    rng = random.Random(0)
    adjacency: dict[int, set[int]] = {node: set() for node in range(8)}
    for source, target in itertools.combinations(range(8), 2):
        if rng.random() < 0.35:
            adjacency[source].add(target)
            adjacency[target].add(source)
    for k in (3, 4):
        subsets, truncated = enumerate_connected_subgraphs(adjacency, k, 100_000)
        assert not truncated
        assert set(subsets) == _brute_force(adjacency, k)
        assert len(subsets) == len(set(subsets))  # each connected set emitted exactly once


# extraction


def _connection(source: int, target: int, *, enabled: bool = True, recurrent: bool = False) -> dict[str, Any]:
    return {"in": source, "out": target, "weight": 1.0, "enabled": enabled, "innovation": source * 100 + target, "recurrent": recurrent}


def test_module_graph_extraction_rules() -> None:
    payload = {
        "nodes": [
            {"id": 0, "kind": "input", "activation": "identity"},
            {"id": 1, "kind": "bias", "activation": "identity"},
            {"id": 2, "kind": "hidden", "activation": "tanh", "aggregation": "product"},
            {"id": 3, "kind": "hidden", "activation": "identity"},
            {"id": 4, "kind": "output", "activation": "identity"},
        ],
        "connections": [
            _connection(0, 2),
            _connection(1, 2),  # from bias: excluded with the bias node
            _connection(2, 4, enabled=False),  # dormant gene, not dataflow
            _connection(2, 2, recurrent=True),  # the TRM self-loop
            _connection(3, 4),
        ],
        "macros": [{"ref": "library:m1_000000000000", "inputs": [0], "outputs": [3], "innovation": 9, "trainable": False}],
    }
    labels, edges = module_motif_graph(payload)
    assert 1 not in labels  # bias node gone
    assert labels[2] == ("hidden", "tanh", "product", "")
    assert labels[3] == ("hidden", "identity", "sum", "stub")  # macro output stub, labeled
    assert (1, 2) not in edges and (2, 4) not in edges
    assert edges[(2, 2)] == RECURRENT_EDGE
    assert edges[(0, 3)] == MACRO_EDGE  # macro-implied bipartite dataflow
    skeleton = _skeleton(labels, edges)
    assert 2 not in skeleton[2]  # the self-loop never enters the undirected skeleton


def test_composition_graph_ref_classes() -> None:
    payload = {
        "nodes": [
            {"id": 0, "kind": "input", "ref": "BINARY|K", "in_width": 0, "out_width": 2, "aggregation": "sum", "trainable": False},
            {"id": 1, "kind": "input", "ref": "__bias__", "in_width": 0, "out_width": 1, "aggregation": "sum", "trainable": False},
            {"id": 2, "kind": "module", "ref": "library:m1_aaaaaaaaaaaa", "in_width": 2, "out_width": 1, "aggregation": "sum", "trainable": True},
            {"id": 3, "kind": "output", "ref": "xor", "in_width": 1, "out_width": 0, "aggregation": "sum", "trainable": False},
        ],
        "edges": [
            {"in": 0, "out": 2, "enabled": True, "innovation": 0, "glue": [1.0, 1.0], "glue_rank": 0},
            {"in": 1, "out": 2, "enabled": False, "innovation": 1, "glue": [1.0], "glue_rank": 0},
            {"in": 2, "out": 3, "enabled": True, "innovation": 2, "glue": [1.0], "glue_rank": 0},
        ],
    }
    labels, edges = composition_motif_graph(payload)
    assert labels[0] == ("input", "input", "sum", "")
    assert labels[1] == ("input", "bias", "sum", "")
    assert labels[2] == ("module", "L1", "sum", "")
    assert labels[3] == ("output", "output", "sum", "")
    assert (1, 2) not in edges  # disabled comp edge dropped
    assert edges[(0, 2)] == FORWARD_EDGE and edges[(2, 3)] == FORWARD_EDGE


# census

_GADGET_LABELS: list[NodeLabel] = [_IN, _HID_TANH, _HID_GATE, _OUT]
_GADGET_EDGES = [(0, 1, FORWARD_EDGE), (1, 2, FORWARD_EDGE), (0, 2, FORWARD_EDGE), (2, 3, FORWARD_EDGE)]


def _gadget_payload(id_offset: int, weight: float) -> dict[str, Any]:
    """The planted gated-skip gadget under shifted ids and different weights (different weights keep
    the entry KEYS distinct; the canonical motif is identical)."""
    ids = [id_offset + index for index in range(4)]
    return {
        "nodes": [
            {"id": ids[0], "kind": "input", "activation": "identity"},
            {"id": ids[1], "kind": "hidden", "activation": "tanh"},
            {"id": ids[2], "kind": "hidden", "activation": "tanh", "aggregation": "product"},
            {"id": ids[3], "kind": "output", "activation": "identity"},
        ],
        "connections": [
            {"in": ids[0], "out": ids[1], "weight": weight, "enabled": True, "innovation": 0},
            {"in": ids[1], "out": ids[2], "weight": weight, "enabled": True, "innovation": 1},
            {"in": ids[0], "out": ids[2], "weight": weight, "enabled": True, "innovation": 2},
            {"in": ids[2], "out": ids[3], "weight": weight, "enabled": True, "innovation": 3},
        ],
    }


def _plain_chain_payload() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": 0, "kind": "input", "activation": "identity"},
            {"id": 1, "kind": "hidden", "activation": "identity"},
            {"id": 2, "kind": "output", "activation": "identity"},
        ],
        "connections": [
            {"in": 0, "out": 1, "weight": 1.0, "enabled": True, "innovation": 0},
            {"in": 1, "out": 2, "weight": 1.0, "enabled": True, "innovation": 1},
        ],
    }


def test_planted_motif_support_counting(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    for index in range(3):
        library.add(entry_type=MODULE, payload=_gadget_payload(id_offset=index * 10, weight=0.5 + index), io=_IO, provenance={"accepted_metric": 1.0})
    library.add(entry_type=MODULE, payload=_plain_chain_payload(), io=_IO, provenance={"accepted_metric": 1.0})
    expected = motif_fingerprint(canonical_form(_GADGET_LABELS, _GADGET_EDGES))

    report = motif_census(library, sizes=(4,), min_support=3)
    by_fingerprint = {record.fingerprint: record for record in report.module_motifs}
    assert expected in by_fingerprint
    record = by_fingerprint[expected]
    assert record.support == 3 and record.occurrences == 3
    assert record.diversity_class == "gated"
    assert record.exemplars == tuple(sorted(record.exemplars))

    stricter = motif_census(library, sizes=(4,), min_support=4)
    assert expected not in {record.fingerprint for record in stricter.module_motifs}


def test_cap_truncates_deterministically(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    dense_nodes = [{"id": index, "kind": "hidden", "activation": "tanh"} for index in range(7)]
    for weight in (1.0, 2.0):
        pairs = [(source, target) for source in range(7) for target in range(source + 1, 7)]
        dense_edges = [{"in": source, "out": target, "weight": weight, "enabled": True, "innovation": source * 10 + target} for source, target in pairs]
        payload = {"nodes": dense_nodes, "connections": dense_edges}
        library.add(entry_type=MODULE, payload=payload, io=_IO, provenance={"accepted_metric": 1.0})
    first = motif_census(library, sizes=(3,), min_support=2, per_entry_cap=5)
    second = motif_census(library, sizes=(3,), min_support=2, per_entry_cap=5)
    assert set(first.truncated_entries) == set(library.keys())
    assert first.truncated_entries[library.keys()[0]] == [3]
    assert report_to_dict(first) == report_to_dict(second)  # deterministic truncation


def test_composition_census_counts_shared_wiring(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    for glue in (0.5, 1.5):
        payload = {
            "nodes": [
                {"id": 0, "kind": "input", "ref": "BINARY|K", "in_width": 0, "out_width": 2, "aggregation": "sum", "trainable": False},
                {"id": 1, "kind": "module", "ref": "library:m1_aaaaaaaaaaaa", "in_width": 2, "out_width": 1, "aggregation": "sum", "trainable": True},
                {"id": 2, "kind": "output", "ref": "xor", "in_width": 1, "out_width": 0, "aggregation": "sum", "trainable": False},
            ],
            "edges": [
                {"in": 0, "out": 1, "enabled": True, "innovation": 0, "glue": [glue, glue], "glue_rank": 0},
                {"in": 1, "out": 2, "enabled": True, "innovation": 1, "glue": [glue], "glue_rank": 0},
            ],
        }
        library.add(entry_type=COMPOSITION, payload=payload, io=_IO, provenance={"accepted_metric": 0.9}, level=2)
    report = motif_census(library, min_support=2)
    full_wiring = [record for record in report.composition_motifs if record.size == 3]
    assert len(full_wiring) == 1
    assert full_wiring[0].support == 2
    assert any(label[1] == "L1" for label in full_wiring[0].graph.node_labels)


def test_diversity_class_grouping() -> None:
    assert diversity_class(canonical_form([_HID_ID, _HID_ID, _HID_ID], [(0, 1, FORWARD_EDGE), (1, 2, FORWARD_EDGE)])) == "uniform-identity"
    assert diversity_class(canonical_form([_IN, _OUT], [(0, 1, FORWARD_EDGE)])) == "mixed"
    assert diversity_class(canonical_form([_HID_TANH, _HID_GATE], [(0, 1, FORWARD_EDGE)])) == "gated"
    assert diversity_class(canonical_form([_HID_TANH, _HID_TANH], [(0, 1, RECURRENT_EDGE)])) == "recurrent"
    assert diversity_class(canonical_form([_HID_TANH, _HID_GATE], [(0, 1, RECURRENT_EDGE)])) == "recurrent+gated"
    stub: NodeLabel = ("hidden", "identity", "sum", "stub")
    assert diversity_class(canonical_form([stub, _HID_ID], [(0, 1, FORWARD_EDGE)])) == "macro"


# reuse census


def test_reuse_census_reverse_reference_index(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    inner = library.add(entry_type=MODULE, payload=_plain_chain_payload(), io=_IO, provenance={"accepted_metric": 1.0})
    host_payload = _gadget_payload(id_offset=0, weight=1.0)
    host_payload["macros"] = [{"ref": f"library:{inner}", "inputs": [0], "outputs": [1], "innovation": 5, "trainable": False}]
    host = library.add(entry_type=MODULE, payload=host_payload, io=_IO, provenance={"accepted_metric": 1.0})
    rows = {row["key"]: row for row in reuse_census(library)}
    assert rows[inner]["referenced_by"] == [host]
    assert rows[inner]["reference_count"] == 1 and rows[host]["reference_count"] == 0


def test_reuse_census_handles_legacy_v1_index() -> None:
    library = ModuleLibrary(_FIXTURE_LIBRARY)
    rows = reuse_census(library, include_retired=True)
    assert rows  # pre-gate v1 index rows (missing stats/flags) must not raise
    assert all("reference_count" in row for row in rows)


# CLI core + atlas


def test_run_census_cli_core_writes_json(tmp_path: Path) -> None:
    from versal.tools.motif_census import run_census

    library = ModuleLibrary(tmp_path / "lib")
    for index in range(2):
        library.add(entry_type=MODULE, payload=_gadget_payload(id_offset=index * 10, weight=1.0 + index), io=_IO, provenance={"accepted_metric": 1.0})
    json_out = tmp_path / "motifs.json"
    report = run_census(tmp_path / "lib", sizes=(3, 4), json_out=json_out)
    data = json.loads(json_out.read_text())
    report_keys = {
        "params",
        "entries_scanned",
        "scanned_keys",
        "input_fingerprint",
        "index_total",
        "truncated_entries",
        "module_motifs",
        "composition_motifs",
        "vocabulary",
        "explanations",
    }
    assert set(report_to_dict(report)) == report_keys  # the pure core: no "meta"
    assert set(data) == report_keys | {"meta"}  # the written file adds the volatile stamp
    assert set(data["meta"]) == {"generated_at", "library", "library_resolved"}
    assert len(data["module_motifs"]) == len(report.module_motifs) > 0
    assert data["entries_scanned"] == {"modules": 2, "compositions": 0}
    for row in data["module_motifs"]:
        assert set(row) == {"fingerprint", "size", "diversity_class", "support", "occurrences", "exemplars", "nodes", "edges", "description"}


def test_render_motif_atlas_smoke(tmp_path: Path) -> None:
    from versal.rendering import render_motif_atlas

    graph = canonical_form(_GADGET_LABELS, _GADGET_EDGES)
    loop_graph = canonical_form([_HID_TANH], [(0, 0, RECURRENT_EDGE)])
    records = [
        MotifRecord(fingerprint=motif_fingerprint(graph), size=4, graph=graph, support=3, occurrences=9, exemplars=("m1_a",), diversity_class="gated", description="gadget"),
        MotifRecord(
            fingerprint=motif_fingerprint(loop_graph), size=1, graph=loop_graph, support=2, occurrences=2, exemplars=("m1_b",), diversity_class="recurrent", description="loop"
        ),
    ]
    atlas = render_motif_atlas(tmp_path / "motifs.png", records)
    assert atlas.exists() and atlas.stat().st_size > 0
    placeholder = render_motif_atlas(tmp_path / "empty.png", [])
    assert placeholder.exists() and placeholder.stat().st_size > 0


def test_render_motif_atlas_empty_note_smoke(tmp_path: Path) -> None:
    from versal.rendering import render_motif_atlas

    noted = render_motif_atlas(tmp_path / "empty_note.png", [], empty_note="1 module entries scanned; rerun with --min-support 1 for intra-entry structure.")
    assert noted.exists() and noted.stat().st_size > 0


# provenance stamp + small-library trust


def test_run_census_meta_pure_core_and_fingerprint(tmp_path: Path) -> None:
    from datetime import datetime

    from versal.tools.motif_census import run_census

    library = ModuleLibrary(tmp_path / "lib")
    for index in range(2):
        library.add(entry_type=MODULE, payload=_gadget_payload(id_offset=index * 10, weight=1.0 + index), io=_IO, provenance={"accepted_metric": 1.0})

    first_out, second_out = tmp_path / "first.json", tmp_path / "second.json"
    run_census(tmp_path / "lib", sizes=(3, 4), json_out=first_out)
    run_census(tmp_path / "lib", sizes=(3, 4), json_out=second_out)
    first, second = json.loads(first_out.read_text()), json.loads(second_out.read_text())
    meta_first = first.pop("meta")
    second.pop("meta")
    assert first == second  # the pure core is byte-identical across runs; only meta varies
    datetime.fromisoformat(meta_first["generated_at"])  # parses as an ISO timestamp
    fingerprint_before = first["input_fingerprint"]
    assert second["input_fingerprint"] == fingerprint_before

    library.add(entry_type=MODULE, payload=_gadget_payload(id_offset=20, weight=9.0), io=_IO, provenance={"accepted_metric": 1.0})
    third_out = tmp_path / "third.json"
    run_census(tmp_path / "lib", sizes=(3, 4), json_out=third_out)
    third = json.loads(third_out.read_text())
    assert third["input_fingerprint"] != fingerprint_before  # a new entry moves the content fingerprint


def test_small_library_explains_empty_table(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    key = library.add(entry_type=MODULE, payload=_gadget_payload(id_offset=0, weight=1.0), io=_IO, provenance={"accepted_metric": 1.0})
    report = motif_census(library, sizes=(3, 4), min_support=2)
    assert report.module_motifs == []
    assert report.scanned_keys["modules"] == [key]
    assert report.index_total == 1
    note = report_to_dict(report)["explanations"]["module_motifs"]
    assert note is not None
    assert "1 module" in note and "--min-support 1" in note


def test_min_support_one_yields_intra_entry_motif(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    library.add(entry_type=MODULE, payload=_gadget_payload(id_offset=0, weight=1.0), io=_IO, provenance={"accepted_metric": 1.0})
    expected = motif_fingerprint(canonical_form(_GADGET_LABELS, _GADGET_EDGES))
    report = motif_census(library, sizes=(4,), min_support=1)
    by_fingerprint = {record.fingerprint: record for record in report.module_motifs}
    assert expected in by_fingerprint
    assert by_fingerprint[expected].support == 1  # intra-entry structure, not cross-entry recurrence
    rows = {row["fingerprint"]: row for row in report_to_dict(report)["module_motifs"]}
    assert rows[expected]["diversity_class"] == "gated"
