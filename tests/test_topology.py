"""Weight-independent, exact topology identity used by post-solve refinement."""

from __future__ import annotations

import copy
from pathlib import Path

from ardevo.library import COMPOSITION, MODULE, ModuleLibrary
from ardevo.topology import TopologyTabuSession, TopologyTabuStore, same_topology, topology_record


def _module_payload() -> dict:
    return {
        "nodes": [
            {"id": 0, "kind": "input", "activation": "identity", "coordinate": [0.0], "aggregation": "sum"},
            {"id": 1, "kind": "input", "activation": "identity", "coordinate": [1.0], "aggregation": "sum"},
            {"id": 2, "kind": "bias", "activation": "identity", "coordinate": None, "aggregation": "sum"},
            {"id": 3, "kind": "hidden", "activation": "tanh", "coordinate": [0.5], "aggregation": "product"},
            {"id": 4, "kind": "output", "activation": "sigmoid", "coordinate": None, "aggregation": "sum"},
        ],
        "connections": [
            {"in": 0, "out": 3, "weight": 0.2, "enabled": True, "innovation": 7, "recurrent": False, "tie": 31},
            {"in": 1, "out": 3, "weight": -0.8, "enabled": True, "innovation": 8, "recurrent": False, "tie": 31},
            {"in": 2, "out": 3, "weight": 1.0, "enabled": True, "innovation": 9, "recurrent": False},
            {"in": 3, "out": 4, "weight": 0.5, "enabled": True, "innovation": 10, "recurrent": False},
        ],
        "macros": [],
        "refine_steps": 2,
        "operator_rates": {"add_node": 0.7},
    }


def _renumber_module(payload: dict) -> dict:
    remap = {0: 100, 1: 120, 2: 140, 3: 900, 4: 960}
    changed = copy.deepcopy(payload)
    for node in changed["nodes"]:
        node["id"] = remap[node["id"]]
    for offset, connection in enumerate(changed["connections"]):
        connection["in"] = remap[connection["in"]]
        connection["out"] = remap[connection["out"]]
        connection["weight"] += 5.0
        connection["innovation"] = 1000 + offset
        if "tie" in connection:
            connection["tie"] = 777
    changed["nodes"].reverse()
    changed["connections"].reverse()
    changed["operator_rates"] = {"add_node": 0.01, "remove_node": 0.99}
    return changed


def test_module_topology_is_id_innovation_weight_and_strategy_invariant() -> None:
    baseline = topology_record(MODULE, _module_payload())
    renumbered = topology_record(MODULE, _renumber_module(_module_payload()))
    assert same_topology(baseline, renumbered)


def test_module_topology_preserves_architectural_attributes_and_weight_ties() -> None:
    payload = _module_payload()
    baseline = topology_record(MODULE, payload)
    for mutate in (
        lambda value: value["nodes"][3].update(activation="relu"),
        lambda value: value["nodes"][3].update(aggregation="sum"),
        lambda value: value["nodes"][3].update(coordinate=[0.25]),
        lambda value: value["connections"][0].update(enabled=False),
        lambda value: value["connections"][0].update(recurrent=True),
        lambda value: value["connections"][1].update(tie=99),
        lambda value: value.update(refine_steps=3),
    ):
        changed = copy.deepcopy(payload)
        mutate(changed)
        assert not same_topology(baseline, topology_record(MODULE, changed))


def _composition_payload(reference: str) -> dict:
    return {
        "nodes": [
            {"id": 0, "kind": "input", "ref": "BINARY|K", "in_width": 0, "out_width": 2, "aggregation": "sum", "trainable": True},
            {"id": 7, "kind": "module", "ref": reference, "in_width": 2, "out_width": 1, "aggregation": "product", "trainable": False},
            {"id": 9, "kind": "output", "ref": "xor", "in_width": 1, "out_width": 0, "aggregation": "sum", "trainable": True},
        ],
        "edges": [
            {
                "in": 0,
                "out": 7,
                "enabled": True,
                "innovation": 1,
                "glue": [1.0, -1.0],
                "glue_rank": 0,
                "port_map": [{"source_start": 0, "target_start": 0, "length": 1}],
            },
            {"in": 7, "out": 9, "enabled": True, "innovation": 2, "glue": [0.5], "glue_rank": 0},
        ],
    }


def test_composition_topology_ignores_ids_innovations_and_glue_but_keeps_port_maps() -> None:
    baseline_payload = _composition_payload("live:3")
    changed = copy.deepcopy(baseline_payload)
    remap = {0: 100, 7: 800, 9: 900}
    for node in changed["nodes"]:
        node["id"] = remap[node["id"]]
    for edge in changed["edges"]:
        edge["in"] = remap[edge["in"]]
        edge["out"] = remap[edge["out"]]
        edge["innovation"] += 500
        edge["glue"] = [value + 20.0 for value in edge["glue"]]
    changed["nodes"].reverse()
    changed["edges"].reverse()
    baseline = topology_record(COMPOSITION, baseline_payload)
    assert same_topology(baseline, topology_record(COMPOSITION, changed))

    remapped_port = copy.deepcopy(changed)
    remapped_port["edges"][1]["port_map"][0]["source_start"] = 1
    assert not same_topology(baseline, topology_record(COMPOSITION, remapped_port))


def test_library_references_compare_the_referenced_topology_not_entry_key(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "library")
    first = _module_payload()
    second = copy.deepcopy(first)
    second["connections"][0]["weight"] += 0.25
    different = copy.deepcopy(first)
    different["nodes"][3]["activation"] = "relu"
    io = {"inputs": [{"signature": "BINARY|K", "width": 2}], "output": {"signature": "BINARY|K", "width": 1}}
    keys = [library.add(entry_type=MODULE, payload=payload, io=io, provenance={}) for payload in (first, second, different)]

    one = topology_record(COMPOSITION, _composition_payload(f"library:{keys[0]}"), library=library)
    retrained = topology_record(COMPOSITION, _composition_payload(f"library:{keys[1]}"), library=library)
    changed = topology_record(COMPOSITION, _composition_payload(f"library:{keys[2]}"), library=library)
    assert same_topology(one, retrained)
    assert not same_topology(one, changed)


def test_tabu_store_skips_persisted_exact_topologies_within_one_context(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "library")
    store = TopologyTabuStore(library.root / "topology_tabu.sqlite3")
    baseline = _module_payload()

    first = TopologyTabuSession(store, "same-task-lineage-config", library)
    first.prime(MODULE, baseline)
    assert not first.reserve(MODULE, _renumber_module(baseline))
    changed = copy.deepcopy(baseline)
    changed["nodes"][3]["activation"] = "relu"
    assert first.reserve(MODULE, changed)
    first.commit()

    reopened = TopologyTabuSession(store, "same-task-lineage-config", library)
    assert not reopened.reserve(MODULE, _renumber_module(changed))
    assert reopened.duplicates == 1

    isolated = TopologyTabuSession(store, "different-config", library)
    assert isolated.reserve(MODULE, baseline)
