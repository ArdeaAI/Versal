"""
Indexed, resumable grammar preparation. No model randomness is consumed here.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from versal.grammar import BoundaryPort, Grammar, Production, _canonical_mapping, _lineage_root, _Occurrence, _ports_for_occurrence, _production_key, _sha1, save_grammar
from versal.library import MODULE, LibraryEntry, ModuleLibrary
from versal.motifs import MotifGraph, canonical_form, composition_motif_graph, module_motif_graph, motif_fingerprint


def catalog_token(library: ModuleLibrary) -> str:
    """
    Cheap index-only identity; usage counters do not invalidate structural evidence.
    """
    return _sha1(
        [
            [row["key"], row.get("retired", False), row.get("dependency", False), row.get("search_lineage"), row.get("refined_from")]
            for row in sorted(library.summaries(include_retired=True), key=lambda row: row["key"])
        ]
    )


class OccurrenceCursor:
    """
    The original sorted ESU enumeration with an explicit, serializable DFS stack.
    """

    def __init__(self, entry: LibraryEntry, sizes: tuple[int, ...], cap: int, root: str, saved: dict[str, Any] | None = None) -> None:
        if entry.payload.get("representation") == "spatial":
            from versal.spatial import SpatialGenome, circuit_evidence

            entry = replace(entry, payload=circuit_evidence(SpatialGenome.from_payload(entry.payload)))
        self.entry, self.sizes, self.cap, self.root = entry, sizes, cap, root
        graph = module_motif_graph if entry.entry_type == MODULE else composition_motif_graph
        self.labels, self.edges = graph(entry.payload)
        self.nodes = sorted(self.labels)
        self.ranks = {node: i for i, node in enumerate(self.nodes)}
        self.skeleton = {node: set() for node in self.nodes}
        self.incident: dict[int, list[tuple[int, int]]] = {node: [] for node in self.nodes}
        self.outgoing: dict[int, list[tuple[int, int]]] = {node: [] for node in self.nodes}
        for source, target in self.edges:
            self.incident[source].append((source, target))
            self.incident[target].append((source, target))
            self.outgoing[source].append((source, target))
            if source != target:
                self.skeleton[source].add(target)
                self.skeleton[target].add(source)
        self.port_types = self._port_index()
        saved = saved or {}
        self.size_index = int(saved.get("size_index", 0))
        self.root_index = int(saved.get("root_index", 0))
        self.count = int(saved.get("count", 0))
        self.stack = saved.get("stack", [])

    def _port_index(self) -> dict[tuple[int, str, str], tuple[str, int]]:
        nodes = {int(node["id"]): node for node in self.entry.payload.get("nodes", [])}
        inputs = sorted(node for node, value in nodes.items() if value["kind"] == "input")
        signatures = [str(spec["signature"]) for spec in self.entry.io.get("inputs", []) for _ in range(int(spec["width"]))]
        input_signatures = dict(zip(inputs, signatures))
        output_signature = str(self.entry.io.get("output", {}).get("signature", "scalar"))
        result = {}
        for node_id, node in nodes.items():
            for direction in ("input", "output"):
                for role in ("terminal", "cut"):
                    if self.entry.entry_type == MODULE:
                        signature = "scalar" if role == "cut" else input_signatures.get(node_id, "scalar") if direction == "input" else output_signature
                        value = (signature, 1)
                    elif role == "terminal":
                        value = (
                            (str(node.get("ref", "input")), max(1, int(node.get("out_width", 1))))
                            if direction == "input"
                            else (str(self.entry.io.get("output", {}).get("signature", "output")), max(1, int(node.get("in_width", 1))))
                        )
                    else:
                        ref = str(node.get("ref", ""))
                        value = (ref if ref and not ref.startswith("library:") else "any", max(1, int(node.get("in_width" if direction == "input" else "out_width", 1))))
                    result[node_id, direction, role] = value
        return result

    @property
    def done(self) -> bool:
        return self.size_index >= len(self.sizes)

    def state_dict(self) -> dict[str, Any]:
        return {"size_index": self.size_index, "root_index": self.root_index, "count": self.count, "stack": self.stack}

    def step(self) -> _Occurrence | None:
        if self.done:
            return None
        size = self.sizes[self.size_index]
        if size > len(self.nodes) or self.count >= self.cap or (not self.stack and self.root_index >= len(self.nodes)):
            self.size_index += 1
            self.root_index, self.count, self.stack = 0, 0, []
            return None
        if not self.stack:
            root = self.nodes[self.root_index]
            self.root_index += 1
            extension = sorted(node for node in self.skeleton[root] if self.ranks[node] > self.ranks[root])
            self.stack.append([[root], extension, self.ranks[root]])
            return None
        members, extension, root_rank = self.stack[-1]
        if len(members) == size:
            self.stack.pop()
            self.count += 1
            return self._occurrence(tuple(sorted(members)))
        if not extension:
            self.stack.pop()
            return None
        candidate = extension.pop(0)
        neighborhood = {neighbor for node in members for neighbor in self.skeleton[node]}
        grown = extension + sorted(node for node in self.skeleton[candidate] if self.ranks[node] > root_rank and node not in members and node not in neighborhood)
        self.stack.append([members + [candidate], grown, root_rank])
        return None

    def _occurrence(self, members: tuple[int, ...]) -> _Occurrence:
        local = {node: i for i, node in enumerate(members)}
        labels = [self.labels[node] for node in members]
        edges = [(local[source], local[target], self.edges[source, target]) for source in members for _, target in self.outgoing[source] if target in local]
        motif = canonical_form(labels, edges)
        mapping = _canonical_mapping(labels, edges, motif)
        ports = _ports_for_occurrence(self.entry, self.labels, self.edges, members, mapping, index=self)
        return _Occurrence(self.entry.key, self.root, motif, ports)


def occurrence_dict(item: _Occurrence) -> dict[str, Any]:
    return {"entry": item.entry, "root": item.lineage_root, "nodes": item.motif.node_labels, "edges": item.motif.edges, "ports": [port.to_dict() for port in item.ports]}


def occurrence_from_dict(row: dict[str, Any]) -> _Occurrence:
    return _Occurrence(
        row["entry"],
        row["root"],
        MotifGraph(tuple(tuple(value) for value in row["nodes"]), tuple(tuple(value) for value in row["edges"])),
        tuple(BoundaryPort.from_dict(port) for port in row["ports"]),
    )


class GrammarPreparation:
    """
    Mine a stable catalog incrementally, reusing complete immutable-entry evidence.
    """

    def __init__(self, library: ModuleLibrary, params: dict[str, Any], saved: dict[str, Any] | None = None) -> None:
        self.library, self.params = library, params
        self.state: dict[str, Any] = saved or {
            "version": 1,
            "token": catalog_token(library),
            "keys": sorted(library.keys()),
            "selected": sorted(row["key"] for row in library.summaries()),
            "metadata": {},
            "position": 0,
            "phase": "metadata",
            "groups": {},
            "current": [],
            "cursor": None,
            "productions": [],
            "cached": None,
            "cached_position": 0,
        }
        self.cursor: OccurrenceCursor | None = None
        self.group_keys: list[str] = []
        self.grammar: Grammar | None = None

    def state_dict(self) -> dict[str, Any]:
        if self.cursor is not None:
            self.state["cursor"] = self.cursor.state_dict()
        return self.state

    def advance(self, *, should_stop: Any, items: int = 128, seconds: float = 0.1) -> bool:
        end = time.perf_counter() + seconds
        for _ in range(items):
            if should_stop() or time.perf_counter() >= end:
                break
            if self.step():
                return True
        return self.grammar is not None

    def _cache_path(self, key: str):
        return self.library.root / "grammar" / "evidence" / f"{_sha1([1, key, self.params, self.state['metadata'][key], self.state['roots'][key]])}.json"

    def step(self) -> bool:
        state = self.state
        position = state["position"]
        if state["phase"] == "metadata":
            if position < len(state["keys"]):
                key = state["keys"][position]
                entry = self.library.load(key)
                state["metadata"][key] = {"search_lineage": entry.provenance.get("search_lineage"), "refined_from": entry.provenance.get("refined_from"), "io": entry.io}
                state["position"] += 1
                return False
            entries = {key: SimpleNamespace(provenance=value) for key, value in state["metadata"].items()}
            state["roots"] = {key: _lineage_root(key, entries) for key in state["selected"]}
            state["eligible"] = state["selected"] if len(set(state["roots"].values())) >= self.params["min_lineage_support"] else []
            state["position"], state["phase"] = 0, "mine"
            return False
        if state["phase"] == "mine":
            if position >= len(state["eligible"]):
                state["position"], state["phase"] = 0, "promote"
                return False
            key = state["eligible"][position]
            if state["cached"] is not None:
                rows = state["cached"]
                if state["cached_position"] < len(rows):
                    self._group(occurrence_from_dict(rows[state["cached_position"]]))
                    state["cached_position"] += 1
                else:
                    state["cached"], state["cached_position"] = None, 0
                    state["position"] += 1
                return False
            if self.cursor is None:
                path = self._cache_path(key)
                if path.exists():
                    state["cached"] = json.loads(path.read_text())
                    return False
                entry = self.library.load(key)
                if "field_template" in entry.payload:
                    state["position"] += 1
                    return False
                sizes = self.params["module_sizes"] if entry.entry_type == MODULE else self.params["composition_sizes"]
                self.cursor = OccurrenceCursor(entry, tuple(sorted(set(sizes))), self.params["per_entry_cap"], state["roots"][key], state["cursor"])
            occurrence = self.cursor.step()
            if occurrence is not None:
                state["current"].append(occurrence_dict(occurrence))
                self._group(occurrence)
            if self.cursor.done:
                path = self._cache_path(key)
                path.parent.mkdir(parents=True, exist_ok=True)
                self.library._write_json(path, state["current"])
                state["current"] = []
                state["cursor"], self.cursor = None, None
                state["position"] += 1
            return False
        if not self.group_keys:
            self.group_keys = sorted(state["groups"])
        if position < len(self.group_keys):
            group = state["groups"][self.group_keys[position]]
            roots = tuple(sorted(group["roots"]))
            first = occurrence_from_dict(group["first"])
            cost = len(first.motif.node_labels) + len(first.motif.edges)
            gain = len(roots) * cost - cost - len(first.ports) - len(roots)
            if len(roots) >= self.params["min_lineage_support"] and gain > 0:
                kind = group["kind"]
                production = Production(
                    _production_key(kind, first.motif, first.ports), kind, first.motif, first.ports, roots, tuple(sorted(group["entries"])), len(roots), group["count"], gain
                )
                state["productions"].append(production.to_dict())
            state["position"] += 1
            return False
        source = [{"key": key, "refined_from": state["metadata"][key]["refined_from"]} for key in state["selected"]]
        self.grammar = Grammar(
            tuple(sorted((Production.from_dict(row) for row in state["productions"]), key=lambda item: item.key)), _sha1(source), tuple(state["selected"]), **self.params
        )
        save_grammar(self.grammar, self.library)
        return True

    def _group(self, occurrence: _Occurrence) -> None:
        kind = self.library.load(occurrence.entry).entry_type
        identity = repr((kind, motif_fingerprint(occurrence.motif), occurrence.ports))
        group = self.state["groups"].setdefault(identity, {"kind": kind, "first": occurrence_dict(occurrence), "roots": {}, "entries": {}, "count": 0})
        group["roots"][occurrence.lineage_root] = True
        group["entries"][occurrence.entry] = True
        group["count"] += 1
