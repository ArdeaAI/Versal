"""Exact, weight-independent topology identity and a persistent refinement tabu ledger."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import networkx as nx
import torch
from networkx.algorithms.isomorphism import categorical_node_match

if TYPE_CHECKING:
    from ardevo.dataset.icarus import Task
    from ardevo.library import ModuleLibrary

TOPOLOGY_SCHEMA_VERSION = 2
_NODE_MATCH = categorical_node_match("label", "")
_EDGE_MATCH = nx.algorithms.isomorphism.categorical_edge_match("label", "")

# Exact graph isomorphism is only the fallback for small, reordered payloads.  Large neural
# topologies contain thousands of semantically identical connection positions; VF2 can spend hours
# proving that even a graph compared with itself is isomorphic.  Above this boundary the ledger is
# deliberately conservative: normalized payload equality can prove a duplicate, otherwise the
# candidate is retained as unseen.  A false negative costs an assessment; a false positive would
# incorrectly discard a potentially useful architecture.
_MAX_EXACT_ISOMORPHISM_NODES = 4096


@dataclass(frozen=True)
class TopologyRecord:
    bucket: str
    graph: dict[str, Any]


def _label(category: str, **attributes: Any) -> str:
    return json.dumps([category, sorted(attributes.items())], sort_keys=True, separators=(",", ":"), default=str)


def _add_relation(graph: nx.DiGraph, source: int, target: int, kind: str, **attributes: Any) -> None:
    relation = max(graph.nodes, default=-1) + 1
    graph.add_node(relation, label=_label(f"relation:{kind}", **attributes))
    graph.add_edge(source, relation, label="")
    graph.add_edge(relation, target, label="")


def _nested_reference_label(reference: str, library: ModuleLibrary | None, visiting: frozenset[str]) -> str:
    if not reference.startswith("library:") or library is None:
        return reference
    key = reference.removeprefix("library:")
    if key in visiting:
        return "cycle"
    try:
        entry = library.load(key)
    except KeyError:
        return "missing"
    nested = topology_record(entry.entry_type, entry.payload, library=library, visiting=visiting | {key})
    return f"topology:{nested.bucket}"


def _module_graph(payload: dict[str, Any], library: ModuleLibrary | None, visiting: frozenset[str]) -> nx.DiGraph:
    graph = nx.DiGraph()
    nodes = list(payload.get("nodes", []))
    positions: dict[tuple[str, int], int] = {}
    by_kind: dict[str, list[int]] = {}
    for node in nodes:
        by_kind.setdefault(str(node.get("kind", "hidden")), []).append(int(node["id"]))
    for kind, identifiers in by_kind.items():
        for position, identifier in enumerate(sorted(identifiers)):
            positions[(kind, identifier)] = position
    node_map: dict[int, int] = {}
    for node in nodes:
        identifier = int(node["id"])
        kind = str(node.get("kind", "hidden"))
        graph_id = len(graph)
        node_map[identifier] = graph_id
        ordered_port = positions[(kind, identifier)] if kind in {"input", "bias", "output"} else None
        graph.add_node(
            graph_id,
            label=_label(
                "module_node",
                kind=kind,
                port=ordered_port,
                activation=str(node.get("activation", "identity")),
                aggregation=str(node.get("aggregation", "sum")),
                coordinate=node.get("coordinate"),
            ),
        )

    connections = list(payload.get("connections", []))
    pair_counts: dict[tuple[int, int], int] = {}
    for connection in connections:
        pair = (int(connection["in"]), int(connection["out"]))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    tie_nodes: dict[int, int] = {}
    for connection in connections:
        source = node_map.get(int(connection["in"]))
        target = node_map.get(int(connection["out"]))
        if source is None or target is None:
            continue
        # Genome payloads serialize this as ``tie``; accept ``tie_group`` as well for
        # hand-authored/future schemas. The numeric marker itself is an innovation-like id, while
        # the partition it induces across connections is architectural and must be preserved.
        tie_group = connection.get("tie", connection.get("tie_group"))
        connection_label = _label("module_connection", enabled=bool(connection.get("enabled", True)), recurrent=bool(connection.get("recurrent", False)))
        # The ordinary case is one untied gene per endpoint pair.  Store its semantics directly on
        # the edge rather than expanding one indistinguishable relation node per gene.  Parallel
        # forward/recurrent genes and tied genes retain relation nodes so multiplicity and tie
        # partitions remain exact in a plain DiGraph.
        if tie_group is None and pair_counts[(int(connection["in"]), int(connection["out"]))] == 1:
            graph.add_edge(source, target, label=connection_label)
            continue
        relation = max(graph.nodes, default=-1) + 1
        graph.add_node(relation, label=connection_label)
        graph.add_edge(source, relation, label="")
        graph.add_edge(relation, target, label="")
        if tie_group is not None:
            tie_id = int(tie_group)
            if tie_id not in tie_nodes:
                tie_nodes[tie_id] = max(graph.nodes, default=-1) + 1
                graph.add_node(tie_nodes[tie_id], label=_label("weight_tie_group"))
            graph.add_edge(tie_nodes[tie_id], relation, label="")

    for macro in payload.get("macros", []):
        macro_node = max(graph.nodes, default=-1) + 1
        graph.add_node(
            macro_node,
            label=_label(
                "macro",
                reference=_nested_reference_label(str(macro.get("ref", "")), library, visiting),
                trainable=bool(macro.get("trainable", False)),
            ),
        )
        for position, identifier in enumerate(macro.get("inputs", [])):
            if int(identifier) in node_map:
                _add_relation(graph, node_map[int(identifier)], macro_node, "macro_input", position=position)
        for position, identifier in enumerate(macro.get("outputs", [])):
            if int(identifier) in node_map:
                _add_relation(graph, macro_node, node_map[int(identifier)], "macro_output", position=position)
    graph.add_node(max(graph.nodes, default=-1) + 1, label=_label("refine_steps", value=int(payload.get("refine_steps", 1))))
    return graph


def _composition_graph(payload: dict[str, Any], library: ModuleLibrary | None, visiting: frozenset[str]) -> nx.DiGraph:
    graph = nx.DiGraph()
    nodes = list(payload.get("nodes", []))
    by_kind: dict[str, list[int]] = {}
    for node in nodes:
        by_kind.setdefault(str(node.get("kind", "module")), []).append(int(node["id"]))
    positions = {(kind, identifier): position for kind, identifiers in by_kind.items() for position, identifier in enumerate(sorted(identifiers))}
    node_map: dict[int, int] = {}
    for node in nodes:
        identifier = int(node["id"])
        kind = str(node.get("kind", "module"))
        reference = str(node.get("ref", ""))
        if kind == "module":
            reference = _nested_reference_label(reference, library, visiting)
        elif kind == "output":
            reference = "output"
        graph_id = len(graph)
        node_map[identifier] = graph_id
        graph.add_node(
            graph_id,
            label=_label(
                "composition_node",
                kind=kind,
                port=positions[(kind, identifier)] if kind in {"input", "output"} else None,
                reference=reference,
                in_width=int(node.get("in_width", 0)),
                out_width=int(node.get("out_width", 0)),
                aggregation=str(node.get("aggregation", "sum")),
                trainable=bool(node.get("trainable", True)),
            ),
        )
    for edge in payload.get("edges", []):
        source = node_map.get(int(edge["in"]))
        target = node_map.get(int(edge["out"]))
        if source is None or target is None:
            continue
        port_map = tuple((int(run["source_start"]), int(run["target_start"]), int(run["length"])) for run in edge.get("port_map", []))
        _add_relation(
            graph,
            source,
            target,
            "composition_edge",
            enabled=bool(edge.get("enabled", True)),
            glue_rank=int(edge.get("glue_rank", 0)),
            port_map=port_map,
        )
    return graph


def _graph_payload(graph: nx.DiGraph) -> dict[str, Any]:
    return {
        "nodes": [{"id": int(node), "label": str(attributes["label"])} for node, attributes in graph.nodes(data=True)],
        "edges": [{"source": int(source), "target": int(target), "label": str(attributes.get("label", ""))} for source, target, attributes in graph.edges(data=True)],
    }


def _payload_graph(payload: dict[str, Any]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in payload["nodes"]:
        graph.add_node(int(node["id"]), label=str(node["label"]))
    for edge in payload["edges"]:
        if isinstance(edge, dict):
            graph.add_edge(int(edge["source"]), int(edge["target"]), label=str(edge.get("label", "")))
        else:  # Schema-v1 rows remain readable while versioned contexts prevent cross-schema use.
            source, target = edge
            graph.add_edge(int(source), int(target), label="")
    return graph


def topology_record(
    entry_type: str,
    payload: dict[str, Any],
    *,
    library: ModuleLibrary | None = None,
    visiting: frozenset[str] = frozenset(),
) -> TopologyRecord:
    graph = _module_graph(payload, library, visiting) if entry_type == "module" else _composition_graph(payload, library, visiting)
    # NetworkX warns that directed WL hashes changed in 3.5. This ledger declares its own schema
    # version and never compares buckets written by a different dependency lock, so that migration
    # notice is not actionable at run time.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The hashes produced for directed graphs changed in version v3.5.*", category=UserWarning)
        bucket = nx.weisfeiler_lehman_graph_hash(graph, node_attr="label", edge_attr="label", iterations=4, digest_size=32)
    return TopologyRecord(bucket=bucket, graph=_graph_payload(graph))


def same_topology(left: TopologyRecord, right: TopologyRecord) -> bool:
    if left.bucket != right.bucket:
        return False
    if left.graph == right.graph:
        return True
    if max(len(left.graph["nodes"]), len(right.graph["nodes"])) > _MAX_EXACT_ISOMORPHISM_NODES:
        return False
    return nx.is_isomorphic(_payload_graph(left.graph), _payload_graph(right.graph), node_match=_NODE_MATCH, edge_match=_EDGE_MATCH)


def task_content_fingerprint(task: Task) -> str:
    from ardevo.dataset.icarus import support_loader
    from ardevo.library import task_io

    digest = hashlib.sha256(json.dumps(task_io(task), sort_keys=True, separators=(",", ":")).encode())
    for field_value in support_loader(task):
        tensor = torch.as_tensor(field_value.data).detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
        descriptor = field_value.descriptor
        digest.update(str(descriptor.value_type.value).encode())
        digest.update(json.dumps([axis.value for axis in descriptor.axes]).encode())
    return digest.hexdigest()


def refinement_lineage_root(library: ModuleLibrary, key: str) -> str:
    current = key
    seen: set[str] = set()
    while current not in seen:
        seen.add(current)
        try:
            parent = library.load(current).provenance.get("refined_from")
        except KeyError:
            break
        if not parent:
            break
        current = str(parent)
    return current


class TopologyTabuStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute("CREATE TABLE IF NOT EXISTS seen (context TEXT NOT NULL, bucket TEXT NOT NULL, graph TEXT NOT NULL)")
        connection.execute("CREATE INDEX IF NOT EXISTS seen_lookup ON seen(context, bucket)")
        return connection

    def load_bucket(self, context: str, bucket: str) -> list[TopologyRecord]:
        if not self.path.exists():
            return []
        with self._connect() as connection:
            return [TopologyRecord(bucket, json.loads(row[0])) for row in connection.execute("SELECT graph FROM seen WHERE context=? AND bucket=?", (context, bucket))]

    def append(self, context: str, records: list[TopologyRecord]) -> None:
        if not records:
            return
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO seen(context, bucket, graph) VALUES (?, ?, ?)",
                [(context, record.bucket, json.dumps(record.graph, sort_keys=True, separators=(",", ":"))) for record in records],
            )


@dataclass
class TopologyTabuSession:
    store: TopologyTabuStore
    context: str
    library: ModuleLibrary
    retry_limit: int = 8
    deadline_exceeded: Callable[[], bool] | None = None
    pending: list[TopologyRecord] = field(default_factory=list)
    _buckets: dict[str, list[TopologyRecord]] = field(default_factory=dict)
    candidates: int = 0
    duplicates: int = 0
    unique: int = 0
    retry_exhaustions: int = 0
    exhausted: bool = False

    def _past_deadline(self) -> bool:
        if self.deadline_exceeded is None or not self.deadline_exceeded():
            return False
        self.exhausted = True
        return True

    def _insert_if_new(self, entry_type: str, payload: dict[str, Any]) -> bool:
        if self._past_deadline():
            return False
        record = topology_record(entry_type, payload, library=self.library)
        if self._past_deadline():
            return False
        known = self._buckets.setdefault(record.bucket, self.store.load_bucket(self.context, record.bucket))
        for previous in known:
            if self._past_deadline():
                return False
            if same_topology(record, previous):
                return False
        known.append(record)
        self.pending.append(record)
        return True

    def prime(self, entry_type: str, payload: dict[str, Any]) -> None:
        """Seed known lineage structure without presenting it as work from this attempt."""

        self._insert_if_new(entry_type, payload)

    def observe_evaluated(self, entry_type: str, payload: dict[str, Any]) -> None:
        """Remember an initial-population topology that was necessarily assessed once."""

        if self._insert_if_new(entry_type, payload):
            self.unique += 1

    def reserve(self, entry_type: str, payload: dict[str, Any]) -> bool:
        if self._past_deadline():
            return False
        self.candidates += 1
        if not self._insert_if_new(entry_type, payload):
            if not self.exhausted:
                self.duplicates += 1
            return False
        self.unique += 1
        return True

    def commit(self) -> None:
        self.store.append(self.context, self.pending)
        self.pending.clear()

    def metrics(self) -> dict[str, float]:
        return {
            "topology_candidates": float(self.candidates),
            "topology_unique_evaluated": float(self.unique),
            "topology_duplicates_skipped": float(self.duplicates),
            "topology_retry_exhaustions": float(self.retry_exhaustions),
            "topology_exhausted": float(self.exhausted),
        }


def refinement_context(*, lineage_root: str, task_fingerprint: str, config_fingerprint: str) -> str:
    source = f"{TOPOLOGY_SCHEMA_VERSION}:{lineage_root}:{task_fingerprint}:{config_fingerprint}"
    return hashlib.sha256(source.encode()).hexdigest()
