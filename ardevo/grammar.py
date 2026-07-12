"""Library-derived typed graph grammar and evolvable programs.

The grammar is rebuildable state: productions are induced from canonical motifs that recur in
independent library lineages, never hand-authored task rules. Programs arrange those productions in
a typed DAG and compile back into the existing Genome/CompositionGenome representations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from typing import Any, Literal, Sequence, cast

from ardevo.evolution.composition import CompEdgeGene, CompNodeGene, CompNodeKind, CompositionGenome
from ardevo.evolution.genome import ConnectionGene, Genome, InnovationTracker, NodeGene, NodeKind, topological_order
from ardevo.library import MODULE, LibraryEntry, ModuleLibrary
from ardevo.motifs import (
    DEFAULT_COMPOSITION_SIZES,
    DEFAULT_MODULE_SIZES,
    DEFAULT_PER_ENTRY_CAP,
    FORWARD_EDGE,
    MACRO_EDGE,
    MAX_MOTIF_SIZE,
    RECURRENT_EDGE,
    TIED_EDGE,
    MotifGraph,
    NodeLabel,
    canonical_form,
    composition_motif_graph,
    enumerate_connected_subgraphs,
    module_motif_graph,
    motif_fingerprint,
)

GRAMMAR_VERSION = 1
PROGRAM_VERSION = 1
GRAMMAR_RELATIVE_PATH = Path("grammar") / "grammar.json"

__all__ = [
    "BoundaryPort",
    "Grammar",
    "GrammarError",
    "Production",
    "Program",
    "ProgramEdge",
    "ProgramNode",
    "compile_program",
    "crossover_program",
    "delete_production",
    "grammar_path",
    "induce_grammar",
    "insert_production",
    "load_grammar",
    "mutate_program",
    "parallelize_production",
    "rebuild_grammar",
    "reconnect_program",
    "repeat_production",
    "replace_production",
    "save_grammar",
    "seed_program",
    "validate_program",
]

PortDirection = Literal["input", "output"]
PortRole = Literal["terminal", "cut"]


class GrammarError(ValueError):
    """Invalid grammar, program, or compilation request."""


@dataclass(frozen=True, slots=True, order=True)
class BoundaryPort:
    """One typed attachment point on a canonical production body."""

    name: str
    direction: PortDirection
    node: int
    signature: str
    width: int
    role: PortRole

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "node": self.node,
            "signature": self.signature,
            "width": self.width,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoundaryPort:
        direction_value = str(data["direction"])
        role_value = str(data["role"])
        if direction_value not in ("input", "output"):
            raise GrammarError(f"unknown port direction {direction_value!r}")
        if role_value not in ("terminal", "cut"):
            raise GrammarError(f"unknown port role {role_value!r}")
        width = int(data["width"])
        if width <= 0:
            raise GrammarError(f"port width must be positive, got {width}")
        direction = cast(PortDirection, direction_value)
        role = cast(PortRole, role_value)
        return cls(str(data["name"]), direction, int(data["node"]), str(data["signature"]), width, role)


@dataclass(frozen=True, slots=True)
class Production:
    """A canonical motif promoted by independent-lineage support and MDL compression."""

    key: str
    source_kind: str
    motif: MotifGraph
    ports: tuple[BoundaryPort, ...]
    lineage_roots: tuple[str, ...]
    exemplars: tuple[str, ...]
    support: int
    occurrences: int
    mdl_gain: int

    def port(self, name: str) -> BoundaryPort:
        for port in self.ports:
            if port.name == name:
                return port
        raise GrammarError(f"production {self.key!r} has no port {name!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "source_kind": self.source_kind,
            "motif_fingerprint": motif_fingerprint(self.motif),
            "body": {
                "nodes": [list(label) for label in self.motif.node_labels],
                "edges": [list(edge) for edge in self.motif.edges],
            },
            "ports": [port.to_dict() for port in self.ports],
            "lineage_roots": list(self.lineage_roots),
            "exemplars": list(self.exemplars),
            "support": self.support,
            "occurrences": self.occurrences,
            "mdl_gain": self.mdl_gain,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Production:
        body = data["body"]
        node_labels: list[NodeLabel] = []
        for label in body["nodes"]:
            if len(label) != 4:
                raise GrammarError(f"motif node label needs four fields, got {label!r}")
            node_labels.append((str(label[0]), str(label[1]), str(label[2]), str(label[3])))
        edges: list[tuple[int, int, int]] = []
        for edge in body["edges"]:
            if len(edge) != 3:
                raise GrammarError(f"motif edge needs three fields, got {edge!r}")
            edges.append((int(edge[0]), int(edge[1]), int(edge[2])))
        motif = MotifGraph(node_labels=tuple(node_labels), edges=tuple(edges))
        if str(data.get("motif_fingerprint", "")) != motif_fingerprint(motif):
            raise GrammarError(f"production {data.get('key')!r} has a corrupt motif fingerprint")
        production = cls(
            key=str(data["key"]),
            source_kind=str(data["source_kind"]),
            motif=motif,
            ports=tuple(
                sorted(
                    (BoundaryPort.from_dict(item) for item in data["ports"]),
                    key=lambda port: (0 if port.direction == "input" else 1, port.node, port.signature, port.width, port.role),
                )
            ),
            lineage_roots=tuple(sorted(str(value) for value in data["lineage_roots"])),
            exemplars=tuple(sorted(str(value) for value in data["exemplars"])),
            support=int(data["support"]),
            occurrences=int(data["occurrences"]),
            mdl_gain=int(data["mdl_gain"]),
        )
        if production.key != _production_key(production.source_kind, production.motif, production.ports):
            raise GrammarError(f"production key {production.key!r} does not match its body and ports")
        return production


@dataclass(frozen=True, slots=True)
class Grammar:
    """Versioned, deterministic grammar derived from one library snapshot."""

    productions: tuple[Production, ...]
    source_fingerprint: str
    source_entries: tuple[str, ...]
    module_sizes: tuple[int, ...] = DEFAULT_MODULE_SIZES
    composition_sizes: tuple[int, ...] = DEFAULT_COMPOSITION_SIZES
    min_lineage_support: int = 2
    per_entry_cap: int = DEFAULT_PER_ENTRY_CAP
    version: int = GRAMMAR_VERSION

    def __post_init__(self) -> None:
        if self.version != GRAMMAR_VERSION:
            raise GrammarError(f"unsupported grammar version {self.version}; expected {GRAMMAR_VERSION}")
        keys = [production.key for production in self.productions]
        if len(keys) != len(set(keys)):
            raise GrammarError("grammar contains duplicate production keys")

    @classmethod
    def empty(cls) -> Grammar:
        return cls(productions=(), source_fingerprint=_sha1([]), source_entries=())

    def production(self, key: str) -> Production:
        for production in self.productions:
            if production.key == key:
                return production
        raise GrammarError(f"grammar has no production {key!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_fingerprint": self.source_fingerprint,
            "source_entries": list(self.source_entries),
            "params": {
                "module_sizes": list(self.module_sizes),
                "composition_sizes": list(self.composition_sizes),
                "min_lineage_support": self.min_lineage_support,
                "per_entry_cap": self.per_entry_cap,
            },
            "productions": [production.to_dict() for production in sorted(self.productions, key=lambda item: item.key)],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Grammar:
        version = int(data["version"])
        if version != GRAMMAR_VERSION:
            raise GrammarError(f"unsupported grammar version {version}; expected {GRAMMAR_VERSION}")
        params = data["params"]
        return cls(
            productions=tuple(sorted((Production.from_dict(item) for item in data["productions"]), key=lambda item: item.key)),
            source_fingerprint=str(data["source_fingerprint"]),
            source_entries=tuple(sorted(str(value) for value in data["source_entries"])),
            module_sizes=tuple(int(value) for value in params["module_sizes"]),
            composition_sizes=tuple(int(value) for value in params["composition_sizes"]),
            min_lineage_support=int(params["min_lineage_support"]),
            per_entry_cap=int(params["per_entry_cap"]),
            version=version,
        )


@dataclass(frozen=True, slots=True, order=True)
class ProgramNode:
    id: int
    production: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "production": self.production}


@dataclass(frozen=True, slots=True, order=True)
class ProgramEdge:
    source: int
    source_port: str
    target: int
    target_port: str

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "source_port": self.source_port, "target": self.target, "target_port": self.target_port}


@dataclass(frozen=True, slots=True)
class Program:
    """A typed DAG over grammar productions."""

    nodes: tuple[ProgramNode, ...]
    edges: tuple[ProgramEdge, ...] = ()
    version: int = PROGRAM_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "nodes": [node.to_dict() for node in sorted(self.nodes)],
            "edges": [edge.to_dict() for edge in sorted(self.edges)],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Program:
        version = int(data["version"])
        if version != PROGRAM_VERSION:
            raise GrammarError(f"unsupported program version {version}; expected {PROGRAM_VERSION}")
        nodes = tuple(sorted(ProgramNode(int(item["id"]), str(item["production"])) for item in data["nodes"]))
        edges = tuple(sorted(ProgramEdge(int(item["source"]), str(item["source_port"]), int(item["target"]), str(item["target_port"])) for item in data["edges"]))
        return cls(nodes=nodes, edges=edges, version=version)


@dataclass(frozen=True, slots=True)
class _Occurrence:
    entry: str
    lineage_root: str
    motif: MotifGraph
    ports: tuple[BoundaryPort, ...]


def _sha1(value: Any, length: int = 16) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:length]


def _production_key(source_kind: str, motif: MotifGraph, ports: Sequence[BoundaryPort]) -> str:
    payload = {
        "source_kind": source_kind,
        "nodes": motif.node_labels,
        "edges": motif.edges,
        "ports": [port.to_dict() for port in ports],
    }
    return f"g1_{_sha1(payload)}"


def _lineage_root(key: str, entries: dict[str, LibraryEntry]) -> str:
    path: list[str] = []
    current = key
    while current not in path:
        path.append(current)
        entry = entries.get(current)
        parent = entry.provenance.get("refined_from") if entry is not None else None
        if not isinstance(parent, str) or not parent:
            return current
        current = parent
    return min(path[path.index(current) :])


def _canonical_mapping(labels: Sequence[NodeLabel], edges: Sequence[tuple[int, int, int]], motif: MotifGraph) -> tuple[int, ...]:
    """Recover the deterministic old-index -> canonical-index map used by canonical_form."""
    positions_by_label: dict[NodeLabel, list[int]] = {}
    for position, label in enumerate(motif.node_labels):
        positions_by_label.setdefault(label, []).append(position)
    indices_by_label: dict[NodeLabel, list[int]] = {}
    for index, label in enumerate(labels):
        indices_by_label.setdefault(label, []).append(index)
    blocks = [(indices_by_label[label], positions_by_label[label]) for label in sorted(indices_by_label)]
    matches: list[tuple[int, ...]] = []
    for assignment in product(*(permutations(indices) for indices, _positions in blocks)):
        mapping = [0] * len(labels)
        for (_, block_positions), ordered_indices in zip(blocks, assignment):
            for position, index in zip(block_positions, ordered_indices):
                mapping[index] = position
        remapped = tuple(sorted((mapping[source], mapping[target], mask) for source, target, mask in edges))
        if remapped == motif.edges:
            matches.append(tuple(mapping))
    if not matches:
        raise GrammarError("canonical motif mapping could not be recovered")
    return min(matches)


def _expanded_signatures(specs: Sequence[dict[str, Any]]) -> list[str]:
    return [str(spec["signature"]) for spec in specs for _ in range(max(0, int(spec["width"])))]


def _module_port_type(entry: LibraryEntry, payload: dict[str, Any], node_id: int, direction: PortDirection, role: PortRole) -> tuple[str, int]:
    nodes = {int(node["id"]): node for node in payload.get("nodes", [])}
    if role == "cut":
        return "scalar", 1
    if direction == "input":
        ids = sorted(candidate for candidate, item in nodes.items() if item["kind"] == "input")
        signatures = _expanded_signatures(entry.io.get("inputs", []))
        index = ids.index(node_id)
        return (signatures[index] if index < len(signatures) else "scalar"), 1
    ids = sorted(candidate for candidate, item in nodes.items() if item["kind"] == "output")
    signature = str(entry.io.get("output", {}).get("signature", "scalar"))
    return signature if node_id in ids else "scalar", 1


def _composition_port_type(entry: LibraryEntry, payload: dict[str, Any], node_id: int, direction: PortDirection, role: PortRole) -> tuple[str, int]:
    node = next(item for item in payload.get("nodes", []) if int(item["id"]) == node_id)
    if role == "terminal" and direction == "input":
        return str(node.get("ref", "input")), max(1, int(node.get("out_width", 1)))
    if role == "terminal":
        return str(entry.io.get("output", {}).get("signature", "output")), max(1, int(node.get("in_width", 1)))
    ref = str(node.get("ref", ""))
    signature = ref if ref and not ref.startswith("library:") else "any"
    width_key = "in_width" if direction == "input" else "out_width"
    return signature, max(1, int(node.get(width_key, 1)))


def _ports_for_occurrence(
    entry: LibraryEntry,
    labels: dict[int, NodeLabel],
    edges: dict[tuple[int, int], int],
    members: tuple[int, ...],
    mapping: tuple[int, ...],
) -> tuple[BoundaryPort, ...]:
    member_set = set(members)
    canonical_of = {node_id: mapping[index] for index, node_id in enumerate(members)}
    raw: dict[tuple[PortDirection, int], tuple[str, int, PortRole]] = {}

    for node_id in members:
        kind = labels[node_id][0]
        if kind == "input":
            raw[("input", canonical_of[node_id])] = ("", 0, "terminal")
        elif kind == "output":
            raw[("output", canonical_of[node_id])] = ("", 0, "terminal")
    for source, target in edges:
        if source not in member_set and target in member_set:
            raw.setdefault(("input", canonical_of[target]), ("", 0, "cut"))
        elif source in member_set and target not in member_set:
            raw.setdefault(("output", canonical_of[source]), ("", 0, "cut"))

    typed: list[tuple[PortDirection, int, str, int, PortRole]] = []
    for (direction, canonical_node), (_signature, _width, role) in raw.items():
        original_node = members[mapping.index(canonical_node)]
        if entry.entry_type == MODULE:
            signature, width = _module_port_type(entry, entry.payload, original_node, direction, role)
        else:
            signature, width = _composition_port_type(entry, entry.payload, original_node, direction, role)
        typed.append((direction, canonical_node, signature, width, role))
    typed.sort(key=lambda item: (0 if item[0] == "input" else 1, item[1], item[2], item[3], item[4]))

    counts = {"input": 0, "output": 0}
    ports: list[BoundaryPort] = []
    for direction, node, signature, width, role in typed:
        prefix = "in" if direction == "input" else "out"
        name = f"{prefix}{counts[direction]}"
        counts[direction] += 1
        ports.append(BoundaryPort(name, direction, node, signature, width, role))
    return tuple(ports)


def _entry_occurrences(entry: LibraryEntry, sizes: tuple[int, ...], cap: int, root: str) -> list[_Occurrence]:
    if entry.entry_type == MODULE:
        labels, edges = module_motif_graph(entry.payload)
    else:
        labels, edges = composition_motif_graph(entry.payload)
    skeleton: dict[int, set[int]] = {node_id: set() for node_id in labels}
    for source, target in edges:
        if source != target:
            skeleton[source].add(target)
            skeleton[target].add(source)

    occurrences: list[_Occurrence] = []
    for size in sizes:
        if size > len(labels):
            continue
        subsets, _truncated = enumerate_connected_subgraphs(skeleton, size, cap)
        for subset in subsets:
            members = tuple(sorted(subset))
            local_index = {node_id: index for index, node_id in enumerate(members)}
            local_labels = [labels[node_id] for node_id in members]
            local_edges = [(local_index[source], local_index[target], mask) for (source, target), mask in edges.items() if source in subset and target in subset]
            motif = canonical_form(local_labels, local_edges)
            mapping = _canonical_mapping(local_labels, local_edges, motif)
            ports = _ports_for_occurrence(entry, labels, edges, members, mapping)
            occurrences.append(_Occurrence(entry.key, root, motif, ports))
    return occurrences


def induce_grammar(
    library: ModuleLibrary,
    *,
    module_sizes: tuple[int, ...] = DEFAULT_MODULE_SIZES,
    composition_sizes: tuple[int, ...] = DEFAULT_COMPOSITION_SIZES,
    min_lineage_support: int = 2,
    per_entry_cap: int = DEFAULT_PER_ENTRY_CAP,
    include_retired: bool = False,
) -> Grammar:
    """Mine and promote deterministic productions from independent solution lineages."""
    module_sizes = tuple(sorted(set(module_sizes)))
    composition_sizes = tuple(sorted(set(composition_sizes)))
    if min_lineage_support < 2:
        raise GrammarError("min_lineage_support must be at least 2")
    if per_entry_cap <= 0:
        raise GrammarError("per_entry_cap must be positive")
    if any(size < 2 or size > MAX_MOTIF_SIZE for size in (*module_sizes, *composition_sizes)):
        raise GrammarError(f"grammar motif sizes must be in 2..{MAX_MOTIF_SIZE}")

    all_entries = {key: library.load(key) for key in library.keys()}
    selected_keys = [str(row["key"]) for row in library.summaries(include_retired=include_retired)]
    selected_keys.sort()
    groups: dict[tuple[str, str, tuple[BoundaryPort, ...]], list[_Occurrence]] = {}
    for key in selected_keys:
        entry = all_entries[key]
        sizes = module_sizes if entry.entry_type == MODULE else composition_sizes
        root = _lineage_root(key, all_entries)
        for occurrence in _entry_occurrences(entry, sizes, per_entry_cap, root):
            group_key = (entry.entry_type, motif_fingerprint(occurrence.motif), occurrence.ports)
            groups.setdefault(group_key, []).append(occurrence)

    productions: list[Production] = []
    for (source_kind, _motif_key, ports), occurrences in sorted(groups.items(), key=lambda item: repr(item[0])):
        lineage_roots = tuple(sorted({occurrence.lineage_root for occurrence in occurrences}))
        if len(lineage_roots) < min_lineage_support:
            continue
        motif = occurrences[0].motif
        body_cost = len(motif.node_labels) + len(motif.edges)
        expanded_cost = len(lineage_roots) * body_cost
        compressed_cost = body_cost + len(ports) + len(lineage_roots)
        mdl_gain = expanded_cost - compressed_cost
        if mdl_gain <= 0:
            continue
        productions.append(
            Production(
                key=_production_key(source_kind, motif, ports),
                source_kind=source_kind,
                motif=motif,
                ports=ports,
                lineage_roots=lineage_roots,
                exemplars=tuple(sorted({occurrence.entry for occurrence in occurrences})),
                support=len(lineage_roots),
                occurrences=len(occurrences),
                mdl_gain=mdl_gain,
            )
        )

    source_description = [{"key": key, "refined_from": all_entries[key].provenance.get("refined_from")} for key in selected_keys]
    return Grammar(
        productions=tuple(sorted(productions, key=lambda item: item.key)),
        source_fingerprint=_sha1(source_description),
        source_entries=tuple(selected_keys),
        module_sizes=tuple(module_sizes),
        composition_sizes=tuple(composition_sizes),
        min_lineage_support=min_lineage_support,
        per_entry_cap=per_entry_cap,
    )


def grammar_path(root: ModuleLibrary | str | Path) -> Path:
    base = root.root if isinstance(root, ModuleLibrary) else Path(root)
    return base / GRAMMAR_RELATIVE_PATH


def save_grammar(grammar: Grammar, root: ModuleLibrary | str | Path) -> Path:
    """Atomically persist grammar.json; a crash leaves the previous complete file intact."""
    path = grammar_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(grammar.to_dict(), indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def load_grammar(root: ModuleLibrary | str | Path, *, missing_ok: bool = False) -> Grammar:
    path = grammar_path(root)
    if not path.exists():
        if missing_ok:
            return Grammar.empty()
        raise FileNotFoundError(path)
    return Grammar.from_dict(json.loads(path.read_text()))


def rebuild_grammar(library: ModuleLibrary, **params: Any) -> Grammar:
    grammar = induce_grammar(library, **params)
    save_grammar(grammar, library)
    return grammar


def _compatible(source: BoundaryPort, target: BoundaryPort) -> bool:
    generic = {"", "*", "any", "scalar"}
    return source.direction == "output" and target.direction == "input" and (source.signature == target.signature or source.signature in generic or target.signature in generic)


def validate_program(program: Program, grammar: Grammar) -> None:
    if program.version != PROGRAM_VERSION:
        raise GrammarError(f"unsupported program version {program.version}; expected {PROGRAM_VERSION}")
    if not program.nodes:
        raise GrammarError("program must contain at least one production node")
    node_map = {node.id: node for node in program.nodes}
    if len(node_map) != len(program.nodes) or any(node_id < 0 for node_id in node_map):
        raise GrammarError("program node ids must be unique non-negative integers")
    productions = {node.id: grammar.production(node.production) for node in program.nodes}
    if len(set(program.edges)) != len(program.edges):
        raise GrammarError("program contains duplicate edges")
    if len({(edge.source, edge.target) for edge in program.edges}) != len(program.edges):
        raise GrammarError("program may contain at most one edge between a production pair")
    claimed_inputs: set[tuple[int, str]] = set()
    incoming = {node_id: 0 for node_id in node_map}
    adjacency: dict[int, list[int]] = {}
    for edge in program.edges:
        if edge.source not in node_map or edge.target not in node_map:
            raise GrammarError(f"program edge {edge} names a missing node")
        if edge.source == edge.target:
            raise GrammarError("program self-edges are not allowed")
        source_port = productions[edge.source].port(edge.source_port)
        target_port = productions[edge.target].port(edge.target_port)
        if not _compatible(source_port, target_port):
            raise GrammarError(f"incompatible program ports {source_port.signature!r} -> {target_port.signature!r}")
        claim = (edge.target, edge.target_port)
        if claim in claimed_inputs:
            raise GrammarError(f"input port {edge.target}:{edge.target_port} is connected more than once")
        claimed_inputs.add(claim)
        incoming[edge.target] += 1
        adjacency.setdefault(edge.source, []).append(edge.target)
    ready = sorted(node_id for node_id, count in incoming.items() if count == 0)
    visited = 0
    while ready:
        current = ready.pop(0)
        visited += 1
        for target in sorted(adjacency.get(current, [])):
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
                ready.sort()
    if visited != len(node_map):
        raise GrammarError("program graph contains a cycle")


def seed_program(production: Production | str) -> Program:
    key = production.key if isinstance(production, Production) else production
    return Program(nodes=(ProgramNode(0, key),))


def _normalized_program(nodes: Sequence[ProgramNode], edges: Sequence[ProgramEdge]) -> Program:
    return Program(nodes=tuple(sorted(nodes)), edges=tuple(sorted(edges)))


def insert_production(program: Program, grammar: Grammar, *, rng: random.Random) -> Program:
    validate_program(program, grammar)
    if not program.edges:
        return program
    candidates: list[tuple[ProgramEdge, Production, BoundaryPort, BoundaryPort]] = []
    for edge in program.edges:
        source = grammar.production(next(node.production for node in program.nodes if node.id == edge.source)).port(edge.source_port)
        target = grammar.production(next(node.production for node in program.nodes if node.id == edge.target)).port(edge.target_port)
        for production in grammar.productions:
            for input_port in (port for port in production.ports if port.direction == "input"):
                for output_port in (port for port in production.ports if port.direction == "output"):
                    if _compatible(source, input_port) and _compatible(output_port, target):
                        candidates.append((edge, production, input_port, output_port))
    if not candidates:
        return program
    edge, production, input_port, output_port = candidates[rng.randrange(len(candidates))]
    node_id = max(node.id for node in program.nodes) + 1
    edges = [candidate for candidate in program.edges if candidate != edge]
    edges.extend(
        [
            ProgramEdge(edge.source, edge.source_port, node_id, input_port.name),
            ProgramEdge(node_id, output_port.name, edge.target, edge.target_port),
        ]
    )
    return _normalized_program([*program.nodes, ProgramNode(node_id, production.key)], edges)


def replace_production(program: Program, grammar: Grammar, *, rng: random.Random) -> Program:
    validate_program(program, grammar)
    candidates: list[tuple[int, Production]] = []
    for node in program.nodes:
        current = grammar.production(node.production)
        interface = tuple((port.name, port.direction, port.signature, port.width) for port in current.ports)
        for production in grammar.productions:
            replacement = tuple((port.name, port.direction, port.signature, port.width) for port in production.ports)
            if production.key != current.key and replacement == interface:
                candidates.append((node.id, production))
    if not candidates:
        return program
    node_id, production = candidates[rng.randrange(len(candidates))]
    nodes = [ProgramNode(node.id, production.key if node.id == node_id else node.production) for node in program.nodes]
    return _normalized_program(nodes, program.edges)


def delete_production(program: Program, grammar: Grammar, *, rng: random.Random) -> Program:
    validate_program(program, grammar)
    if len(program.nodes) <= 1:
        return program
    candidates = list(program.nodes)
    rng.shuffle(candidates)
    for node in candidates:
        inbound = [edge for edge in program.edges if edge.target == node.id]
        outbound = [edge for edge in program.edges if edge.source == node.id]
        edges = [edge for edge in program.edges if edge.source != node.id and edge.target != node.id]
        if len(inbound) == len(outbound) == 1:
            source_production = grammar.production(next(item.production for item in program.nodes if item.id == inbound[0].source))
            target_production = grammar.production(next(item.production for item in program.nodes if item.id == outbound[0].target))
            source_port = source_production.port(inbound[0].source_port)
            target_port = target_production.port(outbound[0].target_port)
            if not _compatible(source_port, target_port):
                continue
            edges.append(ProgramEdge(inbound[0].source, inbound[0].source_port, outbound[0].target, outbound[0].target_port))
        candidate = _normalized_program([item for item in program.nodes if item.id != node.id], edges)
        try:
            validate_program(candidate, grammar)
        except GrammarError:
            continue
        return candidate
    return program


def reconnect_program(program: Program, grammar: Grammar, *, rng: random.Random) -> Program:
    validate_program(program, grammar)
    if not program.edges:
        return program
    removed = program.edges[rng.randrange(len(program.edges))]
    remaining = [edge for edge in program.edges if edge != removed]
    occupied_inputs = {(edge.target, edge.target_port) for edge in remaining}
    occupied_pairs = {(edge.source, edge.target) for edge in remaining}
    options: list[ProgramEdge] = []
    for source_node in program.nodes:
        for target_node in program.nodes:
            if source_node.id == target_node.id or (source_node.id, target_node.id) in occupied_pairs:
                continue
            source_production = grammar.production(source_node.production)
            target_production = grammar.production(target_node.production)
            for source_port in (port for port in source_production.ports if port.direction == "output"):
                for target_port in (port for port in target_production.ports if port.direction == "input"):
                    if (target_node.id, target_port.name) not in occupied_inputs and _compatible(source_port, target_port):
                        options.append(ProgramEdge(source_node.id, source_port.name, target_node.id, target_port.name))
    rng.shuffle(options)
    for option in options:
        candidate = _normalized_program(program.nodes, [*remaining, option])
        try:
            validate_program(candidate, grammar)
        except GrammarError:
            continue
        return candidate
    return program


def repeat_production(program: Program, grammar: Grammar, *, rng: random.Random) -> Program:
    """Insert another copy of an adjacent rule on a compatible program edge."""
    validate_program(program, grammar)
    candidates: list[tuple[ProgramEdge, Production, BoundaryPort, BoundaryPort]] = []
    by_id = {node.id: node for node in program.nodes}
    for edge in program.edges:
        source_port = grammar.production(by_id[edge.source].production).port(edge.source_port)
        target_port = grammar.production(by_id[edge.target].production).port(edge.target_port)
        for production_key in (by_id[edge.source].production, by_id[edge.target].production):
            production = grammar.production(production_key)
            for input_port in (port for port in production.ports if port.direction == "input"):
                for output_port in (port for port in production.ports if port.direction == "output"):
                    if _compatible(source_port, input_port) and _compatible(output_port, target_port):
                        candidates.append((edge, production, input_port, output_port))
    if not candidates:
        return program
    edge, production, input_port, output_port = candidates[rng.randrange(len(candidates))]
    node_id = max(node.id for node in program.nodes) + 1
    edges = [candidate for candidate in program.edges if candidate != edge]
    edges.extend([ProgramEdge(edge.source, edge.source_port, node_id, input_port.name), ProgramEdge(node_id, output_port.name, edge.target, edge.target_port)])
    return _normalized_program([*program.nodes, ProgramNode(node_id, production.key)], edges)


def parallelize_production(program: Program, grammar: Grammar, *, rng: random.Random) -> Program:
    """Duplicate a branch point, copying its inputs and splitting its outputs across both copies."""
    validate_program(program, grammar)
    candidates = [node for node in program.nodes if sum(edge.source == node.id for edge in program.edges) >= 2]
    if not candidates:
        return program
    node = candidates[rng.randrange(len(candidates))]
    node_id = max(item.id for item in program.nodes) + 1
    inbound = [edge for edge in program.edges if edge.target == node.id]
    outbound = [edge for edge in program.edges if edge.source == node.id]
    rng.shuffle(outbound)
    moved = set(outbound[: max(1, len(outbound) // 2)])
    edges = list(program.edges)
    edges.extend(ProgramEdge(edge.source, edge.source_port, node_id, edge.target_port) for edge in inbound)
    edges = [ProgramEdge(node_id, edge.source_port, edge.target, edge.target_port) if edge in moved else edge for edge in edges]
    candidate = _normalized_program([*program.nodes, ProgramNode(node_id, node.production)], edges)
    validate_program(candidate, grammar)
    return candidate


def mutate_program(program: Program, grammar: Grammar, *, rng: random.Random, operator: str | None = None) -> Program:
    operators = {
        "insert": insert_production,
        "delete": delete_production,
        "replace": replace_production,
        "reconnect": reconnect_program,
        "repeat": repeat_production,
        "parallelize": parallelize_production,
    }
    if operator is not None and operator not in operators:
        raise GrammarError(f"unknown program mutation {operator!r}")
    selected = operator or rng.choice(sorted(operators))
    return operators[selected](program, grammar, rng=rng)


def crossover_program(parent_a: Program, parent_b: Program, grammar: Grammar, *, rng: random.Random) -> Program:
    """Innovation-like aligned crossover; parent_a supplies disjoint structure."""
    validate_program(parent_a, grammar)
    validate_program(parent_b, grammar)
    by_id_b = {node.id: node for node in parent_b.nodes}
    nodes: list[ProgramNode] = []
    for node_a in parent_a.nodes:
        node_b = by_id_b.get(node_a.id)
        if node_b is None or rng.random() >= 0.5:
            nodes.append(node_a)
            continue
        interface_a = tuple((port.name, port.direction, port.signature, port.width) for port in grammar.production(node_a.production).ports)
        interface_b = tuple((port.name, port.direction, port.signature, port.width) for port in grammar.production(node_b.production).ports)
        nodes.append(node_b if interface_a == interface_b else node_a)
    child = _normalized_program(nodes, parent_a.edges)
    validate_program(child, grammar)
    return child


def _dangling_ports(program: Program, grammar: Grammar) -> tuple[list[tuple[int, BoundaryPort]], list[tuple[int, BoundaryPort]]]:
    used_inputs = {(edge.target, edge.target_port) for edge in program.edges}
    used_outputs = {(edge.source, edge.source_port) for edge in program.edges}
    inputs: list[tuple[int, BoundaryPort]] = []
    outputs: list[tuple[int, BoundaryPort]] = []
    for node in sorted(program.nodes):
        for port in grammar.production(node.production).ports:
            if port.direction == "input" and (node.id, port.name) not in used_inputs:
                inputs.append((node.id, port))
            elif port.direction == "output" and (node.id, port.name) not in used_outputs:
                outputs.append((node.id, port))
    return inputs, outputs


def _compile_as_genome(program: Program, grammar: Grammar, tracker: InnovationTracker | None) -> Genome:
    productions = {node.id: grammar.production(node.production) for node in program.nodes}
    for production in productions.values():
        unsupported_edges = MACRO_EDGE | RECURRENT_EDGE | TIED_EDGE
        if production.source_kind != MODULE or any(mask & unsupported_edges for _source, _target, mask in production.motif.edges):
            raise GrammarError("program contains a production that cannot expand into a flat Genome")
        if any(label[0] not in {"input", "hidden", "output"} for label in production.motif.node_labels):
            raise GrammarError("program contains a non-module node kind")
        if any(port.width != 1 for port in production.ports):
            raise GrammarError("flat Genome compilation requires scalar production ports")

    dangling_inputs, dangling_outputs = _dangling_ports(program, grammar)
    if not dangling_inputs or not dangling_outputs:
        raise GrammarError("compiled program needs at least one dangling input and output port")
    used_inputs = {(edge.target, edge.target_port) for edge in program.edges}
    used_outputs = {(edge.source, edge.source_port) for edge in program.edges}
    innovations = tracker or InnovationTracker(_next_node_id=0)
    node_ids: dict[tuple[int, int], int] = {}
    nodes: dict[int, NodeGene] = {}
    for program_node in sorted(program.nodes):
        production = productions[program_node.id]
        terminal_inputs = {port.node for port in production.ports if port.direction == "input" and port.role == "terminal" and (program_node.id, port.name) not in used_inputs}
        terminal_outputs = {port.node for port in production.ports if port.direction == "output" and port.role == "terminal" and (program_node.id, port.name) not in used_outputs}
        for local_id, label in enumerate(production.motif.node_labels):
            kind = NodeKind.HIDDEN
            if local_id in terminal_inputs:
                kind = NodeKind.INPUT
            elif local_id in terminal_outputs:
                kind = NodeKind.OUTPUT
            node_id = innovations.new_node_id()
            node_ids[(program_node.id, local_id)] = node_id
            nodes[node_id] = NodeGene(node_id, kind, label[1], None, label[2])

    connections: list[ConnectionGene] = []
    for program_node in sorted(program.nodes):
        production = productions[program_node.id]
        for source, target, mask in production.motif.edges:
            source_id = node_ids[(program_node.id, source)]
            target_id = node_ids[(program_node.id, target)]
            if mask & FORWARD_EDGE:
                connections.append(ConnectionGene(source_id, target_id, 1.0, True, innovations.innovation(source_id, target_id)))

    by_program_node = {node.id: node for node in program.nodes}
    for edge in program.edges:
        source_port = productions[edge.source].port(edge.source_port)
        target_port = productions[edge.target].port(edge.target_port)
        source_id = node_ids[(by_program_node[edge.source].id, source_port.node)]
        target_id = node_ids[(by_program_node[edge.target].id, target_port.node)]
        connections.append(ConnectionGene(source_id, target_id, 1.0, True, innovations.innovation(source_id, target_id)))

    for program_node_id, port in [*dangling_inputs, *dangling_outputs]:
        if port.role != "cut":
            continue
        anchor = node_ids[(program_node_id, port.node)]
        boundary = innovations.new_node_id()
        if port.direction == "input":
            nodes[boundary] = NodeGene(boundary, NodeKind.INPUT, "identity")
            connections.append(ConnectionGene(boundary, anchor, 1.0, True, innovations.innovation(boundary, anchor)))
        else:
            nodes[boundary] = NodeGene(boundary, NodeKind.OUTPUT, "identity")
            connections.append(ConnectionGene(anchor, boundary, 1.0, True, innovations.innovation(anchor, boundary)))

    genome = Genome(nodes=nodes, connections=connections)
    try:
        topological_order(genome)
    except ValueError as error:
        raise GrammarError(f"expanded program is not a feedforward DAG: {error}") from error
    return genome


def _first_exemplar(production: Production, library: ModuleLibrary) -> LibraryEntry:
    for key in production.exemplars:
        try:
            return library.load(key)
        except KeyError:
            continue
    raise GrammarError(f"no exemplar for production {production.key!r} is present in the library")


def _dense_glue(source_width: int, target_width: int, rng: random.Random) -> tuple[float, ...]:
    sigma = 1.0 / math.sqrt(max(source_width, 1))
    return tuple(rng.gauss(0.0, sigma) for _ in range(source_width * target_width))


def _compile_as_composition(
    program: Program,
    grammar: Grammar,
    library: ModuleLibrary,
    tracker: InnovationTracker | None,
    rng: random.Random,
) -> CompositionGenome:
    innovations = tracker or InnovationTracker(_next_node_id=0)
    productions = {node.id: grammar.production(node.production) for node in program.nodes}
    entries = {node.id: _first_exemplar(productions[node.id], library) for node in program.nodes}
    nodes: dict[int, CompNodeGene] = {}
    module_ids: dict[int, int] = {}
    for program_node in sorted(program.nodes):
        entry = entries[program_node.id]
        node_id = innovations.new_node_id()
        module_ids[program_node.id] = node_id
        in_width = sum(int(spec["width"]) for spec in entry.io.get("inputs", []))
        out_width = int(entry.io.get("output", {}).get("width", 0))
        if in_width <= 0 or out_width <= 0:
            raise GrammarError(f"exemplar {entry.key!r} has an invalid io contract")
        nodes[node_id] = CompNodeGene(node_id, CompNodeKind.MODULE, f"library:{entry.key}", in_width, out_width, trainable=False)

    edges: list[CompEdgeGene] = []
    for edge in program.edges:
        source_id, target_id = module_ids[edge.source], module_ids[edge.target]
        source_width, target_width = nodes[source_id].out_width, nodes[target_id].in_width
        edges.append(CompEdgeGene(source_id, target_id, True, innovations.innovation(source_id, target_id), _dense_glue(source_width, target_width, rng)))

    dangling_inputs, dangling_outputs = _dangling_ports(program, grammar)
    if not dangling_inputs or not dangling_outputs:
        raise GrammarError("compiled program needs at least one dangling input and output port")
    for program_node_id, port in dangling_inputs:
        input_id = innovations.new_node_id()
        nodes[input_id] = CompNodeGene(input_id, CompNodeKind.INPUT, port.signature, 0, port.width, trainable=False)
        target_id = module_ids[program_node_id]
        edges.append(CompEdgeGene(input_id, target_id, True, innovations.innovation(input_id, target_id), _dense_glue(port.width, nodes[target_id].in_width, rng)))

    output_width = sum(port.width for _node_id, port in dangling_outputs)
    output_id = innovations.new_node_id()
    nodes[output_id] = CompNodeGene(output_id, CompNodeKind.OUTPUT, "grammar:output", output_width, 0, trainable=False)
    for program_node_id in sorted({program_node_id for program_node_id, _port in dangling_outputs}):
        source_id = module_ids[program_node_id]
        edges.append(CompEdgeGene(source_id, output_id, True, innovations.innovation(source_id, output_id), _dense_glue(nodes[source_id].out_width, output_width, rng)))
    return CompositionGenome(nodes=nodes, edges=edges)


def compile_program(
    program: Program,
    grammar: Grammar,
    *,
    library: ModuleLibrary | None = None,
    tracker: InnovationTracker | None = None,
    rng: random.Random | None = None,
) -> Genome | CompositionGenome:
    """Compile to a flat Genome when motifs are losslessly expandable, otherwise a composition.

    Composition fallback uses immutable exemplar entries and requires the source library. Glue is
    trainable and deterministically initialized by the supplied RNG.
    """
    validate_program(program, grammar)
    try:
        return _compile_as_genome(program, grammar, tracker)
    except GrammarError:
        if library is None:
            raise
    return _compile_as_composition(program, grammar, library, tracker, rng or random.Random(0))
