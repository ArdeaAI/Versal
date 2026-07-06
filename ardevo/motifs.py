"""Motif census: mining recurring substructures across library entries.

`structural_fingerprint` dedupes WHOLE payloads but keeps node ids verbatim, so it cannot see that
two entries grew the same 4-node gadget under different numbering. This module adds the missing
piece: extract a labeled dataflow graph from each payload, enumerate its small connected subgraphs
(ESU), reduce each to an exact permutation-invariant canonical form, and count how many DISTINCT
entries contain each canonical motif. A spontaneously evolved skip connection, gating unit, or
attention-like wiring shows up here as a motif recurring across independent solutions, which is the
signal the census exists to surface.

Extraction decisions (why, not what):
- Enabled connections only: disabled genes are dormant genome history, not dataflow.
- Bias nodes excluded from module graphs: a bias edge is an affine parameter of its target, and the
  bias node fans out to nearly everything, so keeping it manufactures meaningless star motifs
  around one hub. The motifs worth noticing (skip, gate, attention) are structurally bias-free.
- Macro output stubs kept and labeled: they are the visible footprint of a reused library unit, and
  a motif routing through one is a different discovery than one through a plain neuron. The
  macro-implied bipartite edges carry their own mask bit so that difference survives hashing.
- Recurrent self-loops (the `enable_refinement` TRM motif) stay in the induced subgraphs but never
  enter the undirected ESU skeleton, where a self-adjacency would corrupt the extension rule.

Identity-flood note: most evolved modules are dominated by identity/sum nodes, so raw support
ranking would drown everything under identity chains. The census GROUPS by an intrinsic
`diversity_class` instead of filtering: identity chains compete inside one section while the
recurrent/gated/macro/mixed classes (where a real discovery would land) keep guaranteed visibility.
"""

import hashlib
import json
from dataclasses import dataclass, field
from itertools import permutations, product
from typing import Any, Iterable, Mapping, Sequence

from ardevo.library import COMPOSITION, MODULE, LibraryEntry, ModuleLibrary, payload_refs

NodeLabel = tuple[str, str, str, str]  # (kind, activation_or_ref_class, aggregation, "stub" | "")

FORWARD_EDGE = 1
RECURRENT_EDGE = 2
MACRO_EDGE = 4

DEFAULT_MODULE_SIZES = (3, 4)
DEFAULT_COMPOSITION_SIZES = (2, 3, 4)
DEFAULT_PER_ENTRY_CAP = 50_000
MAX_MOTIF_SIZE = 5  # the canonicalizer is exact brute force; past 5 nodes the permutation space explodes


# --- graph extraction -------------------------------------------------------------------------------


def module_motif_graph(payload: dict[str, Any]) -> tuple[dict[int, NodeLabel], dict[tuple[int, int], int]]:
    """The labeled dataflow graph of a module payload: node id -> label, (in, out) -> edge mask."""
    stub_ids = {int(node_id) for macro in payload.get("macros", []) for node_id in macro.get("outputs", [])}
    labels: dict[int, NodeLabel] = {}
    for node in payload.get("nodes", []):
        if node["kind"] == "bias":
            continue
        node_id = int(node["id"])
        labels[node_id] = (node["kind"], node["activation"], node.get("aggregation", "sum"), "stub" if node_id in stub_ids else "")
    edges: dict[tuple[int, int], int] = {}
    for conn in payload.get("connections", []):
        if not conn["enabled"]:
            continue
        source, target = int(conn["in"]), int(conn["out"])
        if source not in labels or target not in labels:
            continue
        edges[(source, target)] = edges.get((source, target), 0) | (RECURRENT_EDGE if conn.get("recurrent", False) else FORWARD_EDGE)
    for macro in payload.get("macros", []):
        for source in macro.get("inputs", []):
            for target in macro.get("outputs", []):
                source_id, target_id = int(source), int(target)
                if source_id not in labels or target_id not in labels:
                    continue
                edges[(source_id, target_id)] = edges.get((source_id, target_id), 0) | MACRO_EDGE
    return labels, edges


def _ref_class(ref: str) -> str:
    """A composition node's structural role: the referenced entry's LEVEL for module refs (parsed
    from the key text so retired and even GC-vanished refs still classify), io roles otherwise."""
    if ref == "__bias__":
        return "bias"
    if not ref.startswith("library:"):
        return "io"
    level_text = ref.removeprefix("library:")[1:].split("_", 1)[0]
    return f"L{level_text}" if level_text.isdigit() else "L?"


def composition_motif_graph(payload: dict[str, Any]) -> tuple[dict[int, NodeLabel], dict[tuple[int, int], int]]:
    """The labeled wiring graph of a composition payload. Refs collapse to their structural class
    (level for modules, io role otherwise): the census asks HOW compositions wire levels together,
    not WHICH exact entries they picked (that is the reuse census's job)."""
    labels: dict[int, NodeLabel] = {}
    for node in payload.get("nodes", []):
        kind = node["kind"]
        if kind == "module":
            ref_class = _ref_class(node.get("ref", ""))
        elif kind == "input":
            ref_class = "bias" if node.get("ref", "") == "__bias__" else "input"
        else:
            ref_class = "output"
        labels[int(node["id"])] = (kind, ref_class, node.get("aggregation", "sum"), "")
    edges: dict[tuple[int, int], int] = {}
    for edge in payload.get("edges", []):
        if not edge["enabled"]:
            continue
        source, target = int(edge["in"]), int(edge["out"])
        if source not in labels or target not in labels:
            continue
        edges[(source, target)] = edges.get((source, target), 0) | FORWARD_EDGE
    return labels, edges


# --- canonicalization -------------------------------------------------------------------------------


@dataclass(frozen=True)
class MotifGraph:
    """A canonical labeled digraph: node index order is the canonical order, edges are relabeled
    and sorted. Two isomorphic labeled subgraphs produce the identical MotifGraph."""

    node_labels: tuple[NodeLabel, ...]
    edges: tuple[tuple[int, int, int], ...]  # (source, target, mask)


def canonical_form(node_labels: Sequence[NodeLabel], edges: Iterable[tuple[int, int, int]]) -> MotifGraph:
    """Exact canonical form by lexicographic minimization over label-preserving permutations.

    The label sequence dominates the serialization, so only permutations that realize the SORTED
    label sequence can win; nodes therefore only permute within equal-label blocks (worst case
    5! = 120 when all labels match, the practical ceiling that pins MAX_MOTIF_SIZE)."""
    size = len(node_labels)
    sorted_labels = tuple(sorted(node_labels))
    positions_by_label: dict[NodeLabel, list[int]] = {}
    for position, label in enumerate(sorted_labels):
        positions_by_label.setdefault(label, []).append(position)
    indices_by_label: dict[NodeLabel, list[int]] = {}
    for index, label in enumerate(node_labels):
        indices_by_label.setdefault(label, []).append(index)
    edge_list = list(edges)

    blocks = [(indices_by_label[label], positions_by_label[label]) for label in sorted(indices_by_label)]
    best: tuple[tuple[int, int, int], ...] | None = None
    for assignment in product(*(permutations(indices) for indices, _positions in blocks)):
        mapping = [0] * size
        for (_, block_positions), ordered_indices in zip(blocks, assignment):
            for position, index in zip(block_positions, ordered_indices):
                mapping[index] = position
        candidate = tuple(sorted((mapping[source], mapping[target], mask) for source, target, mask in edge_list))
        if best is None or candidate < best:
            best = candidate
    return MotifGraph(node_labels=sorted_labels, edges=best or ())


def motif_fingerprint(graph: MotifGraph) -> str:
    serialized = json.dumps({"nodes": [list(label) for label in graph.node_labels], "edges": [list(edge) for edge in graph.edges]}, sort_keys=True)
    return hashlib.sha1(serialized.encode()).hexdigest()[:16]


# --- enumeration (ESU / Wernicke) --------------------------------------------------------------------


def enumerate_connected_subgraphs(skeleton: Mapping[int, set[int]], k: int, cap: int) -> tuple[list[frozenset[int]], bool]:
    """Every connected k-node subset of the undirected skeleton, each emitted exactly once (the ESU
    exclusive-neighborhood rule). Roots and extensions process in sorted order, so truncation at
    `cap` is deterministic across runs. Returns (subsets, truncated)."""
    results: list[frozenset[int]] = []
    truncated = False
    order = {node: index for index, node in enumerate(sorted(skeleton))}

    def extend(subgraph: list[int], extension: list[int], root_rank: int) -> bool:
        nonlocal truncated
        if len(subgraph) == k:
            results.append(frozenset(subgraph))
            if len(results) >= cap:
                truncated = True
                return False
            return True
        members = set(subgraph)
        neighborhood = {neighbor for node in subgraph for neighbor in skeleton[node]}
        while extension:
            candidate = extension.pop(0)
            grown = extension + sorted(neighbor for neighbor in skeleton[candidate] if order[neighbor] > root_rank and neighbor not in members and neighbor not in neighborhood)
            if not extend(subgraph + [candidate], grown, root_rank):
                return False
        return True

    for root in sorted(skeleton):
        root_rank = order[root]
        initial = sorted(neighbor for neighbor in skeleton[root] if order[neighbor] > root_rank)
        if not extend([root], initial, root_rank):
            break
    return results, truncated


def _skeleton(labels: Mapping[int, NodeLabel], edges: Mapping[tuple[int, int], int]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {node_id: set() for node_id in labels}
    for source, target in edges:
        if source != target:  # self-loops stay OUT of the skeleton; they ride along in induced subgraphs
            adjacency[source].add(target)
            adjacency[target].add(source)
    return adjacency


# --- census -----------------------------------------------------------------------------------------


@dataclass
class MotifRecord:
    fingerprint: str
    size: int
    graph: MotifGraph
    support: int  # distinct entries containing the motif
    occurrences: int  # total embeddings across entries (post-cap)
    exemplars: tuple[str, ...]
    diversity_class: str
    description: str


@dataclass
class CensusReport:
    module_motifs: list[MotifRecord]
    composition_motifs: list[MotifRecord]
    vocabulary: list[dict[str, Any]]
    truncated_entries: dict[str, list[int]]  # entry key -> sizes that hit the cap
    entries_scanned: dict[str, int]
    params: dict[str, Any] = field(default_factory=dict)
    # Provenance the mining core CAN derive purely from library content (no clock, no path): the exact
    # keys scanned, a content fingerprint over them (entry keys are already payload hashes), and the
    # index size the scan drew from. The volatile stamps (wall-clock, on-disk path) live only in the
    # file's "meta" object, added at write time, so this report stays a pure function of its inputs.
    scanned_keys: dict[str, list[str]] = field(default_factory=dict)
    input_fingerprint: str = ""
    index_total: int = 0  # total index rows (retired included), so a viewer sees what the scan skipped


def diversity_class(graph: MotifGraph) -> str:
    """An intrinsic content class, so ranking can group instead of filter: identity chains all land
    in one uniform section while the classes a real discovery would occupy stay visible."""
    flags: list[str] = []
    if any(mask & RECURRENT_EDGE for _source, _target, mask in graph.edges):
        flags.append("recurrent")
    if any(label[2] == "product" for label in graph.node_labels):
        flags.append("gated")
    if any(label[3] == "stub" for label in graph.node_labels) or any(mask & MACRO_EDGE for _source, _target, mask in graph.edges):
        flags.append("macro")
    if flags:
        return "+".join(flags)
    if len(set(graph.node_labels)) >= 2:
        return "mixed"
    return f"uniform-{graph.node_labels[0][1]}" if graph.node_labels else "empty"


_KIND_ABBREV = {"input": "in", "bias": "bias", "hidden": "hid", "output": "out", "module": "mod"}


def describe_motif(graph: MotifGraph) -> str:
    node_parts: list[str] = []
    for kind, second, aggregation, stub in graph.node_labels:
        text = f"{_KIND_ABBREV.get(kind, kind)}/{second}"
        if aggregation == "product":
            text += "(prod)"
        if stub:
            text += "[stub]"
        node_parts.append(text)
    edge_parts: list[str] = []
    for source, target, mask in graph.edges:
        suffix = ("r" if mask & RECURRENT_EDGE else "") + ("m" if mask & MACRO_EDGE else "")
        edge_parts.append(f"{source}->{target}{suffix}")
    return "; ".join((", ".join(node_parts), ", ".join(edge_parts)))


def _load_entries(library: ModuleLibrary, *, include_retired: bool) -> list[LibraryEntry]:
    """One disk pass, shared by the motif and reuse censuses (`load` re-reads JSON on every call)."""
    entries: list[LibraryEntry] = []
    for row in library.summaries(include_retired=include_retired, include_dependencies=True):
        try:
            entries.append(library.load(row["key"]))
        except KeyError:
            continue  # an index row whose file vanished mid-sweep is not the census's problem
    return entries


def _mine_entry(
    labels: Mapping[int, NodeLabel],
    edges: Mapping[tuple[int, int], int],
    sizes: Iterable[int],
    per_entry_cap: int,
    canonical_cache: dict[tuple[tuple[NodeLabel, ...], tuple[tuple[int, int, int], ...]], MotifGraph],
) -> tuple[dict[str, tuple[MotifGraph, int]], list[int]]:
    """All canonical motifs in one entry graph: fingerprint -> (graph, occurrence count), plus the
    sizes whose enumeration hit the cap."""
    skeleton = _skeleton(labels, edges)
    edges_from: dict[int, list[tuple[int, int]]] = {}
    for (source, target), mask in edges.items():
        edges_from.setdefault(source, []).append((target, mask))
    found: dict[str, tuple[MotifGraph, int]] = {}
    truncated_sizes: list[int] = []
    for k in sizes:
        if k < 2 or k > MAX_MOTIF_SIZE or k > len(labels):
            continue
        subsets, truncated = enumerate_connected_subgraphs(skeleton, k, per_entry_cap)
        if truncated:
            truncated_sizes.append(k)
        for subset in subsets:
            ordered = sorted(subset)
            index_of = {node_id: index for index, node_id in enumerate(ordered)}
            local_labels = tuple(labels[node_id] for node_id in ordered)
            local_edges = tuple(sorted((index_of[source], index_of[target], mask) for source in ordered for target, mask in edges_from.get(source, []) if target in subset))
            cache_key = (local_labels, local_edges)
            graph = canonical_cache.get(cache_key)
            if graph is None:
                graph = canonical_form(local_labels, local_edges)
                canonical_cache[cache_key] = graph
            fingerprint = motif_fingerprint(graph)
            existing = found.get(fingerprint)
            found[fingerprint] = (graph, existing[1] + 1 if existing else 1)
    return found, truncated_sizes


def _assemble_records(per_entry_motifs: dict[str, dict[str, tuple[MotifGraph, int]]], min_support: int) -> list[MotifRecord]:
    graphs: dict[str, MotifGraph] = {}
    supporters: dict[str, list[str]] = {}
    occurrences: dict[str, int] = {}
    for entry_key in sorted(per_entry_motifs):
        for fingerprint, (graph, count) in per_entry_motifs[entry_key].items():
            graphs[fingerprint] = graph
            supporters.setdefault(fingerprint, []).append(entry_key)
            occurrences[fingerprint] = occurrences.get(fingerprint, 0) + count
    records = [
        MotifRecord(
            fingerprint=fingerprint,
            size=len(graph.node_labels),
            graph=graph,
            support=len(supporters[fingerprint]),
            occurrences=occurrences[fingerprint],
            exemplars=tuple(supporters[fingerprint][:3]),
            diversity_class=diversity_class(graph),
            description=describe_motif(graph),
        )
        for fingerprint, graph in graphs.items()
        if len(supporters[fingerprint]) >= min_support
    ]
    records.sort(key=lambda record: (record.diversity_class, -record.support, -record.occurrences, -record.size, record.fingerprint))
    return records


def reuse_census(library: ModuleLibrary, *, include_retired: bool = False, entries: list[LibraryEntry] | None = None) -> list[dict[str, Any]]:
    """The vocabulary half of the census: which entries other entries are built FROM (reverse
    reference index over payload refs) plus the run-time use counters from the index."""
    if entries is None:
        entries = _load_entries(library, include_retired=include_retired)
    referenced_by: dict[str, list[str]] = {}
    for entry in entries:
        for ref in sorted(payload_refs(entry.entry_type, entry.payload)):
            referenced_by.setdefault(ref, []).append(entry.key)
    rows: list[dict[str, Any]] = []
    for entry in entries:
        summary = library.summary(entry.key) or {}
        stats = summary.get("stats", {}) or {}
        rows.append(
            {
                "key": entry.key,
                "entry_type": entry.entry_type,
                "level": entry.level,
                "referenced_by": referenced_by.get(entry.key, []),
                "reference_count": len(referenced_by.get(entry.key, [])),
                "use_count": int(stats.get("use_count", 0)),
                "max_attributed_fitness": float(stats.get("max_attributed_fitness", 0.0)),
                "retired": bool(summary.get("retired", False)),
                "dependency": bool(summary.get("dependency", False)),
            }
        )
    rows.sort(key=lambda row: (-row["reference_count"], -row["use_count"], row["key"]))
    return rows


def motif_census(
    library: ModuleLibrary,
    *,
    sizes: tuple[int, ...] = DEFAULT_MODULE_SIZES,
    min_support: int = 2,
    include_retired: bool = False,
    per_entry_cap: int = DEFAULT_PER_ENTRY_CAP,
) -> CensusReport:
    """Mine every entry once and count each canonical motif's support across DISTINCT entries.
    `sizes` governs module graphs; composition graphs are tiny (their nodes are opaque refs) and
    always mine the small fixed range, where a 2-node level-to-level link is already informative.

    A support=1 row (only reachable with `min_support=1`) is intra-entry structure: a motif appearing
    inside a single entry, not a cross-entry recurrence. That is the honest small-library reading."""
    entries = _load_entries(library, include_retired=include_retired)
    canonical_cache: dict[tuple[tuple[NodeLabel, ...], tuple[tuple[int, int, int], ...]], MotifGraph] = {}
    module_found: dict[str, dict[str, tuple[MotifGraph, int]]] = {}
    comp_found: dict[str, dict[str, tuple[MotifGraph, int]]] = {}
    truncated_entries: dict[str, list[int]] = {}
    scanned = {"modules": 0, "compositions": 0}
    scanned_keys: dict[str, list[str]] = {"modules": [], "compositions": []}
    for entry in entries:
        if entry.entry_type == MODULE:
            labels, edges = module_motif_graph(entry.payload)
            found, truncated_sizes = _mine_entry(labels, edges, sizes, per_entry_cap, canonical_cache)
            module_found[entry.key] = found
            scanned["modules"] += 1
            scanned_keys["modules"].append(entry.key)
        elif entry.entry_type == COMPOSITION:
            labels, edges = composition_motif_graph(entry.payload)
            found, truncated_sizes = _mine_entry(labels, edges, DEFAULT_COMPOSITION_SIZES, per_entry_cap, canonical_cache)
            comp_found[entry.key] = found
            scanned["compositions"] += 1
            scanned_keys["compositions"].append(entry.key)
        else:
            continue
        if truncated_sizes:
            truncated_entries[entry.key] = truncated_sizes
    scanned_keys = {kind: sorted(keys) for kind, keys in scanned_keys.items()}
    all_scanned_keys = sorted(scanned_keys["modules"] + scanned_keys["compositions"])
    input_fingerprint = hashlib.sha1("\n".join(all_scanned_keys).encode()).hexdigest()[:16]
    return CensusReport(
        module_motifs=_assemble_records(module_found, min_support),
        composition_motifs=_assemble_records(comp_found, min_support),
        vocabulary=reuse_census(library, include_retired=include_retired, entries=entries),
        truncated_entries=truncated_entries,
        entries_scanned=scanned,
        params={"sizes": list(sizes), "min_support": min_support, "per_entry_cap": per_entry_cap, "include_retired": include_retired},
        scanned_keys=scanned_keys,
        input_fingerprint=input_fingerprint,
        index_total=len(library),
    )


def empty_state_explanation(kind: str, scanned: int, min_support: int, records: list[MotifRecord]) -> str | None:
    """Why a motif table is empty, in one sentence, so the reader never mistakes "small library" for
    "no structure". Returns None when the table has rows. Two distinct causes get distinct messages:
    too few entries to ever satisfy min_support (the small-library case, with the intra-entry escape
    hatch spelled out), versus enough entries but no shared recurrence."""
    if records:
        return None
    if scanned < min_support:
        return (
            f"{scanned} {kind} entries scanned; min_support={min_support} requires motifs recurring across >= {min_support} distinct entries, "
            f"so this table is empty by construction. Rerun with --min-support 1 for intra-entry structure, or --library <archive dir> for a multi-entry library."
        )
    return f"no motif recurred across >= {min_support} of the {scanned} scanned entries."


def report_to_dict(report: CensusReport) -> dict[str, Any]:
    """The JSON-report shape: canonical nodes/edges ride along verbatim so a motif can be
    reconstructed (and re-rendered) from the file alone. This stays a PURE function of the report;
    volatile stamps (wall-clock, path) are added by the writer as a separate "meta" object."""

    def record_rows(records: list[MotifRecord]) -> list[dict[str, Any]]:
        return [
            {
                "fingerprint": record.fingerprint,
                "size": record.size,
                "diversity_class": record.diversity_class,
                "support": record.support,
                "occurrences": record.occurrences,
                "exemplars": list(record.exemplars),
                "nodes": [list(label) for label in record.graph.node_labels],
                "edges": [list(edge) for edge in record.graph.edges],
                "description": record.description,
            }
            for record in records
        ]

    min_support = int(report.params.get("min_support", 0))
    return {
        "params": report.params,
        "entries_scanned": report.entries_scanned,
        "scanned_keys": report.scanned_keys,
        "input_fingerprint": report.input_fingerprint,
        "index_total": report.index_total,
        "truncated_entries": report.truncated_entries,
        "module_motifs": record_rows(report.module_motifs),
        "composition_motifs": record_rows(report.composition_motifs),
        "vocabulary": report.vocabulary,
        "explanations": {
            "module_motifs": empty_state_explanation("module", report.entries_scanned.get("modules", 0), min_support, report.module_motifs),
            "composition_motifs": empty_state_explanation("composition", report.entries_scanned.get("compositions", 0), min_support, report.composition_motifs),
        },
    }
