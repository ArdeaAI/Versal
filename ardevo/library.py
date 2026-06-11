"""Module library: the persistent, searchable space of evolved solutions.

This is the memory of the whole system. Mini-model genomes (level 1) and compositions of library
entries (level 2+) are admitted with their I/O signature, weights, and provenance; the orchestrator
queries it BEFORE evolving anything, and evolution operators graft entries back into live genomes.
Entries are immutable once admitted (dedupe is by canonical payload hash), which is what makes a
solved task stay solved: re-encountering it is a lookup, not a retrain.

Layout (file-based, append-only):
    <root>/index.json            summaries for querying without loading payloads
    <root>/entries/<key>.json    full records

Signatures derive ONLY from FieldDescriptor facts (value_type + axes + widths), never from rung or
task name: a future ARC adapter must be able to hit the same entries from tasks that never touched
the Icarus dataset.
"""

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ardevo.dataset.icarus import Level0Encoder, Task, encode_task, support_loader
from ardevo.evaluation import output_features
from ardevo.evolution.genome import ConnectionGene, Genome, InnovationTracker, genome_from_dict
from ardevo.evolution.registry import Registry

MODULE = "module"
COMPOSITION = "composition"

LIBRARY_ADMISSION: Registry = Registry("library_admission")


def descriptor_signature(value_type: str, axes: tuple[str, ...]) -> str:
    """The bank-signature scheme shared with `multitask.task_entry`: value_type plus axis letters."""
    return f"{value_type}|{','.join(axes)}"


def task_io(task: Task) -> dict[str, Any]:
    """The I/O contract of a task, derived purely from Field descriptors and widths.

    Deliberately ignores `task.meta` (rung, name, kind): library matching must work for any future
    task source with the same structural shape.
    """
    support_input, support_output = support_loader(task)
    input_width = 1
    for dim in support_input.data.shape[1:]:
        input_width *= int(dim)
    encoded = encode_task(task, Level0Encoder(input_width))
    return {
        "inputs": [
            {
                "signature": descriptor_signature(support_input.descriptor.value_type.value, tuple(axis.value for axis in support_input.descriptor.axes)),
                "width": input_width,
            }
        ],
        "output": {
            "signature": descriptor_signature(support_output.descriptor.value_type.value, tuple(axis.value for axis in support_output.descriptor.axes)),
            "width": output_features(encoded),
        },
    }


@dataclass
class LibraryEntry:
    """One admitted solution. `payload` is a genome dict (module) or a composition dict."""

    key: str
    entry_type: str
    level: int
    io: dict[str, Any]
    payload: dict[str, Any]
    weights_frozen: bool
    provenance: dict[str, Any]
    stats: dict[str, Any] = field(default_factory=lambda: {"use_count": 0, "max_attributed_fitness": 0.0})

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "entry_type": self.entry_type,
            "level": self.level,
            "io": self.io,
            "payload": self.payload,
            "weights_frozen": self.weights_frozen,
            "provenance": self.provenance,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LibraryEntry":
        return cls(
            key=data["key"],
            entry_type=data["entry_type"],
            level=int(data["level"]),
            io=data["io"],
            payload=data["payload"],
            weights_frozen=bool(data["weights_frozen"]),
            provenance=data["provenance"],
            stats=data.get("stats", {"use_count": 0, "max_attributed_fitness": 0.0}),
        )


def _canonical_key(entry_type: str, level: int, payload: dict[str, Any]) -> str:
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return f"{entry_type[0]}{level}_{digest}"


class ModuleLibrary:
    """File-backed store. The index holds query-able summaries; payloads load on demand."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._entries_dir = self.root / "entries"
        self._index_path = self.root / "index.json"
        self._index: dict[str, dict[str, Any]] = {}
        if self._index_path.exists():
            self._index = {item["key"]: item for item in json.loads(self._index_path.read_text())}

    def __len__(self) -> int:
        return len(self._index)

    def keys(self) -> list[str]:
        return sorted(self._index)

    def _write_index(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text(json.dumps(sorted(self._index.values(), key=lambda item: item["key"]), indent=2))

    def add(
        self,
        *,
        entry_type: str,
        payload: dict[str, Any],
        io: dict[str, Any],
        provenance: dict[str, Any],
        level: int = 1,
        weights_frozen: bool = True,
    ) -> str:
        """Admit a solution; identical payloads dedupe to the existing key (stats are kept)."""
        if entry_type not in (MODULE, COMPOSITION):
            raise ValueError(f"unknown entry_type {entry_type!r}")
        key = _canonical_key(entry_type, level, payload)
        if key in self._index:
            return key
        entry = LibraryEntry(key=key, entry_type=entry_type, level=level, io=io, payload=payload, weights_frozen=weights_frozen, provenance=provenance)
        self._entries_dir.mkdir(parents=True, exist_ok=True)
        (self._entries_dir / f"{key}.json").write_text(json.dumps(entry.to_dict(), indent=2))
        self._index[key] = {
            "key": key,
            "entry_type": entry_type,
            "level": level,
            "io": io,
            "accepted_metric": float(provenance.get("accepted_metric", 0.0)),
            "weight_robustness": float(provenance.get("weight_robustness", 0.0)),
            "stats": entry.stats,
        }
        self._write_index()
        return key

    def load(self, key: str) -> LibraryEntry:
        path = self._entries_dir / f"{key}.json"
        if not path.exists():
            raise KeyError(f"no library entry {key!r} under {self.root}")
        return LibraryEntry.from_dict(json.loads(path.read_text()))

    def query(
        self,
        *,
        entry_type: str | None = None,
        input_signature: str | None = None,
        input_width: int | None = None,
        output_signature: str | None = None,
        output_width: int | None = None,
        min_metric: float | None = None,
        limit: int = 0,
    ) -> list[LibraryEntry]:
        """Entries matching the structural filters, best first by (robustness, accepted metric)."""
        matches: list[dict[str, Any]] = []
        for summary in self._index.values():
            if entry_type is not None and summary["entry_type"] != entry_type:
                continue
            inputs = summary["io"]["inputs"]
            output = summary["io"]["output"]
            if input_signature is not None and not any(item["signature"] == input_signature for item in inputs):
                continue
            if input_width is not None and not any(item["width"] == input_width for item in inputs):
                continue
            if output_signature is not None and output["signature"] != output_signature:
                continue
            if output_width is not None and output["width"] != output_width:
                continue
            if min_metric is not None and summary["accepted_metric"] < min_metric:
                continue
            matches.append(summary)
        matches.sort(key=lambda item: (item["weight_robustness"], item["accepted_metric"]), reverse=True)
        if limit:
            matches = matches[:limit]
        return [self.load(summary["key"]) for summary in matches]

    def bump_stats(self, key: str, attributed_fitness: float) -> None:
        """Record a use of `key` in an assembled network (provenance for future ranking)."""
        summary = self._index.get(key)
        if summary is None:
            return
        stats = summary.setdefault("stats", {"use_count": 0, "max_attributed_fitness": 0.0})
        stats["use_count"] = int(stats.get("use_count", 0)) + 1
        stats["max_attributed_fitness"] = max(float(stats.get("max_attributed_fitness", 0.0)), attributed_fitness)
        entry = self.load(key)
        entry.stats = stats
        (self._entries_dir / f"{key}.json").write_text(json.dumps(entry.to_dict(), indent=2))
        self._write_index()


def graft(entry: LibraryEntry, tracker: InnovationTracker) -> Genome:
    """Rebuild a module entry as a fresh `Genome`: new node ids and innovations allocated through
    the RUN'S tracker (so genes align within the receiving population), weights / activations /
    aggregations / recurrence preserved."""
    if entry.entry_type != MODULE:
        raise ValueError(f"can only graft module entries, got {entry.entry_type!r}")
    source = genome_from_dict(entry.payload)
    id_map = {old_id: tracker.new_node_id() for old_id in sorted(source.nodes)}
    nodes = {id_map[node.id]: replace(node, id=id_map[node.id]) for node in source.nodes.values()}
    connections = [
        ConnectionGene(
            id_map[conn.in_id],
            id_map[conn.out_id],
            conn.weight,
            conn.enabled,
            tracker.innovation(id_map[conn.in_id], id_map[conn.out_id], conn.recurrent),
            conn.recurrent,
        )
        for conn in source.connections
    ]
    return Genome(nodes=nodes, connections=connections)
