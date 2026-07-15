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
from typing import Any, Callable, Iterable

from ardevo.dataset.icarus import Level0Encoder, Task, encode_task, support_loader
from ardevo.evaluation import fit_query_target, output_features
from ardevo.evolution.genome import ConnectionGene, Genome, InnovationTracker, MacroGene, genome_from_dict
from ardevo.evolution.registry import Registry
from ardevo.utils.logging import Logger

logger = Logger.get_logger()

MODULE = "module"
COMPOSITION = "composition"
INVALID_EXPANDED_COMPLEXITY = 10**12

AdmissionPolicy = Callable[..., "AdmissionDecision"]

LIBRARY_ADMISSION: Registry = Registry("library_admission")


@dataclass(frozen=True)
class AdmissionDecision:
    admit: bool
    retire: tuple[str, ...] = ()  # tombstone these keys when the candidate replaces them
    reason: str = ""


@LIBRARY_ADMISSION.register("accept_all")
def _build_accept_all(**_params: object) -> "AdmissionPolicy":
    """The legacy behavior, now an explicit policy: everything that cleared the orchestrator's
    accept threshold is admitted."""

    def policy(library: "ModuleLibrary", *, entry_type: str, io: dict[str, Any], provenance: dict[str, Any]) -> AdmissionDecision:
        return AdmissionDecision(admit=True)

    return policy


@LIBRARY_ADMISSION.register("default")
def _build_default(*, min_metric: float = 0.0, min_robustness: float = 0.0, per_signature_cap: int = 3, **_params: object) -> "AdmissionPolicy":
    """Quality gate + redundancy control: reject below the metric/robustness floors; cap entries
    per exact io shape, replacing (tombstoning) the weakest same-shape entry only when the
    candidate outranks it by (robustness, metric). Replacement never deletes: retired entries stay
    loadable forever so existing composition refs keep assembling."""

    def policy(library: "ModuleLibrary", *, entry_type: str, io: dict[str, Any], provenance: dict[str, Any]) -> AdmissionDecision:
        metric = float(provenance.get("accepted_metric", 0.0))
        robustness = float(provenance.get("weight_robustness", 0.0))
        if metric < min_metric:
            return AdmissionDecision(admit=False, reason=f"metric {metric:.3f} below min_metric {min_metric}")
        if robustness < min_robustness:
            return AdmissionDecision(admit=False, reason=f"robustness {robustness:.3f} below min_robustness {min_robustness}")
        group = library.signature_group(entry_type, io)
        if len(group) < per_signature_cap:
            return AdmissionDecision(admit=True)
        worst = min(group, key=lambda summary: (summary["weight_robustness"], summary["accepted_metric"]))
        if (robustness, metric) > (float(worst["weight_robustness"]), float(worst["accepted_metric"])):
            return AdmissionDecision(admit=True, retire=(worst["key"],), reason=f"replaces weaker same-shape entry {worst['key']}")
        return AdmissionDecision(admit=False, reason=f"per-signature cap {per_signature_cap} reached by stronger entries")

    return policy


@LIBRARY_ADMISSION.register("archive")
def _build_archive(*, min_metric: float = 0.0, min_robustness: float = 0.0, per_niche_cap: int = 2, max_per_signature: int = 12, **_params: object) -> "AdmissionPolicy":
    """Open-ended stepping-stone archive (the DGM idea): keep BEHAVIORALLY DIVERSE solutions instead
    of only the top few by metric. Entries are niched by (io shape, behavior descriptor), where the
    behavior descriptor is a coarse structural fingerprint the orchestrator stamps in provenance
    (complexity bucket, recurrence, refinement, product gating, macros). Each niche holds up to
    `per_niche_cap`; many niches coexist up to `max_per_signature` total. This preserves the
    diversity that recombination (compositions, macros, pool grafting) feeds on, which the `default`
    policy's flat top-k cap destroys. Tombstoning (never deletion) keeps existing refs assembling."""

    def rank(summary: dict[str, Any]) -> tuple[float, float]:
        return (float(summary["weight_robustness"]), float(summary["accepted_metric"]))

    def policy(library: "ModuleLibrary", *, entry_type: str, io: dict[str, Any], provenance: dict[str, Any]) -> AdmissionDecision:
        metric = float(provenance.get("accepted_metric", 0.0))
        robustness = float(provenance.get("weight_robustness", 0.0))
        if metric < min_metric:
            return AdmissionDecision(admit=False, reason=f"metric {metric:.3f} below min_metric {min_metric}")
        if robustness < min_robustness:
            return AdmissionDecision(admit=False, reason=f"robustness {robustness:.3f} below min_robustness {min_robustness}")
        candidate = (robustness, metric)
        behavior = list(provenance.get("behavior", []))
        group = library.signature_group(entry_type, io)
        niche = [summary for summary in group if list(summary.get("behavior", [])) == behavior]
        retire: tuple[str, ...] = ()
        if len(niche) >= per_niche_cap:
            worst = min(niche, key=rank)
            if candidate <= rank(worst):
                return AdmissionDecision(admit=False, reason=f"niche cap {per_niche_cap} reached by stronger entries")
            retire = (worst["key"],)
        remaining = [summary for summary in group if summary["key"] not in retire]
        if len(remaining) >= max_per_signature:
            globally_weakest = min(remaining, key=rank)
            if candidate <= rank(globally_weakest):
                return AdmissionDecision(admit=False, reason=f"signature archive full ({max_per_signature}) of stronger entries")
            retire = retire + (globally_weakest["key"],)
        return AdmissionDecision(admit=True, retire=retire, reason="diverse stepping stone" if not retire else f"replaces {retire}")

    return policy


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
    encoded = fit_query_target(encode_task(task, Level0Encoder(input_width)))
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


def payload_refs(entry_type: str, payload: dict[str, Any]) -> set[str]:
    """Every library key this payload names: composition node refs and module macro refs (both use
    the literal "library:" prefix). The GC's edge function, and the natural motif-census hook."""
    refs: set[str] = set()
    if entry_type == COMPOSITION:
        candidates = (node.get("ref", "") for node in payload.get("nodes", []))
    else:
        candidates = (macro.get("ref", "") for macro in payload.get("macros", []))
    for ref in candidates:
        if ref.startswith("library:"):
            refs.add(ref.removeprefix("library:"))
    return refs


def payload_shell_complexity(entry_type: str, payload: dict[str, Any]) -> int:
    """The historical local-only structural cost, computed without constructing a genome."""

    if entry_type == MODULE:
        enabled = sum(bool(connection.get("enabled", True)) for connection in payload.get("connections", []))
        hidden = sum(node.get("kind") == "hidden" for node in payload.get("nodes", []))
        return enabled + hidden + len(payload.get("macros", []))
    if entry_type == COMPOSITION:
        enabled = sum(bool(edge.get("enabled", True)) for edge in payload.get("edges", []))
        modules = sum(node.get("kind") == "module" for node in payload.get("nodes", []))
        return enabled + modules
    raise ValueError(f"unknown entry_type {entry_type!r}")


def expanded_payload_complexity(
    entry_type: str,
    payload: dict[str, Any],
    library: "ModuleLibrary",
    *,
    visiting: frozenset[str] = frozenset(),
    depth: int = 0,
) -> int:
    """Static assembled-topology cost, including every persistent reference placement.

    Repeated placements count repeatedly because each executes separately even when weights are
    shared. Missing, cyclic, or implausibly deep references receive an effectively infinite cost;
    malformed legacy state can therefore never masquerade as compression.
    """

    if depth > 64:
        return INVALID_EXPANDED_COMPLEXITY
    if entry_type == MODULE:
        references = [macro.get("ref", "") for macro in payload.get("macros", [])]
    elif entry_type == COMPOSITION:
        references = [node.get("ref", "") for node in payload.get("nodes", []) if node.get("kind") == "module"]
    else:
        raise ValueError(f"unknown entry_type {entry_type!r}")
    total = payload_shell_complexity(entry_type, payload)
    for reference in references:
        if not reference.startswith("library:"):
            continue
        key = reference.removeprefix("library:")
        if key in visiting:
            return INVALID_EXPANDED_COMPLEXITY
        try:
            entry = library.load(key)
        except KeyError:
            return INVALID_EXPANDED_COMPLEXITY
        nested = expanded_payload_complexity(entry.entry_type, entry.payload, library, visiting=visiting | {key}, depth=depth + 1)
        if nested >= INVALID_EXPANDED_COMPLEXITY:
            return INVALID_EXPANDED_COMPLEXITY
        total += nested
        if total >= INVALID_EXPANDED_COMPLEXITY:
            return INVALID_EXPANDED_COMPLEXITY
    return total


def structural_fingerprint(entry_type: str, payload: dict[str, Any]) -> str:
    """Weight-agnostic topology hash: two payloads share a fingerprint iff they are the same
    STRUCTURE (nodes, wiring, macro refs, refine depth), regardless of trained weights, glue
    values, or innovation numbering. Entry keys hash the full payload, so a retrained clone
    always gets a fresh key; this is the identity refinement must compare against instead
    (a weight-only "improvement" is not a new solution). Also the natural basis for a future
    motif census over discovered substructures."""
    if entry_type == MODULE:
        skeleton: dict[str, Any] = {
            "nodes": sorted(
                (int(node["id"]), node["kind"], node["activation"], list(node["coordinate"]) if node.get("coordinate") is not None else None, node.get("aggregation", "sum"))
                for node in payload["nodes"]
            ),
            "connections": sorted((int(conn["in"]), int(conn["out"]), bool(conn["enabled"]), bool(conn.get("recurrent", False))) for conn in payload["connections"]),
            "macros": sorted((macro["ref"], list(macro["inputs"]), list(macro["outputs"]), bool(macro.get("trainable", False))) for macro in payload.get("macros", [])),
            "refine_steps": int(payload.get("refine_steps", 1)),
        }
    else:
        skeleton = {
            "nodes": sorted((int(node["id"]), node["kind"], node["ref"], node.get("aggregation", "sum"), bool(node.get("trainable", True))) for node in payload["nodes"]),
            "edges": sorted(
                (
                    int(edge["in"]),
                    int(edge["out"]),
                    bool(edge["enabled"]),
                    int(edge.get("glue_rank", 0)),
                    tuple((int(run["source_start"]), int(run["target_start"]), int(run["length"])) for run in edge.get("port_map", [])),
                )
                for edge in payload["edges"]
            ),
        }
    return hashlib.sha1(json.dumps(skeleton, sort_keys=True).encode()).hexdigest()[:16]


class ModuleLibrary:
    """File-backed store. The index holds query-able summaries; payloads load on demand."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._entries_dir = self.root / "entries"
        self._index_path = self.root / "index.json"
        self._index: dict[str, dict[str, Any]] = {}
        self._macro_depth_cache: dict[str, int] = {}  # entries are immutable, so depths never change
        self._expanded_complexity_cache: dict[str, int] = {}
        # Parsed-entry cache: payloads are immutable and every mutation (dedupe provenance, stats)
        # goes through this object, so the cached instance IS the coherent one. Unbounded is fine at
        # hundreds of entries; revisit alongside the _write_index watchpoint when it reaches thousands.
        self._entry_cache: dict[str, LibraryEntry] = {}
        # Keys whose stats mutated in memory but not yet on disk (bump_stats defers; flush_stats writes).
        self._dirty_stats: set[str] = set()
        if self._index_path.exists():
            self._index = {item["key"]: item for item in json.loads(self._index_path.read_text())}

    def __len__(self) -> int:
        return len(self._index)

    def keys(self) -> list[str]:
        return sorted(self._index)

    def summaries(self, *, include_retired: bool = False, include_dependencies: bool = True) -> list[dict[str, Any]]:
        """Index-row copies for browsing/rendering, sorted by (level, key). `.get` defaults keep
        pre-gate v1 indexes (no retired/dependency flags) loadable."""
        rows = [
            dict(summary)
            for summary in self._index.values()
            if (include_retired or not summary.get("retired", False)) and (include_dependencies or not summary.get("dependency", False))
        ]
        rows.sort(key=lambda item: (item["level"], item["key"]))
        return rows

    def _write_index(self) -> None:
        # Scale watchpoint: a full index rewrite per admission is O(entries); fine at hundreds,
        # revisit (append-log or sqlite) when the library reaches thousands.
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
        """Admit a solution; identical payloads dedupe to the existing key, REFRESHING its ranking
        metadata (an entry is as good as its best verified admission) and recording the readmission."""
        if entry_type not in (MODULE, COMPOSITION):
            raise ValueError(f"unknown entry_type {entry_type!r}")
        serialized = json.dumps(payload, sort_keys=True)
        if len(serialized) > 2_000_000:
            logger.warning(
                "library entry payload is %.1f MB; wide dense glue is the usual culprit (set [evolution.composition] glue_rank_threshold to factorize new edges)",
                len(serialized) / 1e6,
            )
        key = f"{entry_type[0]}{level}_{hashlib.sha1(serialized.encode()).hexdigest()[:12]}"
        if key in self._index:
            self._refresh_on_dedupe(key, provenance)
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
            "retired": False,
            "dependency": bool(provenance.get("dependency", False)),
            "behavior": list(provenance.get("behavior", [])),  # QD niche descriptor (archive policy)
            "stats": entry.stats,
        }
        self._write_index()
        return key

    def _refresh_on_dedupe(self, key: str, provenance: dict[str, Any]) -> None:
        """B7: a re-admission is fresh evidence; ranking fields take the max, the original
        provenance is kept, and a bounded readmission history records the new context."""
        summary = self._index[key]
        summary["accepted_metric"] = max(float(summary.get("accepted_metric", 0.0)), float(provenance.get("accepted_metric", 0.0)))
        summary["weight_robustness"] = max(float(summary.get("weight_robustness", 0.0)), float(provenance.get("weight_robustness", 0.0)))
        entry = self.load(key)
        history = entry.provenance.setdefault("readmissions", [])
        history.append({k: provenance.get(k) for k in ("task", "rung", "depth", "accepted_metric", "weight_robustness")})
        del history[:-10]  # cap file growth
        (self._entries_dir / f"{key}.json").write_text(json.dumps(entry.to_dict(), indent=2))
        self._write_index()

    def retire(self, key: str) -> None:
        """Tombstone an entry: hidden from query/catalog/lookup, but `load()` keeps working forever
        so existing composition refs never dangle. Entries are NEVER deleted."""
        summary = self._index.get(key)
        if summary is None:
            return
        summary["retired"] = True
        self._write_index()

    def is_retired(self, key: str) -> bool:
        summary = self._index.get(key)
        return bool(summary is not None and summary.get("retired", False))

    def signature_group(self, entry_type: str, io: dict[str, Any]) -> list[dict[str, Any]]:
        """Live, non-dependency summaries sharing this exact io shape (the admission cap's unit).

        Dependency entries (module snapshots a composition needs) are excluded: they exist to keep
        refs assembling, not to compete for shelf space."""

        def group_key(candidate_io: dict[str, Any]) -> tuple:
            inputs = tuple((item["signature"], item["width"]) for item in candidate_io["inputs"])
            return (inputs, (candidate_io["output"]["signature"], candidate_io["output"]["width"]))

        wanted = group_key(io)
        return [
            summary
            for summary in self._index.values()
            if summary["entry_type"] == entry_type and not summary.get("retired", False) and not summary.get("dependency", False) and group_key(summary["io"]) == wanted
        ]

    def load(self, key: str) -> LibraryEntry:
        cached = self._entry_cache.get(key)
        if cached is not None:
            return cached
        path = self._entries_dir / f"{key}.json"
        if not path.exists():
            raise KeyError(f"no library entry {key!r} under {self.root}")
        entry = LibraryEntry.from_dict(json.loads(path.read_text()))
        summary = self._index.get(key)
        if summary is not None:
            # Share the index row's stats dict so deferred bump_stats mutations stay visible through
            # every loaded handle without a per-bump disk write.
            entry.stats = summary.setdefault("stats", entry.stats)
        self._entry_cache[key] = entry
        return entry

    def reference_subtree_depth(self, key: str, _visiting: frozenset[str] = frozenset()) -> int:
        """Deepest persistent-reference path below ``key``.

        A leaf has depth 0; otherwise the result is one plus the deepest referenced entry. The
        root entry itself is not counted, matching ``max_inline_depth`` decode/assembly semantics.
        Both composition module refs and genome macro refs participate through ``payload_refs``.
        Missing and cyclic references return a deliberately unbounded sentinel so construction
        filters never select them.
        """
        cached = self._macro_depth_cache.get(key)
        if cached is not None:
            return cached
        if key in _visiting:
            return 999  # defensive: immutable append-only entries cannot cycle, but never recurse forever
        try:
            entry = self.load(key)
        except KeyError:
            return 999
        refs = payload_refs(entry.entry_type, entry.payload)
        depth = 0 if not refs else 1 + max(self.reference_subtree_depth(ref, _visiting | {key}) for ref in refs)
        self._macro_depth_cache[key] = depth
        return depth

    def expanded_complexity(self, key: str) -> int:
        """Recursively expanded static topology cost for one immutable library entry."""

        cached = self._expanded_complexity_cache.get(key)
        if cached is not None:
            return cached
        entry = self.load(key)
        value = expanded_payload_complexity(entry.entry_type, entry.payload, self, visiting=frozenset({key}))
        self._expanded_complexity_cache[key] = value
        return value

    def macro_subtree_depth(self, key: str, _visiting: frozenset[str] = frozenset()) -> int:
        """Compatibility name for the now type-agnostic reference-depth query."""
        return self.reference_subtree_depth(key, _visiting)

    def collect_garbage(self, *, protect: Iterable[str] = (), dry_run: bool = False) -> list[str]:
        """Physically delete retired entries nothing retained still references. Mark-and-sweep:
        roots are every LIVE entry plus `protect` (router vertices, resumable checkpoint macro
        refs); marking follows `payload_refs` through RETAINED entries to fixpoint, so a retired
        dependency named by a live composition survives, and a whole retired chain falls together.
        Sweeping removes the entry file, the index row, and the entry's render image. This is the
        one exception to "entries are never deleted": the tombstone contract (refs never dangle)
        is preserved because only provably-unreferenced tombstones go."""
        marked = {key for key, summary in self._index.items() if not summary.get("retired", False)}
        marked.update(key for key in protect if key in self._index)
        frontier = list(marked)
        while frontier:
            key = frontier.pop()
            try:
                entry = self.load(key)
            except KeyError:
                continue
            for ref in payload_refs(entry.entry_type, entry.payload):
                if ref in self._index and ref not in marked:
                    marked.add(ref)
                    frontier.append(ref)
        swept = sorted(key for key in self._index if key not in marked)
        if dry_run or not swept:
            return swept
        for key in swept:
            del self._index[key]
            self._macro_depth_cache.pop(key, None)
            self._expanded_complexity_cache.pop(key, None)
            self._entry_cache.pop(key, None)
            self._dirty_stats.discard(key)
            (self._entries_dir / f"{key}.json").unlink(missing_ok=True)
            (self.root / "images" / f"{key}.png").unlink(missing_ok=True)
        self._write_index()
        logger.info("garbage-collected %d unreferenced tombstones", len(swept))
        return swept

    def query(
        self,
        *,
        entry_type: str | None = None,
        input_signature: str | None = None,
        input_width: int | None = None,
        output_signature: str | None = None,
        output_width: int | None = None,
        min_metric: float | None = None,
        width_tolerance: int = 0,
        include_retired: bool = False,
        limit: int = 0,
    ) -> list[LibraryEntry]:
        """Entries matching the structural filters, best first by (robustness, accepted metric).

        `width_tolerance` widens the width filters by +/- that many absolute columns (glue adapts
        widths at the composition level, so near-width entries are genuinely reusable there; exact
        match remains the default because a stored net cannot RUN on foreign widths without glue).
        Retired (tombstoned) entries are hidden unless `include_retired` is set."""
        matches: list[dict[str, Any]] = []
        for summary in self._index.values():
            if not include_retired and summary.get("retired", False):
                continue
            if entry_type is not None and summary["entry_type"] != entry_type:
                continue
            inputs = summary["io"]["inputs"]
            output = summary["io"]["output"]
            if input_signature is not None and not any(item["signature"] == input_signature for item in inputs):
                continue
            if input_width is not None and not any(abs(item["width"] - input_width) <= width_tolerance for item in inputs):
                continue
            if output_signature is not None and output["signature"] != output_signature:
                continue
            if output_width is not None and not abs(output["width"] - output_width) <= width_tolerance:
                continue
            if min_metric is not None and summary["accepted_metric"] < min_metric:
                continue
            matches.append(summary)
        matches.sort(key=lambda item: (item["weight_robustness"], item["accepted_metric"]), reverse=True)
        if limit:
            matches = matches[:limit]
        return [self.load(summary["key"]) for summary in matches]

    def bump_stats(self, key: str, attributed_fitness: float) -> None:
        """Record a use of `key` in an assembled network (provenance for future ranking).

        The HOT stats writer (once per attributed ref per composition generation), so it defers all
        I/O: the index row mutates in place (shared with loaded handles via the `load` overlay) and
        `flush_stats` persists dirty rows at task boundaries. Structural writers (add/retire/refine)
        stay immediate."""
        summary = self._index.get(key)
        if summary is None:
            return
        stats = summary.setdefault("stats", {"use_count": 0, "max_attributed_fitness": 0.0})
        stats["use_count"] = int(stats.get("use_count", 0)) + 1
        stats["max_attributed_fitness"] = max(float(stats.get("max_attributed_fitness", 0.0)), attributed_fitness)
        self._dirty_stats.add(key)

    def flush_stats(self) -> None:
        """Persist deferred `bump_stats` mutations: rewrite each dirty entry's file, then the index
        ONCE. The orchestrated trial calls this after every task (and from its crash handler), which
        is the durability granularity the observability contract promises. No-op when clean."""
        if not self._dirty_stats:
            return
        for key in sorted(self._dirty_stats):
            summary = self._index.get(key)
            if summary is None:
                continue  # swept while dirty; nothing durable to update
            entry = self.load(key)
            entry.stats = summary.setdefault("stats", entry.stats)
            (self._entries_dir / f"{key}.json").write_text(json.dumps(entry.to_dict(), indent=2))
        self._dirty_stats.clear()
        self._write_index()

    def summary(self, key: str) -> dict[str, Any] | None:
        """A copy of the index row for `key` (ranking fields + stats), or None if unknown."""
        summary = self._index.get(key)
        return dict(summary) if summary is not None else None

    def record_refinement(self, key: str, *, improved: bool) -> None:
        """Record a learn-mode refinement attempt against `key`. The refine keys appear LAZILY on
        first call (never in the stats defaults or `add`), so libraries untouched by refinement
        stay byte-identical on disk."""
        summary = self._index.get(key)
        if summary is None:
            return
        stats = summary.setdefault("stats", {"use_count": 0, "max_attributed_fitness": 0.0})
        stats["refine_attempts"] = int(stats.get("refine_attempts", 0)) + 1
        stats["refine_failures_since_gain"] = 0 if improved else int(stats.get("refine_failures_since_gain", 0)) + 1
        entry = self.load(key)
        entry.stats = stats
        (self._entries_dir / f"{key}.json").write_text(json.dumps(entry.to_dict(), indent=2))
        self._write_index()

    def seed_refine_stats(self, key: str, *, attempts: int, failures: int) -> None:
        """Start `key`'s refine ledger from its lineage instead of zero: a refined replacement is
        the SAME solution continuing under a new key, so the family's cooldown must ride the chain
        (a fresh account per polish pass is the treadmill that filled the library with lineage
        variants of one solver). Lazy keys, like `record_refinement`."""
        summary = self._index.get(key)
        if summary is None:
            return
        stats = summary.setdefault("stats", {"use_count": 0, "max_attributed_fitness": 0.0})
        stats["refine_attempts"] = int(attempts)
        stats["refine_failures_since_gain"] = int(failures)
        entry = self.load(key)
        entry.stats = stats
        (self._entries_dir / f"{key}.json").write_text(json.dumps(entry.to_dict(), indent=2))
        self._write_index()


def macro_resolver(library: "ModuleLibrary") -> Callable[[str], Genome]:
    """Decode-time resolver for macro refs. Entries are immutable, so genomes cache per key."""
    cache: dict[str, Genome] = {}

    def resolve(key: str) -> Genome:
        if key not in cache:
            entry = library.load(key)
            if entry.entry_type != MODULE:
                raise ValueError(f"macro ref {key!r} is not a module entry")
            cache[key] = genome_from_dict(entry.payload)
        return cache[key]

    return resolve


def module_level(genome: Genome, library: "ModuleLibrary") -> int:
    """Entry level for a module genome: plain modules are level 1; a module embedding macros sits
    one level above its deepest reference (the same convention compositions use)."""
    levels = [library.load(macro.ref.removeprefix("library:")).level for macro in genome.macros]
    return 1 + max(levels, default=0)


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
    macros = [
        MacroGene(
            ref=macro.ref,
            input_node_ids=tuple(id_map[node_id] for node_id in macro.input_node_ids),
            output_node_ids=tuple(id_map[node_id] for node_id in macro.output_node_ids),
            innovation=tracker.new_marker(),
            trainable=macro.trainable,
        )
        for macro in source.macros
    ]
    return Genome(nodes=nodes, connections=connections, macros=macros)
