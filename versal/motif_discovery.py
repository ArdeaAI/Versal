"""Evidence-ranked motif discovery layered on top of the descriptive raw census."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from versal.library import MODULE, LibraryEntry, ModuleLibrary
from versal.motifs import (
    DEFAULT_COMPOSITION_SIZES,
    CensusReport,
    MotifRecord,
    _load_entries,
    _mine_entry,
    composition_motif_graph,
    module_motif_graph,
)

NULL_GRAPHS = 64
TOP_CANDIDATES = 10
MAX_EXEMPLARS = 3
MATCHED_CONTROLS = 16


def classify_counterfactuals(exemplars: list[dict[str, Any]], *, observed: bool = False) -> dict[str, Any]:
    """Classify frozen knockout measurements without relaxing the registered thresholds.

    Each exemplar supplies ``accuracy_drop``, ``control_drops`` and ``lineage_root``. A functional
    exemplar must lose at least 0.02 accuracy and exceed at least 95% of its matched controls;
    replicated evidence requires functional exemplars from two independent lineage roots.
    """

    functional_roots: set[str] = set()
    classified: list[dict[str, Any]] = []
    for exemplar in exemplars[:MAX_EXEMPLARS]:
        drop = float(exemplar["accuracy_drop"])
        controls = [float(value) for value in exemplar["control_drops"]]
        if len(controls) != MATCHED_CONTROLS:
            raise ValueError(f"counterfactual exemplar requires exactly {MATCHED_CONTROLS} matched controls")
        control_percentile = sum(value < drop for value in controls) / len(controls)
        functional = drop >= 0.02 and control_percentile >= 0.95
        lineage_root = str(exemplar["lineage_root"])
        if functional:
            functional_roots.add(lineage_root)
        classified.append({**exemplar, "control_percentile": control_percentile, "functional": functional})
    if len(functional_roots) >= 2:
        evidence = "replicated"
    elif functional_roots:
        evidence = "functional"
    else:
        evidence = "observed" if observed else "candidate"
    return {"evidence": evidence, "functional_lineage_roots": sorted(functional_roots), "exemplars": classified}


def _percentile(value: float, population: list[float]) -> float:
    if not population:
        return 0.0
    return sum(item <= value for item in population) / len(population)


def _entry_graph(entry: LibraryEntry) -> tuple[dict[int, Any], dict[tuple[int, int], int], tuple[int, ...]]:
    if entry.entry_type == MODULE:
        labels, edges = module_motif_graph(entry.payload)
        return labels, edges, tuple()
    labels, edges = composition_motif_graph(entry.payload)
    return labels, edges, DEFAULT_COMPOSITION_SIZES


def _supporters(entries: list[LibraryEntry], report: CensusReport) -> dict[str, set[str]]:
    wanted = {record.fingerprint for record in report.module_motifs + report.composition_motifs}
    found: dict[str, set[str]] = {fingerprint: set() for fingerprint in wanted}
    cache: dict[Any, Any] = {}
    module_sizes = tuple(int(size) for size in report.params.get("sizes", [3, 4]))
    cap = int(report.params.get("per_entry_cap", 50_000))
    for entry in entries:
        labels, edges, fixed_sizes = _entry_graph(entry)
        sizes = fixed_sizes or module_sizes
        motifs, _truncated = _mine_entry(labels, edges, sizes, cap, cache)
        for fingerprint in wanted & motifs.keys():
            found[fingerprint].add(entry.key)
    return found


def _rewire_degree_preserving(edges: dict[tuple[int, int], int], rng: random.Random) -> dict[tuple[int, int], int]:
    """Directed double-edge swaps preserve every vertex's in/out degree and each edge mask."""

    current = dict(edges)
    if len(current) < 2:
        return current
    for _attempt in range(max(16, len(current) * 12)):
        pairs = list(current)
        first, second = rng.sample(pairs, 2)
        if current[first] != current[second]:
            continue
        source_a, target_a = first
        source_b, target_b = second
        swapped_a, swapped_b = (source_a, target_b), (source_b, target_a)
        if source_a == target_b or source_b == target_a or swapped_a in current or swapped_b in current:
            continue
        mask = current.pop(first)
        current.pop(second)
        current[swapped_a] = mask
        current[swapped_b] = mask
    return current


def _null_supports(entries: list[LibraryEntry], report: CensusReport, fingerprints: set[str]) -> dict[str, list[int]]:
    nulls = {fingerprint: [] for fingerprint in fingerprints}
    module_sizes = tuple(int(size) for size in report.params.get("sizes", [3, 4]))
    cap = int(report.params.get("per_entry_cap", 50_000))
    for replicate in range(NULL_GRAPHS):
        counts = {fingerprint: 0 for fingerprint in fingerprints}
        cache: dict[Any, Any] = {}
        for entry in entries:
            labels, edges, fixed_sizes = _entry_graph(entry)
            seed = int(hashlib.sha256(f"{entry.key}:{replicate}".encode()).hexdigest()[:16], 16)
            rewired = _rewire_degree_preserving(edges, random.Random(seed))
            motifs, _truncated = _mine_entry(labels, rewired, fixed_sizes or module_sizes, cap, cache)
            for fingerprint in fingerprints & motifs.keys():
                counts[fingerprint] += 1
        for fingerprint, count in counts.items():
            nulls[fingerprint].append(count)
    return nulls


def _lineage_root(key: str, entries: dict[str, LibraryEntry]) -> str:
    seen: set[str] = set()
    current = key
    while current not in seen and current in entries:
        seen.add(current)
        provenance = entries[current].provenance
        parent = provenance.get("refined_from") or provenance.get("stepping_stone_from") or provenance.get("parent_key")
        if not isinstance(parent, str) or parent not in entries:
            break
        current = parent
    return current


def _signature(entry: LibraryEntry) -> str:
    return json.dumps(entry.io, sort_keys=True, separators=(",", ":"))


def _is_plumbing(record: MotifRecord) -> bool:
    return not any(label[0] in ("hidden", "module") for label in record.graph.node_labels)


def discover_motifs(library: ModuleLibrary, report: CensusReport, *, counterfactual: bool = False) -> dict[str, Any]:
    """Lock candidates from support/library evidence before any query ledger is opened."""

    entries = _load_entries(library, include_retired=bool(report.params.get("include_retired", False)))
    by_key = {entry.key: entry for entry in entries}
    candidates = [record for record in report.module_motifs + report.composition_motifs if not _is_plumbing(record)]
    supporters = _supporters(entries, report)
    nulls = _null_supports(entries, report, {record.fingerprint for record in candidates})
    vocabulary = {str(row["key"]): row for row in report.vocabulary}
    metrics_by_signature: dict[str, list[float]] = {}
    robustness_by_signature: dict[str, list[float]] = {}
    for entry in entries:
        signature = _signature(entry)
        metrics_by_signature.setdefault(signature, []).append(float(entry.provenance.get("accepted_metric", 0.0)))
        robustness_by_signature.setdefault(signature, []).append(float(entry.provenance.get("weight_robustness", 0.0)))
    reuse_population = [float(row.get("reference_count", 0)) + float(row.get("use_count", 0)) for row in report.vocabulary]

    rows: list[dict[str, Any]] = []
    for record in candidates:
        keys = sorted(supporters[record.fingerprint])
        null_values = nulls[record.fingerprint]
        null_mean = statistics.fmean(null_values) if null_values else 0.0
        null_std = statistics.pstdev(null_values) if len(null_values) > 1 else 0.0
        z_score = (record.support - null_mean) / null_std if null_std > 0 else (math.inf if record.support > null_mean else 0.0)
        lineages = {_lineage_root(key, by_key) for key in keys}
        performance_percentiles: list[float] = []
        robustness_percentiles: list[float] = []
        reuse_percentiles: list[float] = []
        associations: list[float] = []
        for key in keys:
            entry = by_key[key]
            signature = _signature(entry)
            metric = float(entry.provenance.get("accepted_metric", 0.0))
            robustness = float(entry.provenance.get("weight_robustness", 0.0))
            performance_percentiles.append(_percentile(metric, metrics_by_signature[signature]))
            robustness_percentiles.append(_percentile(robustness, robustness_by_signature[signature]))
            reuse_value = float(vocabulary.get(key, {}).get("reference_count", 0)) + float(vocabulary.get(key, {}).get("use_count", 0))
            reuse_percentiles.append(_percentile(reuse_value, reuse_population))
            peers = [float(item.provenance.get("accepted_metric", 0.0)) for item in entries if _signature(item) == signature and item.key not in keys]
            if peers:
                associations.append(metric - statistics.fmean(peers))
        performance = statistics.fmean(performance_percentiles) if performance_percentiles else 0.0
        robustness = statistics.fmean(robustness_percentiles) if robustness_percentiles else 0.0
        reuse = statistics.fmean(reuse_percentiles) if reuse_percentiles else 0.0
        association = statistics.fmean(associations) if associations else 0.0
        finite_surprise = min(z_score, 10.0) if math.isfinite(z_score) else 10.0
        rank_score = finite_surprise + math.log1p(len(lineages)) + performance + robustness + reuse
        evidence = "observed" if z_score >= 2.0 and association > 0 else "candidate"
        rows.append(
            {
                "fingerprint": record.fingerprint,
                "size": record.size,
                "description": record.description,
                "diversity_class": record.diversity_class,
                "support": record.support,
                "null_graphs": NULL_GRAPHS,
                "null_support_mean": null_mean,
                "null_support_std": null_std,
                "surprise_z": z_score if math.isfinite(z_score) else 999.0,
                "independent_lineage_support": len(lineages),
                "performance_percentile": performance,
                "performance_association": association,
                "robustness_percentile": robustness,
                "reuse_percentile": reuse,
                "rank_score": rank_score,
                "evidence": evidence,
                "exemplars": keys[:MAX_EXEMPLARS],
                "counterfactual": {
                    "status": "not_run" if not counterfactual else "unavailable",
                    "recovery_steps": 25,
                    "matched_controls": MATCHED_CONTROLS,
                    "reason": None if not counterfactual else "frozen task tensors are not part of the library evidence; no query data were opened",
                },
            }
        )
    rows.sort(key=lambda row: (-float(row["rank_score"]), str(row["fingerprint"])))
    locked = rows[:TOP_CANDIDATES]
    lock_payload = json.dumps([(row["fingerprint"], row["rank_score"], row["exemplars"]) for row in locked], sort_keys=True)
    return {
        "schema_version": 1,
        "method": {
            "null_graphs": NULL_GRAPHS,
            "null_model": "deterministic directed degree-, edge-kind-, and label-preserving double-edge swaps",
            "candidate_lock": "support/library evidence only, before query inspection",
            "counterfactual_protocol": {"max_exemplars": MAX_EXEMPLARS, "recovery_steps": 25, "matched_edge_controls": MATCHED_CONTROLS},
        },
        "candidate_lock_sha256": hashlib.sha256(lock_payload.encode()).hexdigest(),
        "candidates": locked,
        "excluded_plumbing": len(candidates) - len(rows) + sum(_is_plumbing(record) for record in report.module_motifs + report.composition_motifs),
        "limitations": [
            "The raw census remains suitable for grammar induction but is not architectural-discovery evidence.",
            "Functional and replicated labels require frozen knockout/control measurements; unavailable tests are never promoted.",
            "A candidate must not be described as a new cell, convolution, or invention without replicated evidence.",
        ],
    }


def write_discoveries(run_dir: Path, discovery: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "motif_discoveries.json").write_text(json.dumps(discovery, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Motif discoveries",
        "",
        "Candidates were locked from support/library evidence before query inspection. The raw motif atlas is descriptive and is not evidence of architectural invention.",
        "",
        "| Rank | Fingerprint | Evidence | Surprise z | Lineages | Performance percentile | Robustness percentile | Reuse percentile |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(discovery["candidates"], 1):
        lines.append(
            f"| {rank} | `{row['fingerprint']}` | {row['evidence']} | {row['surprise_z']:.2f} | {row['independent_lineage_support']} | "
            f"{row['performance_percentile']:.3f} | {row['robustness_percentile']:.3f} | {row['reuse_percentile']:.3f} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in discovery["limitations"])
    (run_dir / "motif_discoveries.md").write_text("\n".join(lines) + "\n")

    import matplotlib.pyplot as plt

    rows = discovery["candidates"]
    figure, axis = plt.subplots(figsize=(10, max(3, 0.45 * len(rows) + 1)))
    labels = [str(row["fingerprint"]) for row in reversed(rows)]
    scores = [float(row["rank_score"]) for row in reversed(rows)]
    axis.barh(labels, scores, color="#4c78a8")
    axis.set_xlabel("evidence rank score")
    axis.set_title("Locked motif candidates (descriptive until counterfactual validation)")
    figure.tight_layout()
    figure.savefig(run_dir / "motif_discoveries.png", dpi=180)
    plt.close(figure)
