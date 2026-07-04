"""Module library: add/query/load, dedupe, structural signatures, graft, and the flat-loop mutation."""

import random
from pathlib import Path

import torch

from ardevo.dataset.icarus import Task, TaskKind, TaskMeta
from ardevo.evolution.genome import ConnectionGene, Genome, InnovationTracker, NodeGene, NodeKind, genome_to_dict
from ardevo.evolution.mutation import MutationContext, add_library_module
from ardevo.library import MODULE, LibraryEntry, ModuleLibrary, graft, task_io
from ardevo.substrate import decode

_IO = {"inputs": [{"signature": "BINARY|K", "width": 2}], "output": {"signature": "BINARY|K", "width": 1}}


def _module_entry(library: ModuleLibrary, genome: Genome, *, metric: float = 0.9, robustness: float = 0.5) -> str:
    return library.add(
        entry_type=MODULE,
        payload=genome_to_dict(genome),
        io=_IO,
        provenance={"task": "xor", "rung": 1, "accepted_metric": metric, "weight_robustness": robustness},
    )


def test_add_load_round_trip_and_dedupe(tmp_path: Path, solving_genome: Genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    key = _module_entry(library, solving_genome)
    again = _module_entry(library, solving_genome)
    assert key == again and len(library) == 1
    entry = library.load(key)
    assert isinstance(entry, LibraryEntry)
    assert entry.payload == genome_to_dict(solving_genome)
    assert entry.level == 1 and entry.entry_type == MODULE


def test_persistence_across_reopen(tmp_path: Path, solving_genome: Genome) -> None:
    root = tmp_path / "lib"
    key = _module_entry(ModuleLibrary(root), solving_genome)
    reopened = ModuleLibrary(root)
    assert reopened.keys() == [key]
    assert reopened.load(key).payload == genome_to_dict(solving_genome)


def test_query_filters_and_ranking(tmp_path: Path, solving_genome: Genome, linear_genome: Genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    weak = _module_entry(library, linear_genome, metric=0.5, robustness=0.1)
    strong = _module_entry(library, solving_genome, metric=1.0, robustness=0.8)
    ranked = library.query(entry_type=MODULE, input_signature="BINARY|K")
    assert [entry.key for entry in ranked] == [strong, weak]  # robustness-first ordering
    assert [entry.key for entry in library.query(min_metric=0.9)] == [strong]
    assert library.query(input_signature="CONTINUOUS|C") == []
    assert library.query(output_width=1, input_width=2, limit=1)[0].key == strong


def test_task_io_ignores_meta(xor_task: Task) -> None:
    """ARC-portability invariant: signatures derive from Field descriptors, never rung/name/kind."""
    relabeled = Task(
        meta=TaskMeta(rung=17, kind=TaskKind.INTERACTIVE, name="something_else", fixed_split=False),
        support=list(xor_task.support),
        query=list(xor_task.query),
    )
    assert task_io(xor_task) == task_io(relabeled)
    assert task_io(xor_task)["inputs"][0] == {"signature": "BINARY|K", "width": 2}
    assert task_io(xor_task)["output"]["width"] == 1


def test_graft_allocates_fresh_ids_and_preserves_structure(tmp_path: Path, solving_genome: Genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    key = _module_entry(library, solving_genome)
    tracker = InnovationTracker(_next_node_id=100)
    grafted = graft(library.load(key), tracker)
    assert len(grafted.nodes) == len(solving_genome.nodes)
    assert all(node_id >= 100 for node_id in grafted.nodes)
    assert sorted(conn.weight for conn in grafted.connections) == sorted(conn.weight for conn in solving_genome.connections)
    assert {node.activation for node in grafted.nodes.values()} == {node.activation for node in solving_genome.nodes.values()}
    # Same module grafted twice gets DIFFERENT ids but stable per-graft innovations via the tracker.
    second = graft(library.load(key), tracker)
    assert set(second.nodes).isdisjoint(set(grafted.nodes))


def test_add_library_module_inlines_and_stays_decodable(tmp_path: Path, solving_genome: Genome, linear_genome: Genome) -> None:
    library_path = tmp_path / "lib"
    _module_entry(ModuleLibrary(library_path), solving_genome)
    ctx = MutationContext(innovations=InnovationTracker.from_genomes([linear_genome]), activations=["tanh"], default_activation="tanh")
    child = add_library_module(linear_genome, ctx, rng=random.Random(0), prob=1.0, path=str(library_path))
    # 6 source nodes, bias remaps onto the host bias: 5 fresh HIDDEN nodes.
    assert len(child.nodes) == len(linear_genome.nodes) + 5
    new_ids = set(child.nodes) - set(linear_genome.nodes)
    assert all(child.nodes[node_id].kind is NodeKind.HIDDEN for node_id in new_ids)
    module = decode(child, n_inputs=2, n_outputs=1)  # topological_order inside would raise on a cycle
    out = module(torch.tensor([[0.0, 1.0], [1.0, 1.0]]))
    assert out.shape == (2, 1)


def test_add_library_module_wires_distinct_host_sources(tmp_path: Path, solving_genome: Genome, linear_genome: Genome) -> None:
    """Bug-8 regression: the inlined module's input ports must read DISTINCT host sources (sampled
    without replacement), not all the same signal, when the host has enough sources."""
    library_path = tmp_path / "lib"
    _module_entry(ModuleLibrary(library_path), solving_genome)
    ctx = MutationContext(innovations=InnovationTracker.from_genomes([linear_genome]), activations=["tanh"], default_activation="tanh")
    child = add_library_module(linear_genome, ctx, rng=random.Random(0), prob=1.0, path=str(library_path))
    host_sources = set(linear_genome.input_ids) | set(linear_genome.bias_ids) | set(linear_genome.hidden_ids)
    new_ids = set(child.nodes) - set(linear_genome.nodes)
    input_wires = [conn for conn in child.connections if conn.in_id in host_sources and conn.out_id in new_ids and conn.weight == 1.0]
    sources = [conn.in_id for conn in input_wires]
    assert len(sources) >= 2 and len(sources) == len(set(sources))  # distinct, no duplicate wiring


def test_dedupe_refreshes_ranking_metadata(tmp_path: Path, solving_genome: Genome) -> None:
    """B7 regression: a re-admission is fresh evidence; ranking fields take the max."""
    library = ModuleLibrary(tmp_path / "lib")
    key = _module_entry(library, solving_genome, metric=0.5, robustness=0.1)
    library.bump_stats(key, attributed_fitness=0.4)
    again = library.add(
        entry_type=MODULE,
        payload=genome_to_dict(solving_genome),
        io=_IO,
        provenance={"task": "xor", "rung": 1, "accepted_metric": 0.9, "weight_robustness": 0.7},
    )
    assert again == key
    ranked = library.query(min_metric=0.8)
    assert [entry.key for entry in ranked] == [key]  # index metric refreshed to 0.9
    entry = library.load(key)
    assert entry.provenance["accepted_metric"] == 0.5  # original provenance preserved
    assert len(entry.provenance["readmissions"]) == 1
    assert entry.provenance["readmissions"][0]["accepted_metric"] == 0.9
    assert entry.stats["use_count"] == 1  # stats untouched by readmission


def test_retire_tombstones_but_never_deletes(tmp_path: Path, solving_genome: Genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    key = _module_entry(library, solving_genome)
    library.retire(key)
    assert library.is_retired(key)
    assert library.query(entry_type=MODULE) == []  # hidden from search
    assert [entry.key for entry in library.query(entry_type=MODULE, include_retired=True)] == [key]
    assert library.load(key).payload == genome_to_dict(solving_genome)  # refs never dangle
    reopened = ModuleLibrary(tmp_path / "lib")
    assert reopened.is_retired(key)  # persisted


def test_width_tolerance_enables_near_miss_reuse(tmp_path: Path, solving_genome: Genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    _module_entry(library, solving_genome)  # io is 2 -> 1
    assert library.query(input_width=4) == []
    assert len(library.query(input_width=4, width_tolerance=2)) == 1
    assert library.query(input_width=4, output_width=4, width_tolerance=2) == []  # output off by 3 exceeds tolerance
    assert len(library.query(input_width=4, output_width=2, width_tolerance=2)) == 1


def test_v1_on_disk_format_still_loads_and_assembles() -> None:
    """Format-compat pin: entries admitted by phase 3 (no retired/dependency keys) keep working."""
    import torch

    from ardevo.evolution.composition import AssemblyContext, assemble, comp_from_dict

    library = ModuleLibrary(Path(__file__).parent / "fixtures" / "library_v1")
    assert len(library) == 4
    composition_entries = library.query(entry_type="composition")
    assert composition_entries, "v1 compositions must be queryable with default flags"
    top = library.load("c3_87a6e67226b1")
    comp = comp_from_dict(top.payload)
    width = top.io["inputs"][0]["width"]
    ctx = AssemblyContext(bank_columns={top.io["inputs"][0]["signature"]: list(range(width))}, library=library)
    net = assemble(comp, ctx, n_inputs=width)
    out = net(torch.zeros(2, width))
    assert out.shape == (2, top.io["output"]["width"])
    for key in library.keys():
        assert not library.is_retired(key)  # missing flag defaults to live


def test_oversized_payload_warns_but_admits(tmp_path: Path, caplog) -> None:
    import logging

    library = ModuleLibrary(tmp_path / "lib")
    payload = {"nodes": [], "connections": [], "macros": [], "blob": "x" * 2_100_000}
    ardevo_logger = logging.getLogger("ardevo")  # propagate=False, so caplog needs a direct handler
    ardevo_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="ardevo"):
            key = library.add(entry_type=MODULE, payload=payload, io=_IO, provenance={})
    finally:
        ardevo_logger.removeHandler(caplog.handler)
    assert key in library.keys()  # champions are never dropped for size
    assert any("glue_rank_threshold" in record.message for record in caplog.records)


def test_bump_stats_tracks_use(tmp_path: Path, solving_genome: Genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    key = _module_entry(library, solving_genome)
    library.bump_stats(key, attributed_fitness=0.7)
    library.bump_stats(key, attributed_fitness=0.4)
    entry = library.load(key)
    assert entry.stats["use_count"] == 2
    assert entry.stats["max_attributed_fitness"] == 0.7


def test_bump_stats_defers_disk_writes_until_flush(tmp_path: Path, solving_genome: Genome) -> None:
    """The hot stats path is write-deferred: bumps are visible through every in-process handle
    immediately, hit disk only on flush_stats (the per-task boundary), and flushing twice is safe."""
    root = tmp_path / "lib"
    library = ModuleLibrary(root)
    key = _module_entry(library, solving_genome)
    library.bump_stats(key, attributed_fitness=0.7)
    library.bump_stats(key, attributed_fitness=0.4)
    assert library.load(key).stats["use_count"] == 2  # live handle sees the bumps
    stale = ModuleLibrary(root)
    assert stale.load(key).stats["use_count"] == 0  # nothing on disk yet
    library.flush_stats()
    fresh = ModuleLibrary(root)
    summary = fresh.summary(key)
    assert summary is not None and summary["stats"]["use_count"] == 2  # index row persisted
    assert fresh.load(key).stats["max_attributed_fitness"] == 0.7  # entry file persisted
    library.flush_stats()  # clean flush is a no-op


def test_record_refinement_increments_and_resets(tmp_path: Path, solving_genome: Genome) -> None:
    root = tmp_path / "lib"
    library = ModuleLibrary(root)
    key = _module_entry(library, solving_genome)
    library.record_refinement(key, improved=False)
    library.record_refinement(key, improved=False)
    entry = library.load(key)
    assert entry.stats["refine_attempts"] == 2
    assert entry.stats["refine_failures_since_gain"] == 2
    library.record_refinement(key, improved=True)  # a shelved gain resets the decay
    reopened = ModuleLibrary(root)  # persisted through both index.json and the entry JSON
    summary = reopened.summary(key)
    assert summary is not None
    assert summary["stats"]["refine_attempts"] == 3
    assert summary["stats"]["refine_failures_since_gain"] == 0
    assert reopened.load(key).stats["refine_failures_since_gain"] == 0
    library.record_refinement("missing", improved=True)  # unknown keys are a no-op, like bump_stats


def test_summary_accessor_and_legacy_stats_default(tmp_path: Path, solving_genome: Genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    assert library.summary("missing") is None
    key = _module_entry(library, solving_genome)
    summary = library.summary(key)
    assert summary is not None and summary["key"] == key
    # A never-refined entry reads as zero failures AND carries no refine keys at all, so libraries
    # untouched by learn mode stay byte-identical on disk.
    assert "refine_attempts" not in summary["stats"]
    assert int(summary["stats"].get("refine_failures_since_gain", 0)) == 0


def test_summaries_filters_retired_and_dependency(tmp_path: Path, solving_genome: Genome, linear_genome: Genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    kept = _module_entry(library, solving_genome)
    dependency = library.add(entry_type=MODULE, payload=genome_to_dict(linear_genome), io=_IO, provenance={"dependency": True})
    third = Genome(
        nodes={0: NodeGene(0, NodeKind.INPUT, "identity"), 1: NodeGene(1, NodeKind.OUTPUT, "identity")},
        connections=[ConnectionGene(0, 1, 0.123, True, 0)],
    )
    retired = library.add(entry_type=MODULE, payload=genome_to_dict(third), io=_IO, provenance={})
    library.retire(retired)

    assert {row["key"] for row in library.summaries()} == {kept, dependency}
    assert {row["key"] for row in library.summaries(include_dependencies=False)} == {kept}
    assert {row["key"] for row in library.summaries(include_retired=True)} == {kept, dependency, retired}
    rows = library.summaries(include_retired=True)
    assert rows == sorted(rows, key=lambda row: (row["level"], row["key"]))


def test_structural_fingerprint_ignores_weights_but_not_topology(tmp_path: Path, solving_genome: Genome) -> None:
    """The refine identity: a retrained clone shares the fingerprint (entry KEYS never do, they
    hash weights); any structural edit, however small, changes it."""
    from ardevo.library import structural_fingerprint

    payload = genome_to_dict(solving_genome)
    reweighted = genome_to_dict(solving_genome)
    for connection in reweighted["connections"]:
        connection["weight"] += 0.75
    renumbered = genome_to_dict(solving_genome)
    for offset, connection in enumerate(renumbered["connections"]):
        connection["innovation"] += 100 + offset
    assert structural_fingerprint(MODULE, payload) == structural_fingerprint(MODULE, reweighted)
    assert structural_fingerprint(MODULE, payload) == structural_fingerprint(MODULE, renumbered)

    library = ModuleLibrary(tmp_path / "lib")
    assert _module_entry(library, solving_genome) != library.add(entry_type=MODULE, payload=reweighted, io=_IO, provenance={})  # keys DO differ

    toggled = genome_to_dict(solving_genome)
    toggled["connections"][0]["enabled"] = not toggled["connections"][0]["enabled"]
    deepened = genome_to_dict(solving_genome)
    deepened["refine_steps"] = int(deepened.get("refine_steps", 1)) + 2
    assert structural_fingerprint(MODULE, payload) != structural_fingerprint(MODULE, toggled)
    assert structural_fingerprint(MODULE, payload) != structural_fingerprint(MODULE, deepened)


def test_structural_fingerprint_composition_ignores_glue_values() -> None:
    import random as random_module

    from ardevo.evolution.composition import comp_to_dict, minimal_composition
    from ardevo.library import COMPOSITION, structural_fingerprint

    comp = minimal_composition([("BINARY|K", 2)], "xor", 1, InnovationTracker(_next_node_id=0), random_module.Random(1))
    baseline = comp_to_dict(comp)
    reglued = comp_to_dict(comp)
    for edge in reglued["edges"]:
        edge["glue"] = [value + 1.5 for value in edge["glue"]]
    disabled = comp_to_dict(comp)
    disabled["edges"][0]["enabled"] = False
    assert structural_fingerprint(COMPOSITION, baseline) == structural_fingerprint(COMPOSITION, reglued)
    assert structural_fingerprint(COMPOSITION, baseline) != structural_fingerprint(COMPOSITION, disabled)


def test_payload_refs_extracts_comp_and_macro_keys(tmp_path: Path, solving_genome: Genome) -> None:
    import random as random_module

    from ardevo.evolution.composition import CompNodeGene, CompNodeKind, comp_to_dict, minimal_composition
    from ardevo.library import COMPOSITION, payload_refs

    assert payload_refs(MODULE, genome_to_dict(solving_genome)) == set()  # no macros, no refs
    macro_payload = genome_to_dict(solving_genome)
    macro_payload["macros"] = [{"ref": "library:m1_aaa", "inputs": [0], "outputs": [1], "innovation": 7, "trainable": False}]
    assert payload_refs(MODULE, macro_payload) == {"m1_aaa"}

    comp = minimal_composition([("BINARY|K", 2)], "xor", 1, InnovationTracker(_next_node_id=0), random_module.Random(1))
    comp.nodes[99] = CompNodeGene(99, CompNodeKind.MODULE, "library:m1_bbb", 2, 1)
    assert payload_refs(COMPOSITION, comp_to_dict(comp)) == {"m1_bbb"}  # INPUT bank refs are not library keys


def test_collect_garbage_sweeps_cascades_and_protects(tmp_path: Path, solving_genome: Genome) -> None:
    """Only retired-and-unreferenced tombstones go; references from RETAINED entries (live or
    protected) pin their targets; a retired chain falls together; dry-run touches nothing."""
    import random as random_module

    from ardevo.evolution.composition import CompNodeGene, CompNodeKind, comp_to_dict, minimal_composition
    from ardevo.library import COMPOSITION

    library = ModuleLibrary(tmp_path / "lib")

    def module(weight_bump: float) -> str:
        payload = genome_to_dict(solving_genome)
        payload["connections"][0]["weight"] += weight_bump
        return library.add(entry_type=MODULE, payload=payload, io=_IO, provenance={"accepted_metric": 1.0})

    def comp_over(ref_key: str, bump: int) -> str:
        comp = minimal_composition([("BINARY|K", 2)], "xor", 1, InnovationTracker(_next_node_id=100 * bump), random_module.Random(bump))
        comp.nodes[999] = CompNodeGene(999, CompNodeKind.MODULE, f"library:{ref_key}", 2, 1)
        return library.add(entry_type=COMPOSITION, payload=comp_to_dict(comp), io=_IO, provenance={"accepted_metric": 1.0}, level=2)

    live_dep = module(1.0)  # retired below, but referenced by the LIVE comp: must survive
    live_comp = comp_over(live_dep, 1)
    chain_dep = module(2.0)  # referenced only by the RETIRED comp: falls with it
    chain_comp = comp_over(chain_dep, 2)
    loose = module(3.0)  # retired, unreferenced: goes
    protected = module(4.0)  # retired, unreferenced, but protected (e.g. a checkpoint macro ref)
    for key in (live_dep, chain_dep, chain_comp, loose, protected):
        library.retire(key)
    image = tmp_path / "lib" / "images" / f"{loose}.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"png")

    would_sweep = library.collect_garbage(protect=[protected], dry_run=True)
    assert set(would_sweep) == {chain_dep, chain_comp, loose}
    assert len(library) == 6 and image.exists()  # dry run touched nothing

    swept = library.collect_garbage(protect=[protected])
    assert set(swept) == {chain_dep, chain_comp, loose}
    assert set(library.keys()) == {live_dep, live_comp, protected}
    assert not image.exists()
    library.load(live_dep)  # the live comp's dependency still loads
    import pytest as pytest_module

    with pytest_module.raises(KeyError):
        library.load(loose)
