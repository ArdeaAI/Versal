"""Module library: add/query/load, dedupe, structural signatures, graft, and the flat-loop mutation."""

import random
from pathlib import Path

import torch

from ardevo.dataset.icarus import Task, TaskKind, TaskMeta
from ardevo.evolution.genome import Genome, InnovationTracker, NodeKind, genome_to_dict
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
