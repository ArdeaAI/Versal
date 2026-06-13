"""Admission gate: quality floors, per-signature caps with replace-if-better, dependency bypass."""

import random
from pathlib import Path

from ardevo.dataset.icarus import Task
from ardevo.evolution.genome import Genome, InnovationTracker, genome_to_dict
from ardevo.evolution.init import minimal
from ardevo.evolution.mutation import MutationContext, add_rich_node
from ardevo.library import LIBRARY_ADMISSION, MODULE, ModuleLibrary
from tests.test_orchestrator import _orchestrator

_IO = {"inputs": [{"signature": "BINARY|K", "width": 2}], "output": {"signature": "BINARY|K", "width": 1}}


def _distinct_genomes(count: int) -> list[Genome]:
    rng = random.Random(0)
    genomes = []
    for _ in range(count):
        genome = minimal(2, 1, rng=rng)
        ctx = MutationContext(InnovationTracker.from_genomes([genome]), ["tanh"], "tanh")
        genome = add_rich_node(genome, ctx, rng=rng, prob=1.0, fan_in=2)
        genomes.append(genome)
    return genomes


def _admit(library: ModuleLibrary, policy, genome: Genome, *, metric: float, robustness: float, dependency: bool = False) -> str | None:
    provenance = {"accepted_metric": metric, "weight_robustness": robustness}
    if dependency:
        provenance["dependency"] = True
        return library.add(entry_type=MODULE, payload=genome_to_dict(genome), io=_IO, provenance=provenance)
    decision = policy(library, entry_type=MODULE, io=_IO, provenance=provenance)
    if not decision.admit:
        return None
    for key in decision.retire:
        library.retire(key)
    return library.add(entry_type=MODULE, payload=genome_to_dict(genome), io=_IO, provenance=provenance)


def test_quality_floors_reject(tmp_path: Path) -> None:
    policy = LIBRARY_ADMISSION.get("default")(min_metric=0.9, min_robustness=0.3)
    library = ModuleLibrary(tmp_path / "lib")
    genome = _distinct_genomes(1)[0]
    assert _admit(library, policy, genome, metric=0.8, robustness=0.9) is None  # metric floor
    assert _admit(library, policy, genome, metric=0.95, robustness=0.1) is None  # robustness floor
    assert _admit(library, policy, genome, metric=0.95, robustness=0.5) is not None


def test_per_signature_cap_replaces_only_when_better(tmp_path: Path) -> None:
    policy = LIBRARY_ADMISSION.get("default")(per_signature_cap=2)
    library = ModuleLibrary(tmp_path / "lib")
    genomes = _distinct_genomes(4)
    weak = _admit(library, policy, genomes[0], metric=0.95, robustness=0.2)
    strong = _admit(library, policy, genomes[1], metric=0.99, robustness=0.8)
    assert weak is not None and strong is not None

    # Cap reached: a weaker candidate is refused.
    assert _admit(library, policy, genomes[2], metric=0.95, robustness=0.1) is None
    assert not library.is_retired(weak)

    # A better candidate replaces the weakest, which is tombstoned, never deleted.
    better = _admit(library, policy, genomes[3], metric=0.99, robustness=0.9)
    assert better is not None
    assert library.is_retired(weak)
    assert library.load(weak).payload == genome_to_dict(genomes[0])  # still loadable forever
    live = {entry.key for entry in library.query(entry_type=MODULE)}
    assert live == {strong, better}


def test_dependency_entries_bypass_gate_and_cap_ranking(tmp_path: Path) -> None:
    policy = LIBRARY_ADMISSION.get("default")(per_signature_cap=1)
    library = ModuleLibrary(tmp_path / "lib")
    genomes = _distinct_genomes(3)
    top = _admit(library, policy, genomes[0], metric=0.99, robustness=0.9)
    assert top is not None
    # Dependencies always land (a composition must never dangle) and never occupy cap slots.
    dependency = _admit(library, policy, genomes[1], metric=0.0, robustness=0.0, dependency=True)
    assert dependency is not None
    assert len(library.signature_group(MODULE, _IO)) == 1  # the dependency is not in the group
    # The cap still applies to top-level candidates exactly as before.
    assert _admit(library, policy, genomes[2], metric=0.5, robustness=0.5) is None


def test_orchestrator_counts_rejections_and_still_solves(tmp_path: Path, xor_task: Task) -> None:
    """A gate-rejected winner still counts as solved; the ledger records the honest metric."""
    table = {"evolve": ["direct"], "accept_threshold": 0.2, "decompose": [], "budgets": {"depth0": 2}, "direct": {"pop_size": 8, "elitism": 2}}
    orchestrator = _orchestrator(tmp_path, table=table, config_extra={"library": {"admission": "default", "min_metric": 1.01}})
    solution = orchestrator.solve(xor_task)
    assert solution is not None and solution.key is None  # solved, not shelved
    assert orchestrator.counters["admission_rejected"] == 1
    assert orchestrator.counters["accepts"] == 1
    assert orchestrator.attempts[-1].outcome == "evolved" and orchestrator.attempts[-1].library_key is None
    assert len(orchestrator.library) == 0
