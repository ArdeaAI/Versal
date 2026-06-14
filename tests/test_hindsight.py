"""Phase 7 Pillar G: the hindsight library (recycle failed attempts as synthetic stepping stones).

A failed top-level attempt's best genome is re-admitted as a synthetic DEPENDENCY: it solves SOME
task (the one defined by its own outputs), so it is a partial competency a future related attempt can
graft and improve, instead of all that search budget being discarded. Dependencies bypass the shelf
cap and are re-evaluated at lookup, so a synthetic sub-threshold solver never falsely hits."""

from pathlib import Path

from ardevo.dataset.icarus import Task
from ardevo.library import LIBRARY_ADMISSION, MODULE, ModuleLibrary, task_io
from tests.test_orchestrator import _orchestrator


def test_hindsight_policy_admits_synthetic_dependencies_and_gates_the_rest(tmp_path: Path, xor_task: Task) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    policy = LIBRARY_ADMISSION.get("hindsight")(min_metric=0.9)
    io = task_io(xor_task)
    synthetic = policy(library, entry_type=MODULE, io=io, provenance={"synthetic": True, "dependency": True, "accepted_metric": 0.5})
    assert synthetic.admit  # a sub-threshold synthetic stepping stone is kept
    below = policy(library, entry_type=MODULE, io=io, provenance={"accepted_metric": 0.5})
    assert not below.admit  # a normal entry below the floor is still gated by the archive policy
    above = policy(library, entry_type=MODULE, io=io, provenance={"accepted_metric": 0.95})
    assert above.admit


def test_failed_task_admits_a_hindsight_stepping_stone(tmp_path: Path, xor_task: Task) -> None:
    table = {"evolve": ["direct"], "accept_threshold": 1.5, "decompose": [], "budgets": {"depth0": 2}, "direct": {"pop_size": 8, "elitism": 1}}
    orchestrator = _orchestrator(tmp_path, table=table, config_extra={"library": {"admission": "hindsight"}})
    solution = orchestrator.solve(xor_task)  # threshold 1.5 is unreachable (accuracy maxes at 1.0) -> fails
    assert solution is None
    assert orchestrator.counters["failures"] == 1
    dependencies = [summary for summary in orchestrator.library.summaries(include_dependencies=True) if summary.get("dependency")]
    assert dependencies, "the failed champion should be recycled as a synthetic dependency"
    entry = orchestrator.library.load(dependencies[0]["key"])
    assert entry.provenance.get("synthetic") is True and "hindsight" in entry.provenance.get("behavior", [])


def test_hindsight_off_discards_failures(tmp_path: Path, xor_task: Task) -> None:
    table = {"evolve": ["direct"], "accept_threshold": 1.5, "decompose": [], "budgets": {"depth0": 2}, "direct": {"pop_size": 8, "elitism": 1}}
    orchestrator = _orchestrator(tmp_path, table=table, config_extra={"library": {"admission": "archive"}})
    orchestrator.solve(xor_task)
    assert orchestrator.counters["failures"] == 1
    assert orchestrator.library.summaries(include_dependencies=True) == []  # nothing recycled (byte-identical to phase 6)
