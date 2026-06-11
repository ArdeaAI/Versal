"""Orchestrator: ladder order, stall detection, decomposition recursion, admission, anti-forgetting."""

import json
import random
from pathlib import Path

import torch

from ardevo import checkpoint
from ardevo.dataset.icarus import Task
from ardevo.evolution.composition import AssemblyContext, assemble, comp_from_dict, minimal_composition
from ardevo.evolution.genome import InnovationTracker, genome_to_dict
from ardevo.evolution.loop import AssessedComposition, state_from_dict, state_to_dict
from ardevo.evolution.registry import build_loop
from ardevo.library import COMPOSITION, MODULE, ModuleLibrary, task_io
from ardevo.orchestrator import Orchestrator, StallDetector, attempts_from_dicts, attempts_to_dicts, comp_task_spec
from ardevo.trials.orchestrated_trial import OrchestratedTrial
from tests.test_hierarchical_loop import _config as _loop_config


def _orchestrator(tmp_path: Path, **overrides) -> Orchestrator:
    config = _loop_config()
    config["orchestrator"] = {
        "tasks": 4,
        "accept_metric": "query_accuracy",
        "accept_threshold": 0.95,
        "floor": 0.05,
        "stall_generations": 2,
        "stall_epsilon": 0.005,
        "max_depth": 2,
        "quick_eval_top_k": 5,
        "decompose": ["output_slices", "input_subsets"],
        "output_slices_n_groups": 2,
        "input_subsets_n_subsets": 2,
        "budgets": {"depth0": 3, "depth1": 2, "depth2": 2},
        **overrides.pop("table", {}),
    }
    loop = build_loop(config)
    library = ModuleLibrary(tmp_path / "lib")
    loop.attach_library(library)
    state = loop.fresh_state(random.Random(0))
    return Orchestrator(config, loop, library, state, **overrides)


def _patch_run_task(orchestrator: Orchestrator, fake) -> None:
    # Dynamic method shadowing for white-box ladder tests; setattr keeps the type checker out of it.
    setattr(orchestrator.loop, "run_task", fake)


def _fake_run_task(metric_by_ref: dict[str, float], calls: list[str]):
    def run_task(spec, state, *, budget, stop=None, seed_comps=None, on_generation=None):
        calls.append(spec.output_ref)
        metric = metric_by_ref.get(spec.output_ref, 0.0)
        if seed_comps:
            comp = seed_comps[0]  # pretend the wired skeleton won, so its refs flow into admission
        else:
            comp = minimal_composition(spec.input_specs, spec.output_ref, spec.output_width, state.comp_innovations, state.rng)
        metrics = {"query_accuracy": metric, "query_loss": 0.1, "support_accuracy": metric, "support_loss": 0.1}
        return AssessedComposition(comp=comp, metrics=metrics, fitness=metric, net=None)

    return run_task


def test_library_hit_short_circuits_evolution(tmp_path: Path, xor_task: Task, solving_genome) -> None:
    orchestrator = _orchestrator(tmp_path)
    orchestrator.library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={"accepted_metric": 1.0})
    calls: list[str] = []
    _patch_run_task(orchestrator, _fake_run_task({}, calls))  # would record any evolve attempt
    solution = orchestrator.solve(xor_task)
    assert solution is not None and solution.metric == 1.0
    assert calls == []  # not one generation spent
    assert orchestrator.attempts[-1].outcome == "library_hit"
    assert orchestrator.counters["library_hits"] == 1


def test_stall_triggers_decompose_recurse_and_level2_admission(tmp_path: Path, decomposable_task: Task) -> None:
    orchestrator = _orchestrator(tmp_path)
    calls: list[str] = []
    metrics = {"half_parity": 0.2, "half_parity.out0": 1.0, "half_parity.out1": 1.0}

    def staged_run_task(spec, state, **kwargs):
        if spec.output_ref == "half_parity" and "half_parity" in calls:
            return _fake_run_task({"half_parity": 1.0}, calls)(spec, state, **kwargs)  # the re-evolve after wiring
        return _fake_run_task(metrics, calls)(spec, state, **kwargs)

    _patch_run_task(orchestrator, staged_run_task)
    solution = orchestrator.solve(decomposable_task)
    assert solution is not None and solution.entry_type == COMPOSITION
    assert calls == ["half_parity", "half_parity.out0", "half_parity.out1", "half_parity"]
    outcomes = [attempt.outcome for attempt in orchestrator.attempts]
    assert outcomes == ["evolved", "evolved", "decomposed"]  # two depth-1 accepts, then the wired parent
    assert orchestrator.counters["decompositions"] == 1
    entry = orchestrator.library.load(solution.key)
    assert entry.level == 3  # composition over level-2 sub-compositions
    assert len(orchestrator.library) == 3


def test_max_depth_prevents_decomposition(tmp_path: Path, decomposable_task: Task) -> None:
    orchestrator = _orchestrator(tmp_path, table={"max_depth": 0})
    calls: list[str] = []
    _patch_run_task(orchestrator, _fake_run_task({}, calls))
    solution = orchestrator.solve(decomposable_task)
    assert solution is None
    assert orchestrator.attempts[-1].outcome == "failed"
    assert orchestrator.counters["decompositions"] == 0
    assert calls == ["half_parity"]  # one evolve attempt, no recursion


def test_failed_subtask_fails_the_decomposition_gracefully(tmp_path: Path, decomposable_task: Task) -> None:
    orchestrator = _orchestrator(tmp_path)
    calls: list[str] = []
    _patch_run_task(orchestrator, _fake_run_task({"half_parity.out0": 1.0}, calls))  # out1 never clears
    solution = orchestrator.solve(decomposable_task)
    assert solution is None
    assert orchestrator.attempts[-1].outcome == "failed"
    # out1 itself recursed to depth 2 (output_slices cannot split a 1-wide output, so it went
    # straight to failed) before the parent gave up.
    assert any(attempt.task == "half_parity.out1" and attempt.outcome == "failed" for attempt in orchestrator.attempts)


def test_stall_detector_flatline_and_floor() -> None:
    from ardevo.evolution.composition import CompositionGenome

    def metric_of(item: AssessedComposition) -> float:
        return item.metrics.get("query_accuracy", 0.0)

    def assessed(accuracy: float, fitness: float) -> AssessedComposition:
        return AssessedComposition(comp=CompositionGenome(), metrics={"query_accuracy": accuracy}, fitness=fitness, net=None)

    flat = assessed(0.9, 0.5)
    detector = StallDetector(stall_generations=3, stall_epsilon=0.01, floor=0.2, budget=100, metric_of=metric_of)
    assert not detector(0, flat)  # the first observation counts as the baseline improvement
    assert not detector(1, flat) and not detector(2, flat)
    assert detector(3, flat)  # three consecutive generations without improvement
    improving = StallDetector(stall_generations=3, stall_epsilon=0.01, floor=0.2, budget=100, metric_of=metric_of)
    for generation in range(6):
        assert not improving(generation, assessed(0.9, 0.1 * generation))
    hopeless = assessed(0.1, 1.0)
    floor_detector = StallDetector(stall_generations=50, stall_epsilon=0.01, floor=0.2, budget=10, metric_of=metric_of)
    assert not floor_detector(0, hopeless)  # before half budget the floor is not checked
    assert floor_detector(5, hopeless)


def test_port_wired_skeleton_assembles_and_routes(tmp_path: Path, decomposable_task: Task) -> None:
    from ardevo.decompose import output_slices
    from ardevo.evolution.composition import comp_to_dict
    from ardevo.orchestrator import Solution

    orchestrator = _orchestrator(tmp_path)
    spec = comp_task_spec(decomposable_task)
    subtasks = output_slices(decomposable_task, rng=random.Random(0), n_groups=2)
    solutions = []
    for subtask in subtasks:
        comp = minimal_composition([("BINARY|K", 8)], subtask.task.meta.name, 1, InnovationTracker(_next_node_id=0), random.Random(1))
        key = orchestrator.library.add(entry_type=COMPOSITION, payload=comp_to_dict(comp), io=task_io(subtask.task), provenance={}, level=2)
        solutions.append((subtask, Solution(key=key, entry_type=COMPOSITION, metric=1.0)))
    skeleton = orchestrator._port_wired_skeleton(spec, solutions)
    assert skeleton is not None and len(skeleton.module_ids) == 2
    net = assemble(skeleton, AssemblyContext(bank_columns=dict(spec.bank_columns), library=orchestrator.library), spec.n_inputs)
    out = net(torch.zeros(3, 8))
    assert out.shape == (3, 2)


def test_real_evolve_admit_then_revisit_is_a_hit(tmp_path: Path, xor_task: Task) -> None:
    """The anti-forgetting regression test, on the REAL loop: accept a (weak) solution, admit it,
    and the same task seen again is answered from the library without evolving."""
    orchestrator = _orchestrator(tmp_path, table={"accept_threshold": 0.2, "decompose": [], "budgets": {"depth0": 2}})
    first = orchestrator.solve(xor_task)
    assert first is not None
    assert orchestrator.attempts[-1].outcome == "evolved"
    assert len(orchestrator.library) >= 1
    second = orchestrator.solve(xor_task)
    assert second is not None
    assert orchestrator.attempts[-1].outcome == "library_hit"
    assert orchestrator.counters["library_hits"] == 1 and orchestrator.counters["accepts"] == 1


def test_admission_detaches_live_refs(tmp_path: Path, xor_task: Task) -> None:
    """Admitted compositions must reference only library entries (live species are run-local)."""
    orchestrator = _orchestrator(tmp_path, table={"accept_threshold": 0.0, "decompose": [], "budgets": {"depth0": 2}})
    solution = orchestrator.solve(xor_task)
    assert solution is not None
    entry = orchestrator.library.load(solution.key)
    comp = comp_from_dict(entry.payload)
    assert all(ref.startswith("library:") for ref in comp.refs())


def test_orchestrated_checkpoint_writes_artifacts_only_for_new_library_entries(tmp_path: Path, xor_task: Task) -> None:
    orchestrator = _orchestrator(tmp_path, table={"accept_threshold": 0.0, "decompose": [], "budgets": {"depth0": 1}})
    before = set(orchestrator.library.keys())
    solution = orchestrator.solve(xor_task)
    assert solution is not None
    new_keys = [key for key in orchestrator.library.keys() if key not in before]
    assert new_keys

    trial = object.__new__(OrchestratedTrial)
    trial.config = {"orchestrator": {}, "schedule": {}, "evolution": {}, "fitness": {}, "dataset": "synthetic"}
    trial.library = orchestrator.library
    trial.loop = orchestrator.loop
    trial.scheduler = type("Scheduler", (), {"state_dict": lambda self: {"last": 0}})()
    trial.run_dir = tmp_path / "run"
    trial.task = None

    trial._checkpoint(orchestrator, orchestrator.state, 1, new_keys, solution)
    directory = trial.run_dir / "task_0001"
    assert (directory / "stats.json").exists()
    assert (directory / "checkpoint.json").exists()
    assert (directory / "speciation.png").exists()
    assert (directory / "net.png").exists()

    before_hit = set(orchestrator.library.keys())
    hit = orchestrator.solve(xor_task)
    assert hit is not None
    assert orchestrator.attempts[-1].outcome == "library_hit"
    assert [key for key in orchestrator.library.keys() if key not in before_hit] == []


def test_orchestrated_payload_round_trips(tmp_path: Path, decomposable_task: Task) -> None:
    orchestrator = _orchestrator(tmp_path, table={"accept_threshold": 0.0, "decompose": [], "budgets": {"depth0": 1}})
    orchestrator.solve(decomposable_task)
    payload = checkpoint.build_orchestrated_payload(
        task_cursor=1,
        rng=orchestrator.state.rng,
        scheduler=type("S", (), {"state_dict": lambda self: {"last": 0}})(),
        speciator=orchestrator.loop.evolver.speciate,
        loop_state=state_to_dict(orchestrator.state),
        attempts=attempts_to_dicts(orchestrator.attempts),
        counters=orchestrator.counters,
    )
    raw = json.dumps(payload)  # everything must be JSON-able for checkpoint.json
    restored = json.loads(raw)
    state = state_from_dict(restored["loop_state"], checkpoint.deserialize_rng(restored["rng"]))
    assert state.generation == orchestrator.state.generation
    assert len(state.modules) == len(orchestrator.state.modules)
    attempts = attempts_from_dicts(restored["attempts"])
    assert attempts_to_dicts(attempts) == attempts_to_dicts(orchestrator.attempts)
