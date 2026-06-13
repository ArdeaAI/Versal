"""Multi-task substrate: grow-the-I/O, per-task output heads, and checkpoint resume-equivalence.

All offline: synthetic binary + continuous tasks stand in for the rungs (no Hub access), and the
continuous rung-4/5 analogue (a CONTINUOUS regression task) exercises the differentiable path that
my earlier mistake assumed was a separate "interactive" regime.
"""

import math
import random
from typing import Any, cast

import torch

from ardevo import checkpoint
from ardevo.dataset.icarus import Axis, Field, Task, TaskKind, TaskMeta, ValueType
from ardevo.evolution import multitask
from ardevo.evolution.evolver import EvolverState
from ardevo.evolution.genome import InnovationTracker, genome_from_dict
from ardevo.evolution.multitask import MultiTaskSubstrate, task_entry
from ardevo.evolution.registry import build_evolver
from ardevo.evolution.schedule import build_schedule


def _binary_field(values: list[float]) -> Field:
    return Field(torch.tensor(values, dtype=torch.float32), (Axis.EXTRA,), ValueType.BINARY, None, None, None)


def _continuous_field(values: list[float]) -> Field:
    return Field(torch.tensor(values, dtype=torch.float32), (Axis.EXTRA,), ValueType.CONTINUOUS, None, (0.0, 1.0), None)


def _binary_task(name: str, width: int, rung: int = 2) -> Task:
    rows = [[float((index >> bit) & 1) for bit in range(width)] for index in range(4)]
    pairs = [(_binary_field(row), _binary_field([float(sum(row) % 2)])) for row in rows]
    return Task(meta=TaskMeta(rung=rung, kind=TaskKind.MAP, name=name, fixed_split=True), support=list(pairs), query=list(pairs))


def _continuous_task(name: str, dim: int, rung: int = 4) -> Task:
    rows = [[0.1 * (index + 1)] * dim for index in range(4)]
    pairs = [(_continuous_field(row), _continuous_field([0.2 * (index + 1)])) for index, row in enumerate(rows)]
    return Task(meta=TaskMeta(rung=rung, kind=TaskKind.MAP, name=name, fixed_split=True), support=list(pairs), query=list(pairs))


def _config() -> dict[str, Any]:
    return {
        "seed": 0,
        "substrate": {"available_activations": ["tanh", "identity"], "default_activation": "tanh"},
        "evolution": {
            "pop_size": 6,
            "elitism": 1,
            "init": {"kind": "minimal"},
            "selection": {"kind": "tournament", "tournament_size": 2},
            "crossover": {"kind": "neat", "rate": 0.2},
            "mutation": {"operators": ["add_rich_node", "add_connection"], "add_rich_node_prob": 0.3, "add_rich_node_fan_in": 2, "add_connection_prob": 0.3},
            "train": {"kind": "gradient", "steps": 5, "lr": 0.05, "writeback": True},
            "speciation": {"kind": "neat", "threshold": 1.5, "target_species": 3},
        },
        "fitness": {"components": ["support_accuracy", "complexity_penalty"], "w_support_accuracy": 1.0, "w_complexity_penalty": 0.01},
    }


def test_task_entry_describes_interface() -> None:
    entry = task_entry(_binary_task("parity3", 3))
    assert entry.input_signature.startswith("BINARY")
    assert entry.input_width == 3
    assert entry.output_width == 1


def test_build_pool_closes_backend_datasets(monkeypatch) -> None:
    closed: list[int] = []

    class FakeIcarusDataset:
        def __init__(self, *, rungs, **_kwargs) -> None:
            self.rung = int(rungs[0])
            self.tasks = [_binary_task(f"task{self.rung}", 2, rung=self.rung)]

        def __len__(self) -> int:
            return len(self.tasks)

        def __getitem__(self, index: int) -> Task:
            return self.tasks[index]

        def close(self) -> None:
            closed.append(self.rung)

    monkeypatch.setattr(multitask, "IcarusDataset", FakeIcarusDataset)

    pool = multitask.build_pool("unused", [1, 2], n_samples=4, support_fraction=0.8, tasks_per_rung=1, shuffle=False, seed=0)

    assert [entry.rung for entry in pool] == [1, 2]
    assert closed == [1, 2]


def test_seed_builds_one_bank_and_head() -> None:
    substrate = MultiTaskSubstrate(default_activation="tanh")
    entry = task_entry(_binary_task("parity3", 3))
    genomes = substrate.seed(entry, InnovationTracker(_next_node_id=0), random.Random(0), pop_size=4, weight_scale=1.0)

    assert substrate.n_inputs == 3
    assert substrate.n_outputs == 1
    assert len(genomes) == 4
    module = substrate.adapter(entry).decode(genomes[0])
    output = module(substrate.adapter(entry).encoded.support_input[0])
    assert output.shape[1] == 1  # sliced to this task's head width


def test_expand_widens_a_bank_in_place() -> None:
    substrate = MultiTaskSubstrate(default_activation="tanh")
    tracker = InnovationTracker(_next_node_id=0)
    rng = random.Random(0)
    narrow = task_entry(_binary_task("parity3", 3))
    genomes = substrate.seed(narrow, tracker, rng, pop_size=4, weight_scale=1.0)

    wide = task_entry(_binary_task("parity6", 6))
    grown = substrate.expand(wide, genomes, tracker, rng)

    assert substrate.n_inputs == 6  # same signature -> the binary bank widened 3 -> 6
    assert substrate.n_outputs == 2  # a new head for the new task name
    assert len(grown[0].input_ids) == 6
    module = substrate.adapter(wide).decode(grown[0])
    assert module(substrate.adapter(wide).encoded.support_input[0]).shape[1] == 1


def test_expand_grows_a_new_bank_and_head_for_a_new_type() -> None:
    substrate = MultiTaskSubstrate(default_activation="tanh")
    tracker = InnovationTracker(_next_node_id=0)
    rng = random.Random(0)
    binary = task_entry(_binary_task("parity3", 3))
    genomes = substrate.seed(binary, tracker, rng, pop_size=4, weight_scale=1.0)

    continuous = task_entry(_continuous_task("pole4", 4))
    grown = substrate.expand(continuous, genomes, tracker, rng)

    assert substrate.n_inputs == 3 + 4  # binary bank + a new continuous bank
    assert substrate.n_outputs == 2  # two disjoint heads
    assert len(substrate.banks) == 2
    module = substrate.adapter(continuous).decode(grown[0])
    assert module(substrate.adapter(continuous).encoded.support_input[0]).shape[1] == continuous.output_width


def test_continuous_task_scores_through_the_same_path() -> None:
    """Rungs 4-5 are continuous regression; confirm the differentiable evaluate path handles them."""
    substrate = MultiTaskSubstrate(default_activation="tanh")
    entry = task_entry(_continuous_task("pole4", 4))
    genomes = substrate.seed(entry, InnovationTracker(_next_node_id=0), random.Random(0), pop_size=4, weight_scale=1.0)
    evolver = build_evolver(_config())
    assessed = evolver.evaluate_only(genomes[0], substrate.adapter(entry))
    assert math.isfinite(assessed.metrics["support_loss"])
    assert assessed.metrics["query_accuracy"] >= 0.0


def test_distinct_heads_select_distinct_columns() -> None:
    substrate = MultiTaskSubstrate(default_activation="tanh")
    tracker = InnovationTracker(_next_node_id=0)
    rng = random.Random(0)
    first = task_entry(_binary_task("a", 3))
    substrate.seed(first, tracker, rng, pop_size=2, weight_scale=1.0)
    second = task_entry(_continuous_task("b", 2))
    substrate.expand(second, [], tracker, rng)

    columns_a = substrate.adapter(first).head_columns.tolist()
    columns_b = substrate.adapter(second).head_columns.tolist()
    assert set(columns_a).isdisjoint(set(columns_b))


def test_checkpoint_resume_is_equivalent() -> None:
    """Stepping, serializing, reconstructing, and stepping again must match an uninterrupted step."""
    config = _config()
    substrate = MultiTaskSubstrate(default_activation="tanh")
    tracker = InnovationTracker(_next_node_id=0)
    rng = random.Random(0)
    entry = task_entry(_binary_task("parity3", 3))
    evolver = build_evolver(config)
    scheduler = build_schedule({"kind": "round_robin"})

    genomes = substrate.seed(entry, tracker, rng, pop_size=config["evolution"]["pop_size"], weight_scale=1.0)
    state = EvolverState(population=[], innovations=tracker, rng=rng)
    adapter = substrate.adapter(entry)
    state.population = [evolver.assess(genome, adapter, state) for genome in genomes]
    evolver.advance(state, adapter)
    evolver.advance(state, adapter)

    payload = checkpoint.build_payload(state=state, speciator=evolver.speciate, scheduler=scheduler, substrate=substrate, active_index=0)

    # Straight: one more generation on the live state.
    evolver.advance(state, adapter)
    straight = sorted(item.fitness for item in state.population)

    # Resume: rebuild everything from the payload and take the same step.
    evolver2 = build_evolver(config)
    cast(Any, evolver2.speciate).load_state_dict(payload["speciation"])
    substrate2 = MultiTaskSubstrate.from_dict(payload["substrate"], "tanh")
    tracker2 = InnovationTracker.from_dict(payload["innovations"])
    rng2 = checkpoint.deserialize_rng(payload["rng"])
    adapter2 = substrate2.adapter(entry)
    state2 = EvolverState(population=[], innovations=tracker2, rng=rng2, generation=payload["generation"], species_history=checkpoint.restored_species_history(payload))
    state2.population = [evolver2.evaluate_only(genome_from_dict(genome), adapter2) for genome in payload["population"]]
    evolver2.advance(state2, adapter2)
    resumed = sorted(item.fitness for item in state2.population)

    assert len(straight) == len(resumed)
    for left, right in zip(straight, resumed):
        assert left == right or abs(left - right) < 1e-4
