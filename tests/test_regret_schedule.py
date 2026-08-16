"""Phase-5: the regret scheduler (ACCEL-style frontier prioritization over rungs, fed by the
trial's observe hook) and the Phase-4 contractivity penalty on the refine trainer."""

import random
from typing import cast

from versal.dataset.icarus import Task
from versal.evolution.multitask import TaskEntry
from versal.evolution.schedule import RegretSchedule, build_schedule


def _pool() -> list[TaskEntry]:
    # Schedulers only read rung/name; the structural fields are stubbed.
    return [
        TaskEntry(rung=rung, name=f"r{rung}t{index}", task=cast(Task, None), input_signature="BINARY|E", input_axes=("E",), input_shape=(2,), input_width=2, output_width=1)
        for rung in (1, 2, 3)
        for index in range(2)
    ]


def _greedy(schedule: RegretSchedule) -> RegretSchedule:
    schedule.explore_fraction = 0.0  # deterministic: always the argmax rung
    return schedule


def test_unseen_rungs_are_attempted_first() -> None:
    schedule = _greedy(RegretSchedule())
    pool = _pool()
    rng = random.Random(0)
    first = pool[schedule.next_index(pool, rng)].rung
    schedule.observe(first, 0.5, False)
    second = pool[schedule.next_index(pool, rng)].rung
    schedule.observe(second, 0.5, False)
    third = pool[schedule.next_index(pool, rng)].rung
    assert {first, second, third} == {1, 2, 3}  # every rung gets first contact before any repeat


def test_highest_regret_rung_wins_and_solved_rungs_decay() -> None:
    schedule = _greedy(RegretSchedule())
    pool = _pool()
    rng = random.Random(0)
    for rung, metric, solved in ((1, 1.0, True), (2, 0.6, False), (3, 0.85, False)):
        schedule.observe(rung, metric, solved)
    assert pool[schedule.next_index(pool, rng)].rung == 2  # biggest gap to the bar
    schedule.observe(2, 0.94, False)
    assert pool[schedule.next_index(pool, rng)].rung == 3  # frontier moved (0.10 regret beats 0.05 everywhere else)


def test_hard_floor_halves_no_signal_walls() -> None:
    schedule = _greedy(RegretSchedule(hard_floor=0.1))
    pool = _pool()
    rng = random.Random(0)
    schedule.observe(1, 0.05, False)  # a wall with no signal: regret 0.9 halves to 0.45
    schedule.observe(2, 0.4, False)
    schedule.observe(3, 0.94, False)
    assert pool[schedule.next_index(pool, rng)].rung == 2  # 0.55 regret beats the halved 0.45 wall score


def test_state_round_trips_and_registry_builds_it() -> None:
    schedule = build_schedule({"kind": "regret", "explore_fraction": 0.0, "accept_threshold": 0.9})
    assert isinstance(schedule, RegretSchedule) and schedule.accept_threshold == 0.9
    schedule.observe(4, 0.7, False)
    schedule.next_index(_pool(), random.Random(1))
    restored = RegretSchedule()
    restored.load_state_dict(schedule.state_dict())
    assert restored.best_metric == schedule.best_metric and restored.task_cursors == schedule.task_cursors


def test_within_rung_tasks_advance_by_cursor() -> None:
    schedule = _greedy(RegretSchedule())
    pool = _pool()
    rng = random.Random(0)
    schedule.observe(1, 0.5, False)
    schedule.observe(2, 0.94, False)
    schedule.observe(3, 0.94, False)
    picks = [schedule.next_index(pool, rng) for _ in range(2)]
    assert [pool[index].name for index in picks] == ["r1t0", "r1t1"]


def test_contractivity_penalty_shrinks_recurrent_norm(xor_adapter) -> None:
    import torch

    from versal.evolution.train import gradient_refine
    from versal.substrate import decode_refine

    def refine_genome():
        from versal.evolution.genome import ConnectionGene, Genome, NodeGene, NodeKind

        nodes = {
            0: NodeGene(0, NodeKind.INPUT, "identity"),
            1: NodeGene(1, NodeKind.INPUT, "identity"),
            2: NodeGene(2, NodeKind.BIAS, "identity"),
            3: NodeGene(3, NodeKind.HIDDEN, "tanh"),
            4: NodeGene(4, NodeKind.OUTPUT, "identity"),
        }
        connections = [
            ConnectionGene(0, 3, 1.0, True, 0),
            ConnectionGene(1, 3, 1.0, True, 1),
            ConnectionGene(3, 4, 1.0, True, 2),
            ConnectionGene(3, 3, 3.0, True, 3, recurrent=True),  # far from contractive
        ]
        genome = Genome(nodes=nodes, connections=connections)
        genome.refine_steps = 3
        return genome

    unconstrained, _module = gradient_refine(refine_genome(), decode_refine(refine_genome(), 2, 1), xor_adapter.encoded, rng=random.Random(0), steps=8, lr=0.05)
    constrained, _module = gradient_refine(
        refine_genome(), decode_refine(refine_genome(), 2, 1), xor_adapter.encoded, rng=random.Random(0), steps=8, lr=0.05, contractivity_weight=1.0
    )

    def recurrent_norm(genome) -> float:
        return sum(conn.weight**2 for conn in genome.connections if conn.recurrent) ** 0.5

    assert recurrent_norm(constrained) < recurrent_norm(unconstrained)
    baseline, _module = gradient_refine(
        refine_genome(), decode_refine(refine_genome(), 2, 1), xor_adapter.encoded, rng=random.Random(0), steps=8, lr=0.05, contractivity_weight=0.0
    )
    assert [conn.weight for conn in baseline.connections] == [conn.weight for conn in unconstrained.connections]  # 0.0 = byte-identical
    assert torch is not None
