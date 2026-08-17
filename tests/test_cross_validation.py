import math
from dataclasses import replace

import torch

from versal.cross_validation import CrossValidationConfig, SupportCrossValidator, fresh_composition_glue, fresh_genome_weights
from versal.dataset.icarus import Axis, Field, Task, TaskKind, TaskMeta, ValueType
from versal.evolution.composition import CompEdgeGene, CompNodeGene, CompNodeKind, CompositionGenome, IndexRun, PortMap
from versal.evolution.genome import ConnectionGene, Genome, NodeGene, NodeKind


def _field(value: int) -> Field:
    return Field(torch.tensor([value]), (Axis.EXTRA,), ValueType.CATEGORICAL, 8, None, None)


def _task(values: list[int], *, query_value: int = 7) -> Task:
    pairs = [(_field(value), _field(value)) for value in values]
    return Task(TaskMeta(0, TaskKind.MAP, "cv.synthetic", fixed_split=True), pairs, [(_field(query_value), _field(query_value))])


def _config(**overrides: object) -> CrossValidationConfig:
    return replace(CrossValidationConfig(enabled=True, max_folds=5, min_folds=2, pass_fraction=0.5), **overrides)


def test_support_folds_use_strict_majority_and_never_use_real_query() -> None:
    seen: list[int] = []

    def evaluate(fold: Task, _seed: int, _deadline: float | None) -> dict[str, float]:
        omitted = int(fold.query[0][1].data.item())
        seen.append(omitted)
        return {"query_accuracy": float(omitted < 2), "query_loss": 0.0}

    validator = SupportCrossValidator(_config(), accept_threshold=0.95)
    first = validator.run(_task([0, 1, 2], query_value=6), evaluate, deadline=None)
    first_seen = list(seen)
    seen.clear()
    second = validator.run(_task([0, 1, 2], query_value=7), evaluate, deadline=None)

    assert first.status == second.status == "passed"
    assert first.folds_passed == 2
    assert first_seen == seen == [0, 1, 2]


def test_structured_fold_exactness_takes_precedence_over_dense_accuracy() -> None:
    validator = SupportCrossValidator(_config(), accept_threshold=0.95)
    result = validator.run(
        _task([0, 1, 2]),
        lambda *_args: {"query_accuracy": 1.0, "query_task_exact": 0.0, "query_loss": 0.0},
        deadline=None,
    )
    assert result.status == "failed"
    assert result.folds_passed == 0


def test_validation_is_not_applicable_with_one_support_example() -> None:
    result = SupportCrossValidator(_config(), accept_threshold=0.95).run(_task([0]), lambda *_args: {}, deadline=None)
    assert result.status == "not_applicable"
    assert result.admits


def test_expired_validation_is_inconclusive() -> None:
    result = SupportCrossValidator(_config(), accept_threshold=0.95).run(_task([0, 1]), lambda *_args: {}, deadline=0.0)
    assert result.status == "inconclusive"
    assert not result.admits


def test_fresh_genome_weights_are_deterministic_preserve_ties_and_do_not_mutate_source() -> None:
    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity"),
        1: NodeGene(1, NodeKind.INPUT, "identity"),
        2: NodeGene(2, NodeKind.OUTPUT, "identity"),
    }
    genome = Genome(
        nodes,
        [
            ConnectionGene(0, 2, 9.0, True, 0, tie_group=4),
            ConnectionGene(1, 2, -9.0, True, 1, tie_group=4),
        ],
    )
    first = fresh_genome_weights(genome, 12)
    second = fresh_genome_weights(genome, 12)

    assert genome.connections[0].weight == 9.0
    assert first.connections == second.connections
    assert first.connections[0].weight == first.connections[1].weight
    assert not math.isclose(first.connections[0].weight, 9.0)


def test_fresh_composition_resets_only_trainable_glue() -> None:
    nodes = {
        0: CompNodeGene(0, CompNodeKind.INPUT, "x", 0, 2),
        1: CompNodeGene(1, CompNodeKind.OUTPUT, "y", 2, 0),
    }
    fixed = PortMap((IndexRun(0, 0, 1),))
    comp = CompositionGenome(
        nodes,
        [
            CompEdgeGene(0, 1, True, 0, (9.0, 9.0, 9.0, 9.0)),
            CompEdgeGene(0, 1, True, 1, (), port_map=fixed),
        ],
    )
    fresh = fresh_composition_glue(comp, 3)

    assert comp.edges[0].glue == (9.0, 9.0, 9.0, 9.0)
    assert fresh.edges[0].glue != comp.edges[0].glue
    assert fresh.edges[1] == comp.edges[1]
