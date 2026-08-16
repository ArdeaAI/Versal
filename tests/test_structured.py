import math
from typing import cast

import torch

from versal.dataset.icarus import Axis, Field, Level0Encoder, Task, TaskKind, TaskMeta, ValueType
from versal.evaluation import input_width, output_features
from versal.evolution.evolver import Evolver, TaskAdapter
from versal.evolution.loop import CompTaskSpec
from versal.orchestrator import Orchestrator
from versal.strategy import DirectStrategy, StrategyResult, StrategyRuntime
from versal.structured import ShapeRule, encode_structured_grid, evaluate_structured_grid, fit_shape_program


def _field(height: int, width: int, value: int = 0) -> Field:
    return Field(
        torch.full((height, width), value, dtype=torch.long),
        (Axis.HEIGHT, Axis.WIDTH),
        ValueType.CATEGORICAL,
        3,
        None,
        None,
    )


def _grid(values: list[list[int]]) -> Field:
    return Field(torch.tensor(values, dtype=torch.long), (Axis.HEIGHT, Axis.WIDTH), ValueType.CATEGORICAL, 3, None, None)


def _swap_shape_task(*, query_output: tuple[int, int] = (4, 2)) -> Task:
    support = [
        ((_field(2, 3)), _field(3, 2)),
        ((_field(3, 4)), _field(4, 3)),
    ]
    query = [((_field(2, 4)), _field(*query_output))]
    return Task(TaskMeta(18, TaskKind.MAP, "arc.synthetic", fixed_split=True), support, query)


class _ZeroGrid(torch.nn.Module):
    def __init__(self, outputs: int) -> None:
        super().__init__()
        self.outputs = outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.full((x.shape[0], self.outputs), -10.0)
        logits[:, 0::3] = 10.0
        return logits


def test_shape_program_finds_swap_without_query_labels() -> None:
    program = fit_shape_program(((2, 3), (3, 4)), ((3, 2), (4, 3)), max_height=30, max_width=30)
    assert program.height == ShapeRule(0, 1, 0)
    assert program.width == ShapeRule(1, 0, 0)
    assert program.predict((5, 7)) == (7, 5)


def test_structured_grid_reports_exact_shape_cells_and_baselines() -> None:
    task = _swap_shape_task()
    encoder = Level0Encoder(12)
    encoded = encode_structured_grid(task, encoder)
    assert encoded is not None
    module = _ZeroGrid(output_features(encoded))
    metrics = evaluate_structured_grid(module, encoded, encoder)
    assert metrics["support_task_exact"] == 1.0
    assert metrics["query_task_exact"] == 1.0
    assert metrics["query_shape_accuracy"] == 1.0
    assert metrics["query_baseline_accuracy"] == 1.0
    assert metrics["query_gain_over_baseline"] == 0.0


def test_exact_grid_rejects_correct_cells_at_wrong_predicted_shape() -> None:
    task = _swap_shape_task(query_output=(3, 2))
    encoder = Level0Encoder(12)
    encoded = encode_structured_grid(task, encoder)
    assert encoded is not None
    metrics = evaluate_structured_grid(_ZeroGrid(output_features(encoded)), encoded, encoder)
    assert metrics["query_accuracy"] == 1.0
    assert metrics["query_cell_exact"] == 1.0
    assert metrics["query_shape_accuracy"] == 0.0
    assert metrics["query_task_exact"] == 0.0


def test_structured_search_view_cannot_evaluate_query() -> None:
    task = _swap_shape_task()
    encoder = Level0Encoder(12)
    encoded = encode_structured_grid(task, encoder)
    assert encoded is not None
    blind = encoded.without_query()
    metrics = evaluate_structured_grid(_ZeroGrid(output_features(blind)), blind, encoder)
    assert metrics["support_task_exact"] == 1.0
    assert metrics["query_accuracy"] == 0.0
    assert math.isinf(metrics["query_loss"])


def test_blind_structured_encoding_does_not_inspect_query_target() -> None:
    task = _swap_shape_task()
    malformed_target = Field(torch.tensor([1.0]), (Axis.EXTRA,), ValueType.CONTINUOUS, None, (0.0, 1.0), None)
    changed = Task(task.meta, task.support, [(task.query[0][0], malformed_target)])
    encoder = Level0Encoder(12)

    original = encode_structured_grid(task, encoder, include_query=False)
    altered = encode_structured_grid(changed, encoder, include_query=False)

    assert original is not None and altered is not None
    assert original.shape_program == altered.shape_program
    assert torch.equal(original.support_input[0], altered.support_input[0])
    assert original.query_target is None and altered.query_target is None


def test_structured_query_uses_support_spatial_canvas() -> None:
    support_input = _grid([[0, 1, 2], [2, 1, 0]])
    query_input = _grid([[1, 2], [0, 1]])
    task = Task(TaskMeta(18, TaskKind.MAP, "arc.canvas", fixed_split=True), [(support_input, support_input)], [(query_input, query_input)])
    encoded = encode_structured_grid(task, Level0Encoder(6))

    assert encoded is not None and encoded.query_input is not None and encoded.query_target is not None
    assert encoded.query_input[0].tolist() == [[1.0, 2.0, 0.0, 0.0, 1.0, 0.0]]
    target, mask, _descriptor = encoded.query_target
    assert mask is not None
    assert target.tolist() == [[1, 2, 0, 0, 1, 0]]
    assert mask.tolist() == [[False, False, True, False, False, True]]


def test_larger_query_grid_counts_uncovered_cells_as_incorrect() -> None:
    support = _field(2, 2)
    query = _field(3, 3)
    task = Task(TaskMeta(18, TaskKind.MAP, "arc.coverage", fixed_split=True), [(support, support)], [(query, query)])
    encoded = encode_structured_grid(task, Level0Encoder(4))

    assert encoded is not None
    metrics = evaluate_structured_grid(_ZeroGrid(output_features(encoded)), encoded, Level0Encoder(4))
    assert metrics["query_covered_accuracy"] == 1.0
    assert math.isclose(metrics["query_coverage"], 4 / 9, rel_tol=1e-6)
    assert math.isclose(metrics["query_accuracy"], 4 / 9, rel_tol=1e-6)
    assert metrics["query_covered_cell_exact"] == 1.0
    assert metrics["query_cell_exact"] == 0.0
    assert metrics["query_task_exact"] == 0.0


def test_direct_structured_adapter_preserves_head_width_and_blinds_query() -> None:
    strategy = DirectStrategy(evolver=cast(Evolver, None), structured_grid=True, blind_query=True)
    blind = strategy._adapter(_swap_shape_task(), include_query=False)
    full = strategy._adapter(_swap_shape_task(), include_query=True)
    assert isinstance(blind, TaskAdapter) and isinstance(full, TaskAdapter)
    assert blind.n_inputs == full.n_inputs == input_width(full.encoded)
    assert blind.n_outputs == full.n_outputs == output_features(full.encoded)
    assert blind.encoded.query_input is None and full.encoded.query_input is not None


def test_direct_width_guard_counts_categorical_logits() -> None:
    strategy = DirectStrategy(evolver=cast(Evolver, None), max_flat_outputs=20)
    result = strategy(_swap_shape_task(), cast(CompTaskSpec, None), cast(StrategyRuntime, None), budget=1)
    assert result.champion_metrics["declined_flat_width"] == 36.0  # 4x3 support canvas, three classes


def test_task_appropriate_metric_prefers_exact_only_when_present() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator.search_metric = "support_task_appropriate"
    exact = type("Item", (), {"metrics": {"support_accuracy": 1.0, "support_task_exact": 0.0}})()
    ordinary = type("Item", (), {"metrics": {"support_accuracy": 0.75}})()
    assert orchestrator._metric(exact) == 0.0
    assert orchestrator._metric(ordinary) == 0.75


def test_dense_search_progress_does_not_weaken_exact_admission() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator.search_metric = "support_accuracy"
    orchestrator.accept_metric = "support_task_appropriate"
    orchestrator.accept_threshold = 0.95
    metrics = {"support_accuracy": 0.99, "support_task_exact": 0.0}
    item = type("Item", (), {"metrics": metrics})()
    result = StrategyResult("direct", metric=0.99, generations_used=1, champion_metrics=metrics)

    assert orchestrator._metric(item) == 0.99
    assert not orchestrator._accepts_result(result)
