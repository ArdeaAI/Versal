"""Pool reporting + rung_doctor: silent coverage gaps must become visible rows, offline-testable."""

import torch

from versal.dataset.icarus import Axis, Field, Task, TaskKind, TaskMeta, ValueType
from versal.evolution.genome import genome_to_dict
from versal.evolution.multitask import build_pool_report
from versal.library import MODULE, ModuleLibrary, task_io
from versal.tools.rung_doctor import parse_rungs, rung_report


def _binary_task(rung: int) -> Task:
    pairs = []
    for a in (0.0, 1.0):
        for b in (0.0, 1.0):
            x = Field(torch.tensor([a, b]), (Axis.EXTRA,), ValueType.BINARY, None, None, None)
            y = Field(torch.tensor([float(a != b)]), (Axis.EXTRA,), ValueType.BINARY, None, None, None)
            pairs.append((x, y))
    return Task(meta=TaskMeta(rung=rung, kind=TaskKind.MAP, name=f"fake_r{rung}"), support=pairs, query=pairs)


class _FakeDataset:
    def __init__(self, tasks: list[Task]) -> None:
        self._tasks = tasks

    def __len__(self) -> int:
        return len(self._tasks)

    def __getitem__(self, index: int) -> Task:
        return self._tasks[index]

    def close(self) -> None:
        return None


def _factory(*, rungs, **_kwargs):
    rung = rungs[0]
    if rung == 2:
        raise RuntimeError("simulated arrow overflow")
    if rung == 3:
        return _FakeDataset([])
    return _FakeDataset([_binary_task(rung)])


def test_build_pool_report_records_skips_and_empties() -> None:
    report = build_pool_report("fake", [1, 2, 3], n_samples=4, support_fraction=0.8, tasks_per_rung=2, shuffle=False, seed=0, dataset_factory=_factory)
    assert [entry.rung for entry in report.entries] == [1]
    assert [(s.rung, s.error_type) for s in report.skipped] == [(2, "RuntimeError"), (3, "EmptyRung")]
    assert "arrow overflow" in report.skipped[0].message


def test_rung_report_rows(tmp_path) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    task = _binary_task(1)
    library.add(entry_type=MODULE, payload=genome_to_dict(_solving()), io=task_io(task), provenance={"accepted_metric": 1.0})
    rows = rung_report([1, 2, 3], library=library, dataset_factory=_factory)
    by_rung = {row["rung"]: row for row in rows}
    assert by_rung[1]["status"] == "OK" and by_rung[1]["input"] == "BINARY|K w=2" and by_rung[1]["output"].endswith("w=1")
    assert by_rung[1]["temporal"] is False
    assert by_rung[1]["library"] == "exact=1 near=1"
    assert by_rung[2]["status"] == "FAIL:RuntimeError"
    assert by_rung[3]["status"] == "FAIL:EmptyRung"


def _solving():
    from versal.evolution.genome import ConnectionGene, Genome, NodeGene, NodeKind

    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity"),
        1: NodeGene(1, NodeKind.INPUT, "identity"),
        2: NodeGene(2, NodeKind.BIAS, "identity"),
        3: NodeGene(3, NodeKind.OUTPUT, "identity"),
    }
    return Genome(nodes=nodes, connections=[ConnectionGene(0, 3, 1.0, True, 0)])


def test_parse_rungs() -> None:
    assert parse_rungs("1-3") == [1, 2, 3]
    assert parse_rungs("1,5,7-9") == [1, 5, 7, 8, 9]
