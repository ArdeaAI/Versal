"""Multi-task substrate: grow ONE topology across many Icarus tasks without pre-allocation.

A continuous run interleaves tasks from several rungs and keeps one population alive across the
switches. Tasks differ in I/O width, so the interface GROWS instead of being pre-sized:

- Inputs live in descriptor-keyed BANKS (one per value_type + axes signature). A bank widens on
  demand to the widest task of its type; narrower tasks feed zeros into the surplus columns. Input
  nodes are never overloaded with semantically different values (a bit and a coordinate never share
  a node). Each input node is stamped with its raw unraveled axis-index as a `coordinate`, which the
  geometry-biased mutation operators use to grow local receptive fields.
- Each distinct task name owns a disjoint output HEAD, grown on first encounter and sized by
  `model_output_features`. The active task is scored on its own head via a thin `HeadSlicedNet`
  wrapper, so the shared `GraphNet` decode and the differentiable evaluate/train path never change.

`expand_interface` (here, `MultiTaskSubstrate.expand`) applies a task's interface growth to the WHOLE
population deterministically, allocating shared node ids / innovations so genes stay aligned for
crossover. The layout (banks, heads, bias) serializes for checkpoint/resume.
"""

import random
from dataclasses import dataclass, field
from typing import Any

import torch

from ardevo.dataset.icarus import EncodedTask, IcarusDataset, Level0Encoder, Task, encode_task, support_loader
from ardevo.evaluation import evaluate, output_features
from ardevo.evolution.genome import ConnectionGene, Genome, InnovationTracker, NodeGene, NodeKind
from ardevo.substrate import GraphNet, SubstrateModule, decode


@dataclass(frozen=True)
class TaskEntry:
    """One schedulable task plus the structural facts needed to grow its interface."""

    rung: int
    name: str
    task: Task
    input_signature: str
    input_axes: tuple[str, ...]
    input_shape: tuple[int, ...]
    input_width: int
    output_width: int


def task_entry(task: Task) -> TaskEntry:
    """Derive a `TaskEntry` (signature, shapes, head width) from a raw Icarus task."""
    support_input, _support_output = support_loader(task)
    input_shape = tuple(int(dim) for dim in support_input.data.shape[1:])
    input_axes = tuple(axis.value for axis in support_input.descriptor.axes)
    input_width = 1
    for dim in input_shape:
        input_width *= dim
    signature = f"{support_input.descriptor.value_type.value}|{','.join(input_axes)}"
    encoded = encode_task(task, Level0Encoder(input_width))
    return TaskEntry(
        rung=task.meta.rung,
        name=task.meta.name,
        task=task,
        input_signature=signature,
        input_axes=input_axes,
        input_shape=input_shape,
        input_width=input_width,
        output_width=output_features(encoded),
    )


def build_pool(source: str, rungs: list[int], n_samples: int, support_fraction: float, tasks_per_rung: int, shuffle: bool, seed: int) -> list[TaskEntry]:
    """Load every task across the configured rungs (via `IcarusDataset`) as schedulable entries.

    `source` is passed as `hf_repo` (hyphen form) per the loader's note. Rungs absent from the repo
    are silently skipped by `IcarusDataset`, so the pool spans only what is actually available.
    """
    dataset = IcarusDataset(
        rungs=tuple(rungs),
        n_tasks=tasks_per_rung,
        n_samples=n_samples,
        support_fraction=support_fraction,
        shuffle_within=shuffle,
        seed=seed,
        hf_repo=source,
    )
    return [task_entry(dataset[index]) for index in range(len(dataset))]


def _coordinates(shape: tuple[int, ...]) -> list[tuple[float, ...]]:
    """Row-major unraveled indices over `shape` as float coordinates (raw, so they are stable as a
    bank widens). For a 1-D field this is just (0.,), (1.,), ...; for an image it is (h, w, c)."""
    total = 1
    for dim in shape:
        total *= dim
    coordinates: list[tuple[float, ...]] = []
    for flat in range(total):
        index: list[float] = []
        remainder = flat
        for dim in reversed(shape):
            index.append(float(remainder % dim))
            remainder //= dim
        coordinates.append(tuple(reversed(index)))
    return coordinates


@dataclass
class _Bank:
    signature: str
    axes: tuple[str, ...]
    node_ids: list[int]
    coordinates: list[tuple[float, ...]]
    width: int


@dataclass
class _Head:
    name: str
    node_ids: list[int]


class HeadSlicedNet(SubstrateModule):
    """Wraps a full `GraphNet` to expose only one task's output-head columns.

    The inner net is a normal submodule, so its weights are shared and trainable; this wrapper only
    selects the active head's output columns, which is what lets evaluate/train operate unchanged.
    """

    def __init__(self, inner: GraphNet, head_columns: torch.Tensor) -> None:
        super().__init__()
        self.inner = inner
        self.head_columns: torch.Tensor = head_columns

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x).index_select(1, self.head_columns)

    @property
    def has_edges(self) -> bool:
        return self.inner.has_edges

    def export_weights(self) -> dict[tuple[int, int], float]:
        return self.inner.export_weights()


@dataclass
class MultiTaskAdapter:
    """An evolver `Adapter` for one active task: full-width encoded input + this task's output head."""

    encoded: EncodedTask
    encoder: Level0Encoder
    n_inputs: int
    n_outputs: int
    head_columns: torch.Tensor

    def decode(self, genome: Genome) -> SubstrateModule:
        return HeadSlicedNet(decode(genome, self.n_inputs, self.n_outputs), self.head_columns)

    def evaluate(self, module: SubstrateModule) -> dict[str, float]:
        return evaluate(module, self.encoded, self.encoder)


@dataclass
class MultiTaskSubstrate:
    """The growing-interface layout shared by the whole population: input banks, output heads, bias."""

    default_activation: str
    bias_id: int = -1
    banks: dict[str, _Bank] = field(default_factory=dict)
    heads: dict[str, _Head] = field(default_factory=dict)

    @property
    def n_inputs(self) -> int:
        return sum(bank.width for bank in self.banks.values())

    @property
    def n_outputs(self) -> int:
        return sum(len(head.node_ids) for head in self.heads.values())

    def _all_input_ids(self) -> list[int]:
        return sorted(node_id for bank in self.banks.values() for node_id in bank.node_ids)

    def _all_output_ids(self) -> list[int]:
        return sorted(node_id for head in self.heads.values() for node_id in head.node_ids)

    def _grow_bank(self, entry: TaskEntry, tracker: InnovationTracker) -> list[tuple[int, tuple[float, ...]]]:
        """Ensure the entry's bank exists and is wide enough; return the (id, coordinate) pairs added."""
        bank = self.banks.get(entry.input_signature)
        if bank is None:
            bank = _Bank(signature=entry.input_signature, axes=entry.input_axes, node_ids=[], coordinates=[], width=0)
            self.banks[entry.input_signature] = bank
        coordinates = _coordinates(entry.input_shape)
        added: list[tuple[int, tuple[float, ...]]] = []
        for index in range(bank.width, entry.input_width):
            node_id = tracker.new_node_id()
            coordinate = coordinates[index]
            bank.node_ids.append(node_id)
            bank.coordinates.append(coordinate)
            added.append((node_id, coordinate))
        bank.width = max(bank.width, entry.input_width)
        return added

    def _grow_head(self, entry: TaskEntry, tracker: InnovationTracker) -> list[int]:
        """Ensure the entry's output head exists; return the output-node ids added (empty if present)."""
        if entry.name in self.heads:
            return []
        ids = [tracker.new_node_id() for _ in range(entry.output_width)]
        self.heads[entry.name] = _Head(name=entry.name, node_ids=ids)
        return ids

    def seed(self, entry: TaskEntry, tracker: InnovationTracker, rng: random.Random, pop_size: int, weight_scale: float) -> list[Genome]:
        """Register the first task's bank + head and return a fresh minimal population for it."""
        self._grow_bank(entry, tracker)
        self.bias_id = tracker.new_node_id()
        self._grow_head(entry, tracker)
        return [self._minimal_genome(tracker, rng, weight_scale) for _ in range(pop_size)]

    def _minimal_genome(self, tracker: InnovationTracker, rng: random.Random, weight_scale: float) -> Genome:
        nodes: dict[int, NodeGene] = {}
        for bank in self.banks.values():
            for node_id, coordinate in zip(bank.node_ids, bank.coordinates):
                nodes[node_id] = NodeGene(node_id, NodeKind.INPUT, "identity", coordinate)
        nodes[self.bias_id] = NodeGene(self.bias_id, NodeKind.BIAS, "identity", None)
        output_ids = self._all_output_ids()
        for output_id in output_ids:
            nodes[output_id] = NodeGene(output_id, NodeKind.OUTPUT, "identity", None)
        connections: list[ConnectionGene] = []
        for source in [*self._all_input_ids(), self.bias_id]:
            for output_id in output_ids:
                connections.append(ConnectionGene(source, output_id, rng.gauss(0.0, weight_scale), True, tracker.innovation(source, output_id)))
        return Genome(nodes=nodes, connections=connections)

    def expand(self, entry: TaskEntry, genomes: list[Genome], tracker: InnovationTracker, rng: random.Random) -> list[Genome]:
        """Grow the population's interface to host `entry`: widen its input bank and/or add its head.

        New input nodes are wired to the pre-existing output heads; the new head (if any) is wired
        from all inputs + bias, so both have a trainable readout immediately. A no-op (returns the
        same genomes) when the task is already accommodated.
        """
        new_inputs = self._grow_bank(entry, tracker)
        existing_outputs = self._all_output_ids()
        new_head_ids = self._grow_head(entry, tracker)
        if not new_inputs and not new_head_ids:
            return genomes

        all_input_ids = self._all_input_ids()
        grown: list[Genome] = []
        for genome in genomes:
            child = genome.clone()
            for node_id, coordinate in new_inputs:
                child.nodes[node_id] = NodeGene(node_id, NodeKind.INPUT, "identity", coordinate)
                for output_id in existing_outputs:
                    child.connections.append(ConnectionGene(node_id, output_id, rng.gauss(0.0, 1.0), True, tracker.innovation(node_id, output_id)))
            for output_id in new_head_ids:
                child.nodes[output_id] = NodeGene(output_id, NodeKind.OUTPUT, "identity", None)
                for source in [*all_input_ids, self.bias_id]:
                    child.connections.append(ConnectionGene(source, output_id, rng.gauss(0.0, 1.0), True, tracker.innovation(source, output_id)))
            grown.append(child)
        return grown

    def adapter(self, entry: TaskEntry) -> MultiTaskAdapter:
        """Build the active-task adapter: full-width input (active bank filled, others zero) + head columns."""
        bank = self.banks[entry.input_signature]
        encoded_task = encode_task(entry.task, Level0Encoder(bank.width))
        columns = self._input_columns(bank)
        support_input = (self._scatter(encoded_task.support_input[0], columns), encoded_task.support_input[1])
        query_input = None
        if encoded_task.query_input is not None:
            query_input = (self._scatter(encoded_task.query_input[0], columns), encoded_task.query_input[1])
        encoded = EncodedTask(
            support_input=support_input,
            support_target=encoded_task.support_target,
            query_input=query_input,
            query_target=encoded_task.query_target,
        )
        head_columns = torch.tensor(self._output_columns(entry.name), dtype=torch.long)
        return MultiTaskAdapter(encoded=encoded, encoder=Level0Encoder(bank.width), n_inputs=self.n_inputs, n_outputs=self.n_outputs, head_columns=head_columns)

    def _input_columns(self, bank: _Bank) -> list[int]:
        position = {node_id: index for index, node_id in enumerate(self._all_input_ids())}
        return [position[node_id] for node_id in bank.node_ids]

    def _output_columns(self, name: str) -> list[int]:
        position = {node_id: index for index, node_id in enumerate(self._all_output_ids())}
        return [position[node_id] for node_id in self.heads[name].node_ids]

    def _scatter(self, values: torch.Tensor, columns: list[int]) -> torch.Tensor:
        full = torch.zeros(values.shape[0], self.n_inputs, dtype=values.dtype)
        for source_column, target_column in enumerate(columns):
            full[:, target_column] = values[:, source_column]
        return full

    def to_dict(self) -> dict[str, Any]:
        return {
            "bias_id": self.bias_id,
            "banks": {
                signature: {"axes": list(bank.axes), "node_ids": list(bank.node_ids), "coordinates": [list(coordinate) for coordinate in bank.coordinates], "width": bank.width}
                for signature, bank in self.banks.items()
            },
            "heads": {name: list(head.node_ids) for name, head in self.heads.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_activation: str) -> "MultiTaskSubstrate":
        substrate = cls(default_activation=default_activation, bias_id=int(data["bias_id"]))
        substrate.banks = {
            signature: _Bank(
                signature=signature,
                axes=tuple(bank["axes"]),
                node_ids=[int(node_id) for node_id in bank["node_ids"]],
                coordinates=[tuple(coordinate) for coordinate in bank["coordinates"]],
                width=int(bank["width"]),
            )
            for signature, bank in data["banks"].items()
        }
        substrate.heads = {name: _Head(name=name, node_ids=[int(node_id) for node_id in ids]) for name, ids in data["heads"].items()}
        return substrate
