"""
Compact, mutable placements of ordinary neurons over arbitrary tensor contracts.

Input and output nodes denote port banks. Hidden nodes denote scalar neuron
definitions with mutable placement domains. Connections carry index bindings;
neither a neighborhood stencil nor a pooling feature bank is supplied.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import torch
from torch import nn

from versal.dataset.icarus import Axis, EncodedTask, Field, FieldDescriptor, Level0Encoder, Task, ValueType, as_logits, loss_fn, model_output_features
from versal.evaluation import split_metrics_from_raw
from versal.evolution.genome import ConnectionGene, Genome, MacroGene, NodeGene, NodeKind, genome_from_dict, genome_to_dict, topological_order
from versal.reference_depth import DEFAULT_MAX_INLINE_DEPTH
from versal.representation import REPRESENTATION, RepresentationCodec
from versal.substrate import _ACTIVATIONS, SubstrateModule
from versal.utils.deadline import expired

SPATIAL_VERSION = 1


@dataclass(frozen=True)
class SpatialContract:
    """
    Support-derived ports and dimension rules, independent of task provenance.
    """

    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_axes: tuple[str, ...]
    output_axes: tuple[str, ...]
    input_type: str
    output_type: str
    input_classes: int | None = None
    output_classes: int | None = None
    input_range: tuple[float, float] | None = None
    output_range: tuple[float, float] | None = None
    shape_rules: tuple[tuple[int, int, int], ...] = ()

    @classmethod
    def from_task(cls, task: Task) -> SpatialContract:
        if not task.support:
            raise ValueError("spatial search needs non-empty support")
        source, target = task.support[0]
        for x, y in task.support:
            for current, first in ((x, source), (y, target)):
                if (current.data.ndim, current.axes, current.value_type, current.n_classes, current.value_range) != (
                    first.data.ndim,
                    first.axes,
                    first.value_type,
                    first.n_classes,
                    first.value_range,
                ):
                    raise ValueError("support examples must share typed tensor descriptors")
        input_shape = tuple(max(x.data.shape[d] for x, _ in task.support) for d in range(source.data.ndim))
        output_shape = tuple(max(y.data.shape[d] for _, y in task.support) for d in range(target.data.ndim))
        rules = []
        for d, extent in enumerate(output_shape):
            candidates = []
            for axis in range(len(input_shape)):
                for scale in (-2, -1, 1, 2):
                    bias = int(target.data.shape[d]) - scale * int(source.data.shape[axis])
                    # A single observed size supports identity on the same axis,
                    # but cannot identify a nontrivial affine dimension relation.
                    if len({int(x.data.shape[axis]) for x, _ in task.support}) < 2 and (scale != 1 or bias != 0 or source.axes[axis] != target.axes[d]):
                        continue
                    if all(scale * int(x.data.shape[axis]) + bias == int(y.data.shape[d]) for x, y in task.support):
                        candidates.append((int(source.axes[axis] != target.axes[d]), abs(bias), abs(scale), axis, scale, bias))
            if candidates:
                _, _, _, axis, scale, bias = min(candidates)
                rules.append((axis, scale, bias))
            elif all(int(y.data.shape[d]) == extent for _, y in task.support):
                rules.append((-1, 0, extent))
            else:
                raise ValueError("support output shapes have no consistent tensor dimension binding")
        return cls(
            input_shape,
            output_shape,
            tuple(a.value for a in source.axes),
            tuple(a.value for a in target.axes),
            source.value_type.value,
            target.value_type.value,
            source.n_classes,
            target.n_classes,
            source.value_range,
            target.value_range,
            tuple(rules),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SpatialContract:
        data = dict(value)
        for key in ("input_shape", "output_shape", "input_axes", "output_axes", "input_range", "output_range"):
            if data.get(key) is not None:
                data[key] = tuple(data[key])
        data["shape_rules"] = tuple(tuple(rule) for rule in data.get("shape_rules", ()))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def input_width(self) -> int:
        return math.prod(self.input_shape)

    @property
    def output_width(self) -> int:
        return model_output_features(self.output_descriptor, math.prod(self.output_shape))

    @property
    def output_descriptor(self) -> FieldDescriptor:
        return FieldDescriptor(tuple(Axis(axis) for axis in self.output_axes), ValueType(self.output_type), self.output_classes, self.output_range)

    @property
    def input_descriptor(self) -> FieldDescriptor:
        return FieldDescriptor(tuple(Axis(axis) for axis in self.input_axes), ValueType(self.input_type), self.input_classes, self.input_range)

    @property
    def logit_shape(self) -> tuple[int, ...]:
        return (*self.output_shape, int(self.output_classes or 1)) if self.output_type in {"CATEGORICAL", "ORDINAL"} else self.output_shape

    def compatible(self, other: SpatialContract) -> bool:
        return (self.input_axes, self.output_axes, self.input_type, self.output_type, self.input_classes, self.output_classes, self.input_range, self.output_range) == (
            other.input_axes,
            other.output_axes,
            other.input_type,
            other.output_type,
            other.input_classes,
            other.output_classes,
            other.input_range,
            other.output_range,
        )

    def bind(self, input_shape: tuple[int, ...]) -> SpatialContract:
        if len(input_shape) != len(self.input_shape):
            raise ValueError("spatial binding needs the declared input rank")
        shape = tuple(max(1, scale * input_shape[axis] + bias if axis >= 0 else bias) for axis, scale, bias in self.shape_rules)
        return replace(self, input_shape=input_shape, output_shape=shape)


@dataclass(frozen=True)
class Placement:
    """
    A singleton, fixed repetition, or repetition over selected contract dimensions.
    """

    bank: str = "fixed"
    axes: tuple[int, ...] = ()
    count: int = 1
    start: int = 0
    stop: int | None = None

    def shape(self, contract: SpatialContract) -> tuple[int, ...]:
        if self.bank == "fixed":
            return (max(1, self.count),)
        shape = contract.input_shape if self.bank == "input" else contract.logit_shape
        return tuple(shape[axis] for axis in self.axes) if self.axes else (math.prod(shape),)

    def size(self, contract: SpatialContract) -> int:
        return math.prod(self.shape(contract))

    def active(self, contract: SpatialContract) -> tuple[int, int]:
        size = self.size(contract)
        return min(size, max(0, self.start)), min(size, self.stop if self.stop is not None else size)


@dataclass(frozen=True)
class Binding:
    """
    Mutable integer index maps; a fan is ordinary repeated incoming connections.
    """

    scale: int = 0
    shift: int = 0
    fan_in: int = 1  # zero means the full source bank, not a precomputed statistic
    fan_stride: int = 1
    target_start: int = 0
    target_stride: int = 1
    target_count: int = 0  # zero means the whole target bank
    shared: bool = False
    matrix: tuple[tuple[int, ...], ...] = ()
    offsets: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.fan_in < 0 or self.target_start < 0 or self.target_stride < 1 or self.target_count < 0:
            raise ValueError("invalid connection binding extent")
        if self.matrix and len(self.offsets) != len(self.matrix):
            raise ValueError("each mapped source axis needs an offset")

    def indices(self, source_shape: tuple[int, ...], target_shape: tuple[int, ...], *, chunk_size: int = 32768, device: Any = "cpu"):
        source_size, target_size = math.prod(source_shape), math.prod(target_shape)
        count = self.target_count or max(0, (target_size - self.target_start + self.target_stride - 1) // self.target_stride)
        count = min(count, max(0, (target_size - self.target_start + self.target_stride - 1) // self.target_stride))
        fan = self.fan_in or source_size
        for first in range(0, count * fan, chunk_size):
            ordinal = torch.arange(first, min(count * fan, first + chunk_size), device=device)
            target = self.target_start + ordinal.div(fan, rounding_mode="floor") * self.target_stride
            source = target * self.scale + self.shift + ordinal.remainder(fan) * self.fan_stride
            valid = (source >= 0) & (source < source_size)
            if self.matrix:
                if len(self.matrix) != len(source_shape) or any(len(row) != len(target_shape) for row in self.matrix):
                    raise ValueError("index-map ranks do not match their bound port domains")
                coords = _coordinates(target, target_shape)
                mapped = [sum((coords[d] * coefficient for d, coefficient in enumerate(row)), torch.zeros_like(target)) + self.offsets[i] for i, row in enumerate(self.matrix)]
                source = torch.zeros_like(target)
                valid = torch.ones_like(target, dtype=torch.bool)
                for coordinate, extent in zip(mapped, source_shape):
                    valid &= (coordinate >= 0) & (coordinate < extent)
                    source = source * extent + coordinate
                source += ordinal.remainder(fan) * self.fan_stride
                valid &= (source >= 0) & (source < source_size)
            yield source[valid], target[valid], ordinal[valid]

    def upper_count(self, source_shape: tuple[int, ...], target_shape: tuple[int, ...]) -> int:
        count = max(0, (math.prod(target_shape) - self.target_start + self.target_stride - 1) // self.target_stride)
        return min(count, self.target_count or count) * (self.fan_in or math.prod(source_shape))

    def count(
        self, source_shape: tuple[int, ...], target_shape: tuple[int, ...], *, source_interval: tuple[int, int] | None = None, target_interval: tuple[int, int] | None = None
    ) -> int:
        """
        Count executed edges without expanding a potentially enormous Cartesian fan.
        """
        source_size, target_size = math.prod(source_shape), math.prod(target_shape)
        source_low, source_high = source_interval or (0, source_size)
        target_low, target_high = target_interval or (0, target_size)
        count = max(0, (target_size - self.target_start + self.target_stride - 1) // self.target_stride)
        count = min(count, self.target_count or count)
        first = max(0, (target_low - self.target_start + self.target_stride - 1) // self.target_stride)
        stop = min(count, (target_high - self.target_start + self.target_stride - 1) // self.target_stride)
        if stop <= first or source_high <= source_low:
            return 0
        fan = self.fan_in or source_size
        if not self.matrix:
            if self.scale == 0:
                return (stop - first) * _run_count(self.shift, self.fan_stride, fan, source_low, source_high)
            if fan == 1:
                return _run_count(
                    (self.target_start + first * self.target_stride) * self.scale + self.shift, self.target_stride * self.scale, stop - first, source_low, source_high
                )
        total = 0
        for start in range(first, stop, 32768):
            targets = self.target_start + torch.arange(start, min(stop, start + 32768)) * self.target_stride
            base = targets * self.scale + self.shift
            valid = torch.ones_like(targets, dtype=torch.bool)
            if self.matrix:
                if len(self.matrix) != len(source_shape) or any(len(row) != len(target_shape) for row in self.matrix):
                    raise ValueError("index-map ranks do not match their bound port domains")
                coordinates = _coordinates(targets, target_shape)
                base = torch.zeros_like(targets)
                for row, offset, extent in zip(self.matrix, self.offsets, source_shape):
                    mapped = sum((coordinates[d] * coefficient for d, coefficient in enumerate(row)), torch.zeros_like(targets)) + offset
                    valid &= (mapped >= 0) & (mapped < extent)
                    base = base * extent + mapped
            if self.fan_stride == 0:
                total += int(((base >= source_low) & (base < source_high) & valid).sum()) * fan
                continue
            if self.fan_stride < 0:
                base = base + (fan - 1) * self.fan_stride
            stride = abs(self.fan_stride)
            low = torch.div(source_low - base + stride - 1, stride, rounding_mode="floor").clamp(0, fan)
            high = torch.div(source_high - base + stride - 1, stride, rounding_mode="floor").clamp(0, fan)
            total += int(((high - low).clamp_min(0) * valid).sum())
        return total


def _run_count(base: int, stride: int, count: int, low: int, high: int) -> int:
    if stride == 0:
        return count if low <= base < high else 0
    if stride < 0:
        base, stride = base + (count - 1) * stride, -stride
    first = max(0, min(count, (low - base + stride - 1) // stride))
    stop = max(0, min(count, (high - base + stride - 1) // stride))
    return max(0, stop - first)


def _coordinates(indices: torch.Tensor, shape: tuple[int, ...]) -> list[torch.Tensor]:
    result = []
    for extent in reversed(shape):
        result.append(indices.remainder(extent))
        indices = indices.div(extent, rounding_mode="floor")
    return list(reversed(result))


@dataclass
class SpatialGenome(Genome):
    """
    Ordinary circuit definitions with separate placement and parameter genes.
    """

    representation = "spatial"
    contract: SpatialContract | None = None
    placements: dict[int, Placement] = field(default_factory=dict)
    bindings: dict[int, Binding] = field(default_factory=dict)
    parameters: dict[int, list[float]] = field(default_factory=dict)
    groups: dict[int, list[int]] = field(default_factory=dict)
    _counts_cache: Any = field(default=None, init=False, repr=False, compare=False)

    def clone(self) -> SpatialGenome:
        return copy.deepcopy(self)

    def node_shape(self, node_id: int, contract: SpatialContract | None = None) -> tuple[int, ...]:
        contract = contract or self.contract
        assert contract is not None
        kind = self.nodes[node_id].kind
        if kind is NodeKind.INPUT:
            return contract.input_shape or (1,)
        if kind is NodeKind.OUTPUT:
            return contract.logit_shape or (1,)
        return self.placements.get(node_id, Placement()).shape(contract)

    def complexity(self) -> int:
        return self.expanded_counts()[0]

    def expanded_counts(self) -> tuple[int, int, int]:
        assert self.contract is not None
        identity = (
            self.contract,
            tuple((node.id, node.kind) for node in self.nodes.values()),
            tuple(self.placements.items()),
            tuple((edge.in_id, edge.out_id, edge.innovation, self.bindings[edge.innovation]) for edge in self.enabled_connections()),
            tuple(self.macros),
        )
        if self._counts_cache is not None and self._counts_cache[0] == identity:
            return self._counts_cache[1]
        hidden = sum(max(0, end - start) for node in self.hidden_ids for start, end in [self.placements.get(node, Placement()).active(self.contract)])
        edges = 0
        for edge in self.enabled_connections():
            rule = self.bindings[edge.innovation]
            intervals = [self.placements.get(node, Placement()).active(self.contract) if self.nodes[node].kind is NodeKind.HIDDEN else None for node in (edge.in_id, edge.out_id)]
            edges += rule.count(self.node_shape(edge.in_id), self.node_shape(edge.out_id), source_interval=intervals[0], target_interval=intervals[1])
        copies = sum(self.reference_copies().values())
        counts = (hidden + edges + copies, hidden, edges)
        self._counts_cache = (identity, counts)
        return counts

    def reference_copies(self) -> dict[int, int]:
        assert self.contract is not None
        return {macro.innovation: max(0, high - low) for macro in self.macros for low, high in [self.placements.get(macro.output_node_ids[0], Placement()).active(self.contract)]}

    def to_payload(self) -> dict[str, Any]:
        base = Genome(dict(self.nodes), list(self.connections), list(self.macros), self.refine_steps, dict(self.operator_rates))
        assert self.contract is not None
        return genome_to_dict(base) | {
            "representation": "spatial",
            "spatial": {
                "version": SPATIAL_VERSION,
                "contract": self.contract.to_dict(),
                "placements": {str(key): asdict(value) for key, value in self.placements.items()},
                "bindings": {str(key): asdict(value) for key, value in self.bindings.items()},
                "parameters": {str(key): value for key, value in self.parameters.items()},
                "groups": {str(key): value for key, value in self.groups.items()},
            },
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> SpatialGenome:
        data = payload["spatial"]
        if int(data["version"]) != SPATIAL_VERSION:
            raise ValueError(f"unsupported spatial recipe version {data['version']}")
        base = genome_from_dict({key: value for key, value in payload.items() if key not in {"representation", "spatial"}})
        placements = {int(key): Placement(**(value | {"axes": tuple(value.get("axes", ()))})) for key, value in data["placements"].items()}
        bindings = {
            int(key): Binding(**(value | {"matrix": tuple(tuple(row) for row in value.get("matrix", ())), "offsets": tuple(value.get("offsets", ()))}))
            for key, value in data["bindings"].items()
        }
        return cls(
            nodes=base.nodes,
            connections=base.connections,
            macros=base.macros,
            refine_steps=base.refine_steps,
            operator_rates=base.operator_rates,
            contract=SpatialContract.from_dict(data["contract"]),
            placements=placements,
            bindings=bindings,
            parameters={int(key): list(value) for key, value in data.get("parameters", {}).items()},
            groups={int(key): list(value) for key, value in data.get("groups", {}).items()},
        )

    def repair(self) -> SpatialGenome:
        from versal.evolution.genome import make_acyclic

        base = Genome(dict(self.nodes), list(self.connections), list(self.macros), self.refine_steps, dict(self.operator_rates))
        child = self.clone()
        child.connections = make_acyclic(base).connections
        return child


class SpatialNet(SubstrateModule):
    """
    Sparse scalar-neuron execution, vectorized over placements and bounded edges.
    """

    def __init__(
        self,
        genome: SpatialGenome,
        *,
        contract: SpatialContract | None = None,
        chunk_size: int = 32768,
        deadline: float | None = None,
        library_dir: str | None = None,
        macro_resolver: Any = None,
        max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH,
        _reference_depth: int = 0,
        _reference_stack: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.genome = genome.clone()
        bound_contract = contract or genome.contract
        assert bound_contract is not None
        self.contract: SpatialContract = bound_contract
        self.genome.contract = bound_contract
        genome = self.genome
        self.chunk_size, self.deadline = chunk_size, deadline
        self.order = topological_order(genome)
        for macro in genome.macros:
            placements = [genome.placements.get(node, Placement()) for node in (*macro.input_node_ids, *macro.output_node_ids)]
            if not placements or any(placement != placements[0] for placement in placements):
                raise ValueError("all ports of a repeated module must share its placement domain")
        self.weights = nn.ParameterDict()
        for edge in genome.enabled_connections():
            rule = genome.bindings[edge.innovation]
            if rule.matrix and (len(rule.matrix) != len(genome.node_shape(edge.in_id)) or any(len(row) != len(genome.node_shape(edge.out_id)) for row in rule.matrix)):
                raise ValueError("index-map ranks do not match their bound port domains")
            count = 1 if rule.shared else rule.upper_count(genome.node_shape(edge.in_id), genome.node_shape(edge.out_id))
            values = genome.parameters.get(edge.innovation, [])
            tensor = torch.full((max(1, count),), edge.weight)
            if values:
                tensor[: min(len(tensor), len(values))] = torch.tensor(values[: len(tensor)])
            self.weights[str(edge.innovation)] = nn.Parameter(tensor)
        self._incoming: dict[int, list[ConnectionGene]] = {node: [] for node in genome.nodes}
        for edge in genome.enabled_connections():
            self._incoming[edge.out_id].append(edge)
        self.training_groups: list[tuple[SpatialContract, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._index_cache: dict[Any, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        self._cached_indices = 0
        self._macro_modules = nn.ModuleDict()
        self._macro_outputs = {node: macro for macro in genome.macros for node in macro.output_node_ids}
        if genome.macros:
            from versal.evolution.composition import AssemblyContext, CompNodeGene, CompNodeKind, _resolve_module
            from versal.library import ModuleLibrary
            from versal.substrate import decode_module

            library = ModuleLibrary(library_dir) if library_dir is not None else getattr(macro_resolver, "library", None)
            context = AssemblyContext({}, library=library, max_inline_depth=max_inline_depth, reference_depth=_reference_depth, expansion_stack=list(_reference_stack))
            for macro in genome.macros:
                key = macro.ref.removeprefix("library:")
                if library is not None:
                    node = CompNodeGene(macro.innovation, CompNodeKind.MODULE, f"library:{key}", len(macro.input_node_ids), len(macro.output_node_ids), trainable=macro.trainable)
                    inner = _resolve_module(node, context)
                else:
                    if macro_resolver is None or key in _reference_stack or _reference_depth >= max_inline_depth:
                        raise ValueError("spatial reference cannot be resolved within the reference depth limit")
                    inner = decode_module(
                        macro_resolver(key),
                        len(macro.input_node_ids),
                        len(macro.output_node_ids),
                        macro_resolver=macro_resolver,
                        max_inline_depth=max_inline_depth,
                        _reference_depth=_reference_depth + 1,
                        _reference_stack=(*_reference_stack, key),
                    )
                if not macro.trainable:
                    inner.requires_grad_(False)
                self._macro_modules[str(macro.innovation)] = inner

    @property
    def has_edges(self) -> bool:
        return bool(self.weights) or bool(self._macro_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_bound(x, self.contract)

    def forward_bound(self, x: torch.Tensor, contract: SpatialContract) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != contract.input_width:
            raise ValueError(f"spatial input expects {contract.input_width} values, got {tuple(x.shape)}")
        genome = self.genome
        previous: dict[int, torch.Tensor] = {}
        for _ in range(max(1, genome.refine_steps)):
            values: dict[int, torch.Tensor] = {}
            for node_id in self.order:
                if expired(self.deadline):
                    raise TimeoutError("spatial execution reached its stop boundary")
                node = genome.nodes[node_id]
                if node_id in values:
                    continue
                shape = genome.node_shape(node_id, contract)
                size = math.prod(shape)
                if node.kind is NodeKind.INPUT:
                    values[node_id] = x
                    continue
                if node.kind is NodeKind.BIAS:
                    values[node_id] = torch.ones((len(x), size), device=x.device, dtype=x.dtype)
                    continue
                if node_id in self._macro_outputs:
                    macro = self._macro_outputs[node_id]
                    pieces = [values[source] for source in macro.input_node_ids]
                    if any(piece.shape[1] != size for piece in pieces):
                        raise ValueError("all ports of a repeated module must share its placement domain")
                    low, high = genome.placements.get(node_id, Placement()).active(contract)
                    gathered = torch.stack([piece[:, low:high] for piece in pieces], dim=-1).reshape(-1, len(pieces))
                    blocks = []
                    for first in range(0, len(gathered), self.chunk_size):
                        if expired(self.deadline):
                            raise TimeoutError("spatial reference reached its stop boundary")
                        blocks.append(self._macro_modules[str(macro.innovation)](gathered[first : first + self.chunk_size]))
                    result = torch.cat(blocks).reshape(len(x), high - low, len(macro.output_node_ids)) if blocks else x.new_zeros((len(x), 0, len(macro.output_node_ids)))
                    for index, target in enumerate(macro.output_node_ids):
                        values[target] = torch.nn.functional.pad(result[:, :, index], (low, size - high))
                    continue
                product = node.aggregation == "product"
                value = torch.ones((len(x), size), device=x.device, dtype=x.dtype) if product else torch.zeros((len(x), size), device=x.device, dtype=x.dtype)
                touched = torch.zeros(size, device=x.device, dtype=torch.bool)
                for edge in self._incoming[node_id]:
                    source_shape = genome.node_shape(edge.in_id, contract)
                    source_values = (previous if edge.recurrent else values).get(edge.in_id)
                    if source_values is None:
                        source_values = torch.zeros((len(x), math.prod(source_shape)), device=x.device, dtype=x.dtype)
                    rule = genome.bindings[edge.innovation]
                    parameter = self.weights[str(edge.innovation)]
                    cache_key = (edge.innovation, source_shape, shape, str(x.device))
                    batches = self._index_cache.get(cache_key)
                    if batches is None:
                        count = rule.upper_count(source_shape, shape)
                        generated = rule.indices(source_shape, shape, chunk_size=self.chunk_size, device=x.device)
                        if count + self._cached_indices <= self.chunk_size:
                            batches = list(generated)
                            self._index_cache[cache_key] = batches
                            self._cached_indices += count
                        else:
                            batches = generated
                    for sources, targets, ordinal in batches:
                        if expired(self.deadline):
                            raise TimeoutError("spatial connection gathering reached its stop boundary")
                        valid = torch.ones_like(sources, dtype=torch.bool)
                        for endpoint, positions in ((edge.in_id, sources), (edge.out_id, targets)):
                            if genome.nodes[endpoint].kind is NodeKind.HIDDEN:
                                low, high = genome.placements.get(endpoint, Placement()).active(contract)
                                valid &= (positions >= low) & (positions < high)
                        sources, targets, ordinal = sources[valid], targets[valid], ordinal[valid]
                        weight = parameter[0] if rule.shared else torch.where(ordinal < len(parameter), parameter[ordinal.clamp_max(len(parameter) - 1)], parameter[0])
                        contribution = source_values[:, sources] * weight
                        if product:
                            value = value.scatter_reduce(1, targets.expand(len(x), -1), contribution, reduce="prod", include_self=True)
                        else:
                            value = value.index_add(1, targets, contribution)
                        touched[targets] = True
                if product:
                    value = value * touched
                if node.activation != "identity":
                    value = _ACTIVATIONS[node.activation](value)
                if node.kind is NodeKind.HIDDEN:
                    low, high = genome.placements.get(node_id, Placement()).active(contract)
                    positions = torch.arange(size, device=x.device)
                    value = value * ((positions >= low) & (positions < high))
                values[node_id] = value
            previous = values
        return torch.cat([previous[node] for node in genome.output_ids], dim=1)

    def training_loss(self, encoded: EncodedTask) -> torch.Tensor:
        if not self.training_groups:
            data, _ = encoded.support_input
            target, mask, descriptor = encoded.support_target
            return loss_fn(as_logits(self(data), descriptor, target.shape[-1]), target, descriptor, mask)
        device = next(self.parameters(), torch.empty(0)).device
        total = torch.zeros((), device=device)
        count = 0
        for contract, data, target, mask in self.training_groups:
            valid = int((~mask).sum())
            if valid == 0:
                continue
            raw = as_logits(self.forward_bound(data.to(device), contract), contract.output_descriptor, math.prod(contract.output_shape))
            total = total + loss_fn(raw, target.to(device), contract.output_descriptor, mask.to(device)) * valid
            count += valid
        return total / max(1, count)

    def export_weights(self) -> dict[tuple[int, int, bool], float]:
        return {(edge.in_id, edge.out_id, edge.recurrent): float(self.weights[str(edge.innovation)][0].detach()) for edge in self.genome.enabled_connections()}

    def writeback(self, genome: SpatialGenome) -> SpatialGenome:
        child = genome.clone()
        child.contract = self.contract
        child.parameters = {int(key): value.detach().cpu().tolist() for key, value in self.weights.items()}
        child.connections = [replace(edge, weight=child.parameters[edge.innovation][0]) if edge.innovation in child.parameters else edge for edge in child.connections]
        return child


def encode_field(field: Field, shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    data = field.data.to(torch.float32)
    if field.value_type is ValueType.CONTINUOUS and field.value_range is not None:
        low, high = field.value_range
        if high > low:
            data = (data - low) / (high - low)
    if any(current > maximum for current, maximum in zip(data.shape, shape)):
        raise ValueError("spatial encoding cannot truncate a tensor")
    tensor, mask = torch.zeros(shape), torch.ones(shape, dtype=torch.bool)
    region = tuple(slice(0, extent) for extent in data.shape)
    valid = torch.ones_like(data, dtype=torch.bool) if field.mask is None else ~field.mask
    tensor[region] = torch.where(valid, data, 0.0)
    mask[region] = ~valid
    return tensor.reshape(-1), mask.reshape(-1)


class SpatialTaskAdapter:
    """
    Contract-only adapter with ordinary losses and support-only training tensors.
    """

    n_inputs = 1  # symbolic input bank; not a count of raw tensor cells
    n_outputs = 1

    def __init__(
        self,
        task: Task,
        *,
        include_query: bool = False,
        chunk_size: int = 32768,
        deadline: float | None = None,
        library_dir: str | None = None,
        max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH,
        max_expanded_edges: int = 1_000_000,
        max_activation_cells: int = 8_000_000,
    ) -> None:
        self.task = task if include_query else Task(task.meta, task.support, [])
        self.contract = SpatialContract.from_task(task)
        self.encoder = Level0Encoder(self.contract.input_width)
        self.chunk_size, self.deadline = chunk_size, deadline
        self.library_dir, self.max_inline_depth = library_dir, max_inline_depth
        self.max_expanded_edges, self.max_activation_cells = max_expanded_edges, max_activation_cells
        inputs, targets, masks = [], [], []
        for source, target in task.support:
            inputs.append(encode_field(source, self.contract.input_shape)[0])
            data, mask = encode_field(target, self.contract.output_shape)
            targets.append(data)
            masks.append(mask)
        target_tensor = torch.stack(targets)
        if self.contract.output_type in {"CATEGORICAL", "ORDINAL"}:
            target_tensor = target_tensor.long()
        self.encoded = EncodedTask((torch.stack(inputs), self.contract.input_descriptor), (target_tensor, torch.stack(masks), self.contract.output_descriptor), None, None)

        grouped: dict[SpatialContract, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        for source, target in self.task.support:
            contract = self.contract.bind(tuple(source.data.shape))
            if contract.output_shape != tuple(target.data.shape):
                raise ValueError("support output shapes do not follow a consistent dimension binding")
            data, _ = encode_field(source, contract.input_shape)
            labels, mask = encode_field(target, contract.output_shape)
            if contract.output_type in {"CATEGORICAL", "ORDINAL"}:
                labels = labels.long()
            grouped.setdefault(contract, []).append((data, labels, mask))
        self.training_groups = [
            (contract, torch.stack([row[0] for row in rows]), torch.stack([row[1] for row in rows]), torch.stack([row[2] for row in rows])) for contract, rows in grouped.items()
        ]

    def decode(self, genome: Genome) -> SpatialNet:
        if not isinstance(genome, SpatialGenome):
            raise ValueError("spatial adapter requires a spatial genome")
        edges = sum(
            genome.bindings[edge.innovation].upper_count(genome.node_shape(edge.in_id, self.contract), genome.node_shape(edge.out_id, self.contract))
            for edge in genome.enabled_connections()
        )
        cells = sum(math.prod(genome.node_shape(node, self.contract)) for node in genome.nodes) * len(self.task.support)
        if edges > self.max_expanded_edges or cells > self.max_activation_cells:
            raise ValueError("spatial candidate exceeds the configured execution budget")
        module = SpatialNet(
            genome, contract=self.contract, chunk_size=self.chunk_size, deadline=self.deadline, library_dir=self.library_dir, max_inline_depth=self.max_inline_depth
        )
        module.training_groups = self.training_groups
        return module

    def evaluate(self, module: SubstrateModule) -> dict[str, float]:
        if not isinstance(module, SpatialNet):
            raise ValueError("spatial evaluation requires a spatial executable")
        metrics = evaluate_spatial(module, self.task, split="support")
        if self.task.query:
            metrics.update(evaluate_spatial(module, self.task, split="query"))
        else:
            metrics.update(query_loss=math.inf, query_accuracy=0.0)
        complexity, hidden, edges = module.genome.expanded_counts()
        metrics.update(
            expanded_complexity=float(complexity),
            spatial_expanded_hidden=float(hidden),
            spatial_expanded_edges=float(edges),
            spatial_recipe_nodes=float(len(module.genome.nodes)),
            spatial_parameters=float(sum(p.numel() for p in module.parameters())),
            spatial_placements=float(sum(p.size(module.contract) for p in module.genome.placements.values())),
        )
        return metrics


def evaluate_spatial(module: SpatialNet, task: Task, *, split: str) -> dict[str, float]:
    pairs = task.support if split == "support" else task.query
    if not pairs:
        return {f"{split}_loss": math.inf}
    device = next(module.parameters(), torch.empty(0)).device
    total, correct, loss_sum, exact = 0, 0.0, 0.0, 0
    for source, target in pairs:
        valid = int((~target.mask).sum()) if target.mask is not None else target.data.numel()
        total += valid
        if (
            FieldDescriptor(source.axes, source.value_type, source.n_classes, source.value_range) != module.contract.input_descriptor
            or FieldDescriptor(target.axes, target.value_type, target.n_classes, target.value_range) != module.contract.output_descriptor
        ):
            loss_sum = math.inf
            continue
        if expired(module.deadline):
            raise TimeoutError("spatial verification reached its stop boundary")
        contract = module.contract.bind(tuple(source.data.shape))
        if tuple(target.data.shape) != contract.output_shape:
            loss_sum = math.inf
            continue
        data, _ = encode_field(source, contract.input_shape)
        target_data, mask = encode_field(target, contract.output_shape)
        if contract.output_type in {"CATEGORICAL", "ORDINAL"}:
            target_data = target_data.long()
        with torch.no_grad():
            raw = as_logits(module.forward_bound(data.unsqueeze(0).to(device), contract), contract.output_descriptor, math.prod(contract.output_shape))
            accuracy, loss = split_metrics_from_raw(
                raw, target_data.unsqueeze(0).to(device), mask.unsqueeze(0).to(device), contract.output_descriptor, Level0Encoder(contract.input_width)
            )
        correct += accuracy * valid
        loss_sum += loss * valid
        exact += int(accuracy >= 1.0)
    return {
        f"{split}_accuracy": correct / max(1, total),
        f"{split}_loss": loss_sum / max(1, total),
        f"{split}_valid_cells": float(total),
        f"{split}_correct_cells": correct,
        f"{split}_total_examples": float(len(pairs)),
        f"{split}_exact_examples": float(exact),
    }


def expand_spatial(genome: SpatialGenome, *, limit: int = 16384) -> Genome:
    """
    Materialize a small bound recipe for direct search and exact decoder checks.
    """
    assert genome.contract is not None
    estimate = sum(math.prod(genome.node_shape(node)) for node in genome.nodes)
    estimate += sum(genome.bindings[edge.innovation].upper_count(genome.node_shape(edge.in_id), genome.node_shape(edge.out_id)) for edge in genome.enabled_connections())
    if estimate > limit:
        raise ValueError(f"spatial expansion needs up to {estimate:,} genes; expansion limit is {limit:,}")
    nodes: dict[int, NodeGene] = {}
    ids: dict[tuple[int, int], int] = {}
    for kind in (NodeKind.INPUT, NodeKind.BIAS, NodeKind.HIDDEN, NodeKind.OUTPUT):
        for node in sorted(genome.nodes.values(), key=lambda item: item.id):
            if node.kind is not kind:
                continue
            low, high = genome.placements.get(node.id, Placement()).active(genome.contract) if kind is NodeKind.HIDDEN else (0, math.prod(genome.node_shape(node.id)))
            for index in range(low, high):
                identity = len(nodes)
                ids[node.id, index] = identity
                nodes[identity] = replace(node, id=identity, coordinate=None)
    connections: list[ConnectionGene] = []
    seen: set[tuple[int, int, bool]] = set()
    for edge in genome.enabled_connections():
        rule = genome.bindings[edge.innovation]
        parameters = genome.parameters.get(edge.innovation, [edge.weight])
        for sources, targets, ordinals in rule.indices(genome.node_shape(edge.in_id), genome.node_shape(edge.out_id)):
            for source, target, ordinal in zip(sources.tolist(), targets.tolist(), ordinals.tolist()):
                if (edge.in_id, source) not in ids or (edge.out_id, target) not in ids:
                    continue
                left, right = ids[edge.in_id, source], ids[edge.out_id, target]
                weight = parameters[0] if rule.shared or ordinal >= len(parameters) else parameters[ordinal]
                tie = edge.innovation if rule.shared else None
                if (left, right, edge.recurrent) in seen:
                    bridge = len(nodes)
                    nodes[bridge] = NodeGene(bridge, NodeKind.HIDDEN, "identity")
                    connections.append(ConnectionGene(left, bridge, weight, True, len(connections), edge.recurrent, tie))
                    connections.append(ConnectionGene(bridge, right, 1.0, True, len(connections)))
                else:
                    connections.append(ConnectionGene(left, right, weight, True, len(connections), edge.recurrent, tie))
                    seen.add((left, right, edge.recurrent))
    macros = []
    for macro in genome.macros:
        placement = genome.placements.get(macro.output_node_ids[0], Placement())
        low, high = placement.active(genome.contract)
        for index in range(low, high):
            macros.append(
                MacroGene(
                    macro.ref,
                    tuple(ids[node, index] for node in macro.input_node_ids),
                    tuple(ids[node, index] for node in macro.output_node_ids),
                    len(connections) + len(macros),
                    macro.trainable,
                )
            )
    return Genome(nodes, connections, macros, genome.refine_steps, dict(genome.operator_rates))


def spatial_topology(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload["spatial"]
    bindings = data["bindings"]
    return {
        "spatial": {
            "version": data["version"],
            "contract": data["contract"],
            "placements": data["placements"],
            "bindings": sorted(
                (edge["in"], edge["out"], bool(edge.get("recurrent")), json.dumps(bindings[str(edge["innovation"])], sort_keys=True)) for edge in payload["connections"]
            ),
            "groups": sorted(sorted(nodes) for nodes in data.get("groups", {}).values()),
        }
    }


REPRESENTATION.register("spatial")(
    RepresentationCodec(
        SpatialGenome.from_payload,
        SpatialGenome.to_payload,
        SpatialNet,
        lambda genome: (genome.contract.input_width, genome.contract.output_width),
        expand_spatial,
        spatial_topology,
    )
)


def circuit_evidence(genome: SpatialGenome, *, limit: int = 4096) -> dict[str, Any]:
    """
    Extract one real placement of each learned circuit for scalar motif mining.

    Boundary values become ordered scalar ports. Repetition is not independent
    evidence: the library entry and its original search lineage still own every
    occurrence. No candidate weights or task answers participate in induction.
    """
    assert genome.contract is not None
    selected: dict[int, int] = {}
    for node in genome.hidden_ids + genome.output_ids:
        if node in genome.macro_output_ids:
            continue
        low, high = genome.placements.get(node, Placement()).active(genome.contract) if node in genome.hidden_ids else (0, math.prod(genome.node_shape(node)))
        if high > low:
            selected[node] = low
    nodes: dict[int, NodeGene] = {}
    ids: dict[tuple[int, int], int] = {}
    for node, position in selected.items():
        identifier = len(nodes)
        ids[node, position] = identifier
        nodes[identifier] = replace(genome.nodes[node], id=identifier, coordinate=None)
    connections = []
    for edge in genome.enabled_connections():
        if edge.out_id not in selected:
            continue
        target = selected[edge.out_id]
        rule = genome.bindings[edge.innovation]
        offset = target - rule.target_start
        if offset < 0 or offset % rule.target_stride or (rule.target_count and offset // rule.target_stride >= rule.target_count):
            continue
        local = replace(rule, target_start=target, target_count=1)
        for sources, targets, _ in local.indices(genome.node_shape(edge.in_id), genome.node_shape(edge.out_id), chunk_size=limit):
            for source in sources.tolist():
                if len(connections) + len(nodes) >= limit:
                    return genome_to_dict(Genome())  # incomplete evidence cannot promote a motif
                if genome.nodes[edge.in_id].kind is NodeKind.HIDDEN:
                    low, high = genome.placements.get(edge.in_id, Placement()).active(genome.contract)
                    if not low <= source < high:
                        continue
                address = edge.in_id, source
                if address not in ids:
                    identifier = len(nodes)
                    ids[address] = identifier
                    kind = NodeKind.BIAS if genome.nodes[edge.in_id].kind is NodeKind.BIAS else NodeKind.INPUT
                    nodes[identifier] = NodeGene(identifier, kind, "identity")
                connections.append(ConnectionGene(ids[address], ids[edge.out_id, target], 1.0, True, len(connections), edge.recurrent))
    return genome_to_dict(Genome(nodes, connections))
