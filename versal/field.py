"""Resolution-independent aligned spatial field contracts and lazy feature gathering."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

import torch

from versal.dataset.icarus import Axis, EncodedTask, Field, FieldDescriptor, Level0Encoder, Task, ValueType, as_logits, model_output_features
from versal.evaluation import split_metrics_from_raw
from versal.evolution.genome import Genome, genome_from_dict
from versal.substrate import SubstrateModule, decode_module

if TYPE_CHECKING:
    from versal.library import ModuleLibrary

FIELD_TEMPLATE_VERSION = "local_multiscale_v1"


@dataclass(frozen=True)
class FieldContract:
    version: str
    input_channels: int
    output_channels: int
    input_value_type: str
    output_value_type: str
    input_nonspatial: tuple[tuple[str, int], ...]
    output_nonspatial: tuple[tuple[str, int], ...]
    input_n_classes: int | None = None
    output_n_classes: int | None = None
    input_value_range: tuple[float, float] | None = None
    output_value_range: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "input_value_type": self.input_value_type,
            "output_value_type": self.output_value_type,
            "input_nonspatial": [list(item) for item in self.input_nonspatial],
            "output_nonspatial": [list(item) for item in self.output_nonspatial],
            "input_n_classes": self.input_n_classes,
            "output_n_classes": self.output_n_classes,
            "input_value_range": self.input_value_range,
            "output_value_range": self.output_value_range,
            "spatial": {"height": "H", "width": "W"},
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FieldContract":
        if value.get("version") != FIELD_TEMPLATE_VERSION:
            raise ValueError(f"unsupported field-template version {value.get('version')!r}")
        spatial = value.get("spatial")
        if spatial != {"height": "H", "width": "W"}:
            raise ValueError("field-template spatial bindings are missing or invalid")
        input_range = value.get("input_value_range")
        output_range = value.get("output_value_range")
        return cls(
            version=FIELD_TEMPLATE_VERSION,
            input_channels=int(value["input_channels"]),
            output_channels=int(value["output_channels"]),
            input_value_type=str(value["input_value_type"]),
            output_value_type=str(value["output_value_type"]),
            input_nonspatial=tuple((str(axis), int(size)) for axis, size in value["input_nonspatial"]),
            output_nonspatial=tuple((str(axis), int(size)) for axis, size in value["output_nonspatial"]),
            input_n_classes=int(value["input_n_classes"]) if value.get("input_n_classes") is not None else None,
            output_n_classes=int(value["output_n_classes"]) if value.get("output_n_classes") is not None else None,
            input_value_range=(float(input_range[0]), float(input_range[1])) if input_range is not None else None,
            output_value_range=(float(output_range[0]), float(output_range[1])) if output_range is not None else None,
        )

    @property
    def identity(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


def _layout(field: Field) -> tuple[int, int, tuple[tuple[str, int], ...]] | None:
    if field.axes.count(Axis.HEIGHT) != 1 or field.axes.count(Axis.WIDTH) != 1 or Axis.TIME in field.axes:
        return None
    h = field.axes.index(Axis.HEIGHT)
    w = field.axes.index(Axis.WIDTH)
    nonspatial = tuple((axis.value, int(field.data.shape[index])) for index, axis in enumerate(field.axes) if index not in (h, w))
    return h, w, nonspatial


def field_contract(task: Task) -> FieldContract | None:
    """Return support-only eligibility. Query contents, task names, and rung metadata are ignored."""

    first_input, first_output = task.support[0]
    in_layout, out_layout = _layout(first_input), _layout(first_output)
    if in_layout is None or out_layout is None:
        return None
    _, _, input_nonspatial = in_layout
    _, _, output_nonspatial = out_layout
    for input_field, output_field in task.support:
        input_layout, output_layout = _layout(input_field), _layout(output_field)
        if input_layout is None or output_layout is None:
            return None
        ih, iw, ins = input_layout
        oh, ow, outs = output_layout
        if ins != input_nonspatial or outs != output_nonspatial:
            return None
        if (input_field.data.shape[ih], input_field.data.shape[iw]) != (output_field.data.shape[oh], output_field.data.shape[ow]):
            return None
        if input_field.value_type != first_input.value_type or output_field.value_type != first_output.value_type:
            return None
        if input_field.n_classes != first_input.n_classes or output_field.n_classes != first_output.n_classes:
            return None
        if input_field.value_range != first_input.value_range or output_field.value_range != first_output.value_range:
            return None
    input_channels = _product(size for _axis, size in input_nonspatial)
    output_channels = _product(size for _axis, size in output_nonspatial)
    return FieldContract(
        FIELD_TEMPLATE_VERSION,
        input_channels,
        output_channels,
        first_input.value_type.value,
        first_output.value_type.value,
        input_nonspatial,
        output_nonspatial,
        first_input.n_classes,
        first_output.n_classes,
        first_input.value_range,
        first_output.value_range,
    )


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _channel_first(field: Field) -> tuple[torch.Tensor, torch.Tensor]:
    h_axis, w_axis = field.axes.index(Axis.HEIGHT), field.axes.index(Axis.WIDTH)
    other = [index for index in range(field.data.ndim) if index not in (h_axis, w_axis)]
    order = [*other, h_axis, w_axis]
    data = field.data.permute(order).reshape(-1, field.data.shape[h_axis], field.data.shape[w_axis]).to(torch.float32)
    if field.value_type is ValueType.CONTINUOUS and field.value_range is not None and field.value_range[1] > field.value_range[0]:
        low, high = field.value_range
        data = (data - low) / (high - low)
    valid = torch.ones_like(data, dtype=torch.bool) if field.mask is None else ~field.mask.permute(order).reshape_as(data)
    return data, valid


def gather_local_multiscale_v1(field: Field, positions: torch.Tensor, *, deadline: float | None = None) -> torch.Tensor:
    """Gather the fixed v1 bank for ``[N, 2]`` (row, column) sites without im2col."""

    data, valid = _channel_first(field)
    channels, height, width = data.shape
    positions = positions.to(device=data.device, dtype=torch.long)
    rows, cols = positions[:, 0], positions[:, 1]
    features: list[torch.Tensor] = []
    indicators: list[torch.Tensor] = []
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    offsets.extend((dy * dilation, dx * dilation) for dilation in (2, 4, 8) for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)))
    for dy, dx in offsets:
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("deadline expired during field offset gathering")
        rr, cc = rows + dy, cols + dx
        inside = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        safe_r, safe_c = rr.clamp(0, height - 1), cc.clamp(0, width - 1)
        present = valid[:, safe_r, safe_c].T & inside[:, None]
        features.append(torch.where(present, data[:, safe_r, safe_c].T, 0.0))
        indicators.append(present.to(torch.float32).mean(dim=1, keepdim=True))
    for radius in (1, 3, 7):
        sums = torch.zeros((len(positions), channels), device=data.device)
        maxima = torch.full_like(sums, -torch.inf)
        counts = torch.zeros_like(sums)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if deadline is not None and time.perf_counter() >= deadline:
                    raise TimeoutError("deadline expired during field pooling")
                rr, cc = rows + dy, cols + dx
                inside = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                safe_r, safe_c = rr.clamp(0, height - 1), cc.clamp(0, width - 1)
                present = valid[:, safe_r, safe_c].T & inside[:, None]
                values = data[:, safe_r, safe_c].T
                sums += torch.where(present, values, 0.0)
                maxima = torch.maximum(maxima, torch.where(present, values, -torch.inf))
                counts += present
        features.extend((sums / counts.clamp_min(1), torch.where(counts > 0, maxima, 0.0)))
        indicators.append((counts / float((2 * radius + 1) ** 2)).mean(dim=1, keepdim=True))
    valid_f = valid.to(data.dtype)
    count = valid_f.sum(dim=(1, 2)).clamp_min(1)
    mean = (data * valid_f).sum(dim=(1, 2)) / count
    variance = ((data - mean[:, None, None]) ** 2 * valid_f).sum(dim=(1, 2)) / count
    features.extend((mean.expand(len(positions), -1), variance.sqrt().expand(len(positions), -1)))
    hden, wden = max(1, height - 1), max(1, width - 1)
    coordinates = torch.stack((rows / hden, cols / wden, rows / hden, (height - 1 - rows) / hden, cols / wden, (width - 1 - cols) / wden), dim=1)
    return torch.cat([*features, coordinates.to(data.dtype), *indicators], dim=1)


def field_feature_width(input_channels: int) -> int:
    # 33 offsets + 6 pooled + 2 global values per channel; 6 geometry; 36 validity/coverage values.
    return input_channels * 41 + 42


def field_payload(genome: Genome, contract: FieldContract) -> dict[str, Any]:
    """An ordinary genome payload with optional representation metadata."""

    from versal.evolution.genome import genome_to_dict

    payload = genome_to_dict(genome)
    payload["field_template"] = contract.to_dict()
    return payload


def payload_field_contract(payload: dict[str, Any]) -> FieldContract | None:
    metadata = payload.get("field_template")
    return None if metadata is None else FieldContract.from_dict(metadata)


@dataclass(frozen=True)
class FieldSite:
    pair: int
    row: int
    column: int


def _target_at(field: Field, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    data, valid = _channel_first(field)
    rows, columns = positions[:, 0].long(), positions[:, 1].long()
    return data[:, rows, columns].T, ~valid[:, rows, columns].T


def valid_sites(pairs: list[tuple[Field, Field]]) -> list[FieldSite]:
    sites: list[FieldSite] = []
    for pair_index, (_input, target) in enumerate(pairs):
        _data, valid = _channel_first(target)
        spatial = valid.any(dim=0).nonzero()
        sites.extend(FieldSite(pair_index, int(row), int(column)) for row, column in spatial.tolist())
    return sites


def deterministic_sites(sites: list[FieldSite], count: int, *, salt: str) -> list[FieldSite]:
    if count <= 0 or len(sites) <= count:
        return list(sites)
    ranked = sorted(sites, key=lambda site: hashlib.sha1(f"{salt}:{site.pair}:{site.row}:{site.column}".encode()).digest())
    return ranked[:count]


def encode_sites(
    task: Task, sites: list[FieldSite], contract: FieldContract, *, chunk_size: int = 32768, deadline: float | None = None
) -> EncodedTask:
    """Gather only nominated sites. No task-wide im2col tensor is constructed."""

    feature_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    by_pair: dict[int, list[FieldSite]] = {}
    for site in sites:
        by_pair.setdefault(site.pair, []).append(site)
    for pair_index, selected in sorted(by_pair.items()):
        input_field, target_field = task.support[pair_index]
        for start in range(0, len(selected), max(1, chunk_size)):
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("deadline expired during field feature preparation")
            chunk = selected[start : start + chunk_size]
            positions = torch.tensor([(site.row, site.column) for site in chunk], dtype=torch.long)
            feature_parts.append(gather_local_multiscale_v1(input_field, positions, deadline=deadline))
            target, mask = _target_at(target_field, positions)
            target_parts.append(target)
            mask_parts.append(mask)
    features = torch.cat(feature_parts) if feature_parts else torch.empty((0, field_feature_width(contract.input_channels)))
    targets = torch.cat(target_parts) if target_parts else torch.empty((0, contract.output_channels))
    masks = torch.cat(mask_parts) if mask_parts else torch.empty_like(targets, dtype=torch.bool)
    target_type = ValueType(contract.output_value_type)
    if target_type in {ValueType.CATEGORICAL, ValueType.ORDINAL}:
        targets = targets.long()
    input_descriptor = FieldDescriptor((Axis.EXTRA,), ValueType.CONTINUOUS, None, None)
    target_descriptor = FieldDescriptor((Axis.EXTRA,), target_type, contract.output_n_classes, contract.output_value_range)
    return EncodedTask((features, input_descriptor), (targets, masks, target_descriptor), None, None)


class FieldAdapter:
    """Site-level adapter: training samples and audit samples are separate deterministic rails."""

    def __init__(self, training: EncodedTask, audit: EncodedTask, contract: FieldContract, *, max_inline_depth: int, library: "ModuleLibrary | None" = None) -> None:
        self.encoded = training
        self.audit = audit
        self.contract = contract
        self.encoder = Level0Encoder(field_feature_width(contract.input_channels))
        self.n_inputs = field_feature_width(contract.input_channels)
        self.n_outputs = model_output_features(audit.support_target[2], contract.output_channels)
        self.max_inline_depth = max_inline_depth
        self.library = library

    def decode(self, genome: Genome) -> SubstrateModule:
        from versal.library import macro_resolver

        resolver = macro_resolver(self.library) if self.library is not None else None
        return decode_module(genome, self.n_inputs, self.n_outputs, macro_resolver=resolver, max_inline_depth=self.max_inline_depth)

    def evaluate(self, module: SubstrateModule) -> dict[str, float]:
        from versal.evaluation import evaluate

        metrics = evaluate(module, self.audit, self.encoder)
        return {
            **metrics,
            "sampled_support_accuracy": metrics["support_accuracy"],
            "sampled_support_loss": metrics["support_loss"],
        }


def evaluate_field_module(
    module: SubstrateModule,
    task: Task,
    contract: FieldContract,
    *,
    split: str = "support",
    chunk_size: int = 32768,
    deadline: float | None = None,
) -> dict[str, float]:
    pairs = task.support if split == "support" else task.query
    encoder = Level0Encoder(field_feature_width(contract.input_channels))
    weighted_accuracy = 0.0
    weighted_loss = 0.0
    valid_total = 0
    sites_total = 0
    with torch.no_grad():
        for input_field, target_field in pairs:
            _target_data, target_valid = _channel_first(target_field)
            positions = target_valid.any(dim=0).nonzero()
            for start in range(0, len(positions), max(1, chunk_size)):
                if deadline is not None and time.perf_counter() >= deadline:
                    raise TimeoutError(f"deadline expired during field {split} verification")
                chunk = positions[start : start + chunk_size]
                features = gather_local_multiscale_v1(input_field, chunk, deadline=deadline)
                target, mask = _target_at(target_field, chunk)
                descriptor = FieldDescriptor((Axis.EXTRA,), ValueType(contract.output_value_type), contract.output_n_classes, contract.output_value_range)
                if descriptor.value_type in {ValueType.CATEGORICAL, ValueType.ORDINAL}:
                    target = target.long()
                raw = as_logits(module(features), descriptor, contract.output_channels)
                accuracy, loss = split_metrics_from_raw(raw, target, mask, descriptor, encoder)
                valid_count = int((~mask).sum())
                weighted_accuracy += accuracy * valid_count
                weighted_loss += loss * valid_count
                valid_total += valid_count
                sites_total += len(chunk)
    prefix = "support" if split == "support" else "query"
    return {
        f"{prefix}_accuracy": weighted_accuracy / max(1, valid_total),
        f"{prefix}_loss": weighted_loss / max(1, valid_total),
        f"{prefix}_sites": float(sites_total),
    }


def decode_field_payload(payload: dict[str, Any], *, library: "ModuleLibrary | None" = None, max_inline_depth: int = 8) -> tuple[SubstrateModule, FieldContract]:
    contract = payload_field_contract(payload)
    if contract is None:
        raise ValueError("payload is not a field template")
    from versal.library import macro_resolver

    genome = genome_from_dict(payload)
    descriptor = FieldDescriptor((Axis.EXTRA,), ValueType(contract.output_value_type), contract.output_n_classes, contract.output_value_range)
    module = decode_module(
        genome,
        field_feature_width(contract.input_channels),
        model_output_features(descriptor, contract.output_channels),
        macro_resolver=macro_resolver(library) if library is not None else None,
        max_inline_depth=max_inline_depth,
    )
    return module, contract


def predict_field(module: SubstrateModule, input_field: Field, contract: FieldContract, *, chunk_size: int = 32768, deadline: float | None = None) -> torch.Tensor:
    """Execute at the input's native H×W, including canvases larger than support."""

    h_axis, w_axis = input_field.axes.index(Axis.HEIGHT), input_field.axes.index(Axis.WIDTH)
    height, width = input_field.data.shape[h_axis], input_field.data.shape[w_axis]
    positions = torch.cartesian_prod(torch.arange(height), torch.arange(width))
    parts: list[torch.Tensor] = []
    descriptor = FieldDescriptor((Axis.EXTRA,), ValueType(contract.output_value_type), contract.output_n_classes, contract.output_value_range)
    encoder = Level0Encoder(field_feature_width(contract.input_channels))
    with torch.no_grad():
        for start in range(0, len(positions), max(1, chunk_size)):
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("deadline expired during field inference")
            features = gather_local_multiscale_v1(input_field, positions[start : start + chunk_size], deadline=deadline)
            raw = as_logits(module(features), descriptor, contract.output_channels)
            parts.append(encoder.decode(raw, descriptor).reshape(len(features), contract.output_channels))
    flat = torch.cat(parts).T
    nonspatial_shape = tuple(size for _axis, size in contract.output_nonspatial)
    return flat.reshape(*nonspatial_shape, height, width) if nonspatial_shape else flat.reshape(height, width)
