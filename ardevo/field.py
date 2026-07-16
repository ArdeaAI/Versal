"""Resolution-independent aligned spatial field contracts and lazy feature gathering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from ardevo.dataset.icarus import Axis, Field, Task

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "input_value_type": self.input_value_type,
            "output_value_type": self.output_value_type,
            "input_nonspatial": [list(item) for item in self.input_nonspatial],
            "output_nonspatial": [list(item) for item in self.output_nonspatial],
            "spatial": {"height": "H", "width": "W"},
        }


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
    valid = torch.ones_like(data, dtype=torch.bool) if field.mask is None else ~field.mask.permute(order).reshape_as(data)
    return data, valid


def gather_local_multiscale_v1(field: Field, positions: torch.Tensor) -> torch.Tensor:
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

