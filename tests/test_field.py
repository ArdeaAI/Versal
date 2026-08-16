from __future__ import annotations

import torch

from versal.dataset.icarus import Axis, Field, Task, TaskKind, TaskMeta, ValueType
from versal.field import field_contract, field_feature_width, gather_local_multiscale_v1
from versal.library import MODULE, ModuleLibrary, structural_fingerprint


def _field(data: torch.Tensor, axes: tuple[Axis, ...], mask: torch.Tensor | None = None) -> Field:
    return Field(data, axes, ValueType.CONTINUOUS, None, None, mask)


def _task(query_size: int = 9) -> Task:
    axes = (Axis.CHANNEL, Axis.HEIGHT, Axis.WIDTH)
    support = [(_field(torch.arange(2 * 4 * 5).reshape(2, 4, 5), axes), _field(torch.zeros(1, 4, 5), axes))]
    query = [(_field(torch.zeros(2, query_size, query_size), axes), _field(torch.zeros(1, query_size, query_size), axes))]
    return Task(TaskMeta(999, TaskKind.MAP, "arbitrary"), support, query)


def test_contract_is_support_only_and_symbolic() -> None:
    assert field_contract(_task(9)) == field_contract(_task(101))
    contract = field_contract(_task())
    assert contract is not None and "height" in contract.to_dict()["spatial"]


def test_semantic_axis_permutation_and_lazy_feature_width() -> None:
    task = _task()
    contract = field_contract(task)
    assert contract is not None and contract.input_channels == 2
    original = task.support[0][0]
    permuted = _field(original.data.permute(1, 2, 0), (Axis.HEIGHT, Axis.WIDTH, Axis.CHANNEL))
    sites = torch.tensor([[0, 0], [2, 3]])
    left = gather_local_multiscale_v1(original, sites)
    right = gather_local_multiscale_v1(permuted, sites)
    assert left.shape == (2, field_feature_width(2))
    torch.testing.assert_close(left, right)


def test_rejects_spatial_mismatch_and_time() -> None:
    task = _task()
    inp, out = task.support[0]
    mismatch = _field(torch.zeros(1, 3, 5), out.axes)
    assert field_contract(Task(task.meta, [(inp, mismatch)], [])) is None
    temporal = _field(torch.zeros(2, 4, 5), (Axis.TIME, Axis.HEIGHT, Axis.WIDTH))
    assert field_contract(Task(task.meta, [(temporal, out)], [])) is None


def test_field_payload_cross_resolution_round_trip_and_unknown_version_fails(tmp_path) -> None:
    from versal.evolution.init import minimal
    from versal.field import decode_field_payload, field_payload, payload_field_contract

    contract = field_contract(_task())
    assert contract is not None
    genome = minimal(field_feature_width(contract.input_channels), contract.output_channels, rng=__import__("random").Random(0))
    payload = field_payload(genome, contract)
    library = ModuleLibrary(tmp_path / "lib")
    io = {
        "inputs": [{"signature": "CONTINUOUS|C,H,W", "width": 40}],
        "output": {"signature": "CONTINUOUS|C,H,W", "width": 20},
    }
    key = library.add(entry_type=MODULE, payload=payload, io=io, provenance={})
    assert library.query_field(contract)[0].key == key
    module, restored = decode_field_payload(library.load(key).payload, library=library)
    assert module is not None and restored == contract == payload_field_contract(payload)
    altered = dict(payload)
    altered["field_template"] = dict(payload["field_template"], version="future")
    import pytest

    with pytest.raises(ValueError):
        decode_field_payload(altered)
    assert structural_fingerprint(MODULE, payload) != structural_fingerprint(MODULE, {k: v for k, v in payload.items() if k != "field_template"})


def test_native_larger_resolution_prediction_and_mask_padding_invariance() -> None:
    import random

    from versal.evolution.init import minimal
    from versal.field import FieldAdapter, deterministic_sites, encode_sites, predict_field, valid_sites

    task = _task(11)
    contract = field_contract(task)
    assert contract is not None
    sites = deterministic_sites(valid_sites(task.support), 8, salt="same")
    encoded = encode_sites(task, sites, contract, chunk_size=3)
    adapter = FieldAdapter(encoded, encoded, contract, max_inline_depth=4)
    module = adapter.decode(minimal(adapter.n_inputs, adapter.n_outputs, rng=random.Random(0)))
    prediction = predict_field(module, task.query[0][0], contract, chunk_size=17)
    assert prediction.shape == (1, 11, 11)

    field = task.support[0][0]
    padding_mask = torch.zeros_like(field.data, dtype=torch.bool)
    padding_mask[:, -1, -1] = True
    masked = _field(field.data.clone(), field.axes, padding_mask)
    changed = _field(masked.data.clone(), masked.axes, masked.mask)
    changed.data[:, -1, -1] = 1e9
    sites_tensor = torch.tensor([[0, 0], [1, 1]])
    torch.testing.assert_close(gather_local_multiscale_v1(masked, sites_tensor), gather_local_multiscale_v1(changed, sites_tensor))


def test_chunked_features_match_dense_reference_including_gradients() -> None:
    axes = (Axis.CHANNEL, Axis.HEIGHT, Axis.WIDTH)
    dense_data = torch.randn(2, 5, 6, requires_grad=True)
    chunked_data = dense_data.detach().clone().requires_grad_()
    sites = torch.cartesian_prod(torch.arange(5), torch.arange(6))
    dense = gather_local_multiscale_v1(_field(dense_data, axes), sites)
    chunked = torch.cat([gather_local_multiscale_v1(_field(chunked_data, axes), chunk) for chunk in sites.split(7)])
    torch.testing.assert_close(dense, chunked)
    dense.square().sum().backward()
    chunked.square().sum().backward()
    torch.testing.assert_close(dense_data.grad, chunked_data.grad)


def test_psicov_shaped_feature_generation_never_builds_flat_io_product(monkeypatch) -> None:
    # 64 channels × 128² input and a 128² output would make a flat task-wide graph enormous.
    # Feature generation is instead bounded by sites × the fixed channel vocabulary.
    field = _field(torch.zeros(64, 128, 128), (Axis.CHANNEL, Axis.HEIGHT, Axis.WIDTH))
    sites = torch.tensor([[0, 0], [64, 64], [127, 127]])
    original_zeros = torch.zeros

    def guarded_zeros(*shape, **kwargs):
        dimensions = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        assert all(int(dimension) < 10_000_000 for dimension in dimensions)
        return original_zeros(*shape, **kwargs)

    monkeypatch.setattr(torch, "zeros", guarded_zeros)
    features = gather_local_multiscale_v1(field, sites)
    assert features.shape == (3, field_feature_width(64))


def test_cpu_cuda_feature_parity_when_available() -> None:
    if not torch.cuda.is_available():
        return
    axes = (Axis.CHANNEL, Axis.HEIGHT, Axis.WIDTH)
    data = torch.randn(2, 5, 6)
    sites = torch.tensor([[0, 0], [2, 3], [4, 5]])
    cpu = gather_local_multiscale_v1(_field(data, axes), sites)
    cuda = gather_local_multiscale_v1(_field(data.cuda(), axes), sites.cuda()).cpu()
    torch.testing.assert_close(cpu, cuda, rtol=1e-5, atol=1e-6)


def test_field_module_executes_inside_nested_compositions(tmp_path) -> None:
    import random

    from versal.evolution.composition import (
        AssemblyContext,
        CompEdgeGene,
        CompNodeGene,
        CompNodeKind,
        CompositionGenome,
        IndexRun,
        PortMap,
        assemble,
        comp_to_dict,
    )
    from versal.evolution.init import minimal
    from versal.field import field_payload

    contract = field_contract(_task())
    assert contract is not None
    width = field_feature_width(contract.input_channels)
    genome = minimal(width, contract.output_channels, rng=random.Random(0))
    library = ModuleLibrary(tmp_path / "lib")
    site_io = {"inputs": [{"signature": "SITE", "width": width}], "output": {"signature": "SITE", "width": 1}}
    module_key = library.add(entry_type=MODULE, payload=field_payload(genome, contract), io=site_io, provenance={})

    def wrapper(reference: str) -> CompositionGenome:
        nodes = {
            0: CompNodeGene(0, CompNodeKind.INPUT, "SITE", 0, width),
            1: CompNodeGene(1, CompNodeKind.MODULE, reference, width, 1, trainable=False),
            2: CompNodeGene(2, CompNodeKind.OUTPUT, "target", 1, 0),
        }
        edges = [
            CompEdgeGene(0, 1, True, 0, (), port_map=PortMap((IndexRun(0, 0, width),))),
            CompEdgeGene(1, 2, True, 1, (), port_map=PortMap((IndexRun(0, 0, 1),))),
        ]
        return CompositionGenome(nodes, edges)

    inner = wrapper(f"library:{module_key}")
    inner_key = library.add(
        entry_type="composition",
        payload=comp_to_dict(inner),
        io={"inputs": [{"signature": "SITE", "width": width}], "output": {"signature": "SITE", "width": 1}},
        provenance={},
        level=2,
    )
    outer = wrapper(f"library:{inner_key}")
    net = assemble(outer, AssemblyContext(bank_columns={"SITE": range(width)}, library=library), width)
    assert net(torch.zeros(3, width)).shape == (3, 1)

    from versal.rendering import build_entry_spec
    from versal.routing import build_vertex

    entry = library.load(module_key)
    vertex = build_vertex(entry, library)
    assert vertex is not None and vertex.in_width == width and vertex.out_width == 1
    rendered = build_entry_spec(entry)
    assert any("repeated field H×W" in container.label for container in rendered.containers)

    library.retire(module_key)
    # The live nested composition protects its retired field dependency from GC.
    assert module_key not in library.collect_garbage(dry_run=True)
