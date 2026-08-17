import json

import pytest

from versal.evolution.genome import genome_from_dict, genome_to_dict
from versal.results import render_speciation, write_stats
from versal.substrate import decode


def test_genome_dict_round_trip_preserves_accuracy(solving_genome, xor_adapter):
    restored = genome_from_dict(genome_to_dict(solving_genome))
    module = decode(restored, xor_adapter.n_inputs, xor_adapter.n_outputs)
    assert xor_adapter.evaluate(module)["query_accuracy"] == 1.0


def test_write_stats(tmp_path):
    stats_path = write_stats(tmp_path, {"champion": {"query_accuracy": 1.0}})
    assert json.loads(stats_path.read_text())["champion"]["query_accuracy"] == 1.0


def test_render_speciation_writes_png(tmp_path):
    import matplotlib.image as mpimg

    history = [{0: 64}, {0: 40, 1: 24}, {0: 30, 1: 20, 2: 14}, {1: 32, 2: 32}]
    image_path = render_speciation(tmp_path, history, title="species")
    assert image_path.exists() and image_path.stat().st_size > 0
    assert mpimg.imread(image_path).shape[:2] == (1800, 3300)


def test_render_speciation_handles_empty(tmp_path):
    image_path = render_speciation(tmp_path, [], title="species")
    assert image_path.exists()


def test_render_speciation_preserves_destination_when_save_fails(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    target = tmp_path / "speciation.png"
    target.write_bytes(b"old chart")

    def fail_savefig(self, path, *args, **kwargs):
        del self, args, kwargs
        path.write_bytes(b"partial chart")
        raise RuntimeError("save failed")

    monkeypatch.setattr(Figure, "savefig", fail_savefig)
    with pytest.raises(RuntimeError, match="save failed"):
        render_speciation(tmp_path, [{0: 1}], title="species")

    assert target.read_bytes() == b"old chart"
    assert not list(tmp_path.glob(".speciation-*.tmp.png"))
