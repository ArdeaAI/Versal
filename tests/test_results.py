import json

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
    history = [{0: 64}, {0: 40, 1: 24}, {0: 30, 1: 20, 2: 14}, {1: 32, 2: 32}]
    image_path = render_speciation(tmp_path, history, title="species")
    assert image_path.exists() and image_path.stat().st_size > 0


def test_render_speciation_handles_empty(tmp_path):
    image_path = render_speciation(tmp_path, [], title="species")
    assert image_path.exists()
