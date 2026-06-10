import json

from ardevo.evolution.genome import genome_from_dict, genome_to_dict
from ardevo.results import render_network, render_speciation, run_directory, write_model, write_stats
from ardevo.substrate import decode


def test_genome_dict_round_trip_preserves_accuracy(solving_genome, xor_adapter):
    restored = genome_from_dict(genome_to_dict(solving_genome))
    module = decode(restored, xor_adapter.n_inputs, xor_adapter.n_outputs)
    assert xor_adapter.evaluate(module)["query_accuracy"] == 1.0


def test_run_directory_name_format(tmp_path):
    directory = run_directory("20260609_193733", 0.243, 0.333, 1.038, root=str(tmp_path))
    assert directory.is_dir()
    assert directory.name == "20260609_193733_fit-0.243_acc-0.333_loss-1.038"


def test_write_stats_and_model(tmp_path):
    directory = run_directory("20260609_000000", 1.0, 1.0, 0.0, root=str(tmp_path))
    stats_path = write_stats(directory, {"champion": {"query_accuracy": 1.0}})
    model_path = write_model(directory, {"genome": {"nodes": [], "connections": []}})
    assert json.loads(stats_path.read_text())["champion"]["query_accuracy"] == 1.0
    assert "genome" in json.loads(model_path.read_text())


def test_render_network_writes_png(tmp_path, solving_genome):
    directory = run_directory("20260609_000001", 1.0, 1.0, 0.0, root=str(tmp_path))
    image_path = render_network(directory, solving_genome, title="test")
    assert image_path.exists() and image_path.stat().st_size > 0


def test_render_speciation_writes_png(tmp_path):
    directory = run_directory("20260609_000002", 1.0, 1.0, 0.0, root=str(tmp_path))
    history = [{0: 64}, {0: 40, 1: 24}, {0: 30, 1: 20, 2: 14}, {1: 32, 2: 32}]
    image_path = render_speciation(directory, history, title="species")
    assert image_path.exists() and image_path.stat().st_size > 0


def test_render_speciation_handles_empty(tmp_path):
    directory = run_directory("20260609_000003", 0.0, 0.0, 0.0, root=str(tmp_path))
    image_path = render_speciation(directory, [], title="species")
    assert image_path.exists()
