"""Recursive TOML overlays: deterministic merge semantics and auditable provenance."""

import hashlib
import json
from pathlib import Path

import pytest

from ardevo.utils.config import Config

_FROZEN_PREFLIGHT_SHA256 = "bc023654a4c07cb438b2e3094b90b211a9d2a89c4fdc42405b8d46c491b9f18c"


def _effective_hash(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def test_recursive_relative_extends_deep_merges_and_tracks_sources(tmp_path: Path) -> None:
    base = tmp_path / "base.toml"
    base.write_text(
        """[run]
seed = 1
machine = "local"
[run.duration]
generations = 50
[orchestrator]
tasks = 8
evolve = ["direct", "composition"]
[orchestrator.refine]
budget_k = 24
decay = 0.5
[resources]
gpus = 2
[archive]
upload = true
[campaign]
seeds = [0, 1, 2]
"""
    )
    layer = tmp_path / "layers" / "middle.toml"
    layer.parent.mkdir()
    layer.write_text(
        """extends = "../base.toml"
[run]
seed = 2
[orchestrator.refine]
budget_k = 12
"""
    )
    leaf = tmp_path / "leaf.toml"
    leaf.write_text(
        """extends = "layers/middle.toml"
[orchestrator]
evolve = ["direct"]
[orchestrator.refine]
stall_generations = 4
"""
    )

    config = Config(leaf).current

    assert config["seed"] == 2
    assert config["generations"] == 50
    assert config["orchestrator"]["tasks"] == 8
    assert config["orchestrator"]["evolve"] == ["direct"]
    assert config["orchestrator"]["refine"] == {"budget_k": 12, "decay": 0.5, "stall_generations": 4}
    assert config["resources"] == {"gpus": 2}
    assert config["archive"] == {"upload": True}
    assert config["campaign"] == {"seeds": [0, 1, 2]}
    assert [Path(source["path"]).name for source in config["config_sources"]] == ["base.toml", "middle.toml", "leaf.toml"]
    assert [source["sha256"] for source in config["config_sources"]] == [hashlib.sha256(path.read_bytes()).hexdigest() for path in (base, layer, leaf)]

    effective, _sources = Config._load_config_tree(leaf)
    assert config["config_effective_sha256"] == _effective_hash(effective)
    assert config["config_sha256"] == hashlib.sha256(leaf.read_bytes()).hexdigest()


def test_multiple_parents_merge_left_to_right_before_child(tmp_path: Path) -> None:
    (tmp_path / "first.toml").write_text("[orchestrator]\ntasks = 4\n[orchestrator.refine]\nbudget_k = 2\ndecay = 0.5\n")
    (tmp_path / "second.toml").write_text("[orchestrator]\ntasks = 6\n[orchestrator.refine]\nbudget_k = 3\n")
    child = tmp_path / "child.toml"
    child.write_text('extends = ["first.toml", "second.toml"]\n[orchestrator.refine]\ndecay = 0.25\n')

    config = Config(child).current

    assert config["orchestrator"] == {"tasks": 6, "refine": {"budget_k": 3, "decay": 0.25}}


def test_extends_cycle_reports_the_chain(tmp_path: Path) -> None:
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    first.write_text('extends = "second.toml"\n')
    second.write_text('extends = "first.toml"\n')

    with pytest.raises(ValueError, match=r"extends cycle: .*first\.toml.*second\.toml.*first\.toml"):
        Config(first)


def test_missing_parent_names_declaring_config(tmp_path: Path) -> None:
    child = tmp_path / "child.toml"
    child.write_text('extends = "missing.toml"\n')

    with pytest.raises(FileNotFoundError, match=r"missing\.toml.*declared by.*child\.toml"):
        Config(child)


def test_effective_hash_is_format_independent_and_changes_with_merged_values(tmp_path: Path) -> None:
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    first.write_text("[run]\nseed=3\n[orchestrator]\ntasks=2\n")
    second.write_text("# same values, different bytes\n[run]\nseed = 3\n\n[orchestrator]\ntasks = 2\n")
    changed = tmp_path / "changed.toml"
    changed.write_text("[run]\nseed=4\n[orchestrator]\ntasks=2\n")

    first_config = Config(first).current
    second_config = Config(second).current
    changed_config = Config(changed).current

    assert first_config["config_sha256"] != second_config["config_sha256"]
    assert first_config["config_effective_sha256"] == second_config["config_effective_sha256"]
    assert first_config["config_effective_sha256"] != changed_config["config_effective_sha256"]


def test_full_cluster_is_a_frozen_preflight_overlay_with_declared_scale_and_archival() -> None:
    preflight_path = Config.PROJECT_ROOT / "configs" / "preflight.toml"
    full_path = Config.PROJECT_ROOT / "configs" / "full_cluster.toml"
    assert hashlib.sha256(preflight_path.read_bytes()).hexdigest() == _FROZEN_PREFLIGHT_SHA256

    config = Config(full_path).current

    assert [Path(source["path"]).name for source in config["config_sources"]] == ["preflight.toml", "full_cluster.toml"]
    assert config["machine_env"] == "ClusterCUDA"
    assert config["schedule"]["tasks_per_rung"] == 20
    assert config["orchestrator"]["tasks"] == 360
    assert config["orchestrator"]["budgets"] == {"depth0": 2000, "depth1": 1000, "depth2": 500, "depth3": 250, "depth4": 125}
    assert config["evolution"]["composition"]["max_inline_depth"] == 8
    assert config["evolution"]["composition"]["glue_storage"] == "f32"
    assert config["resources"]["mode"] == "adaptive"
    assert config["archive"]["backend"] == "s3" and config["archive"]["snapshot_every_tasks"] == 1
    assert config["campaign"] == {"seeds": [0, 1, 2], "cold_library": True}
