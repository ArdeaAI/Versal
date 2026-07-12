import hashlib
import json
import tomllib

from ardevo.tools.runtime_inventory import DEFAULT_MANIFEST, RUN_PATHS, build_inventory, check_manifest
from ardevo.utils.config import Config


def test_all_features_config_is_the_canonical_default() -> None:
    assert Config.DEFAULT_CONFIG.name == "orchestrated_overmind_all_features.toml"
    assert Config().current["config_path"] == str(Config.DEFAULT_CONFIG)


def test_checked_in_runtime_inventory_is_current() -> None:
    assert check_manifest()
    assert json.loads(DEFAULT_MANIFEST.read_text()) == build_inventory()


def test_preflight_only_changes_its_declared_operational_scope() -> None:
    canonical = tomllib.loads(Config.DEFAULT_CONFIG.read_text())
    preflight = tomllib.loads((Config.PROJECT_ROOT / "configs" / "preflight.toml").read_text())
    assert preflight["evolution"]["composition"]["max_initial_glue_values"] == 5_000_000

    preflight["run"]["experiment"] = canonical["run"]["experiment"]
    preflight["schedule"]["tasks_per_rung"] = canonical["schedule"]["tasks_per_rung"]
    preflight["orchestrator"]["tasks"] = canonical["orchestrator"]["tasks"]
    preflight["orchestrator"]["library_dir"] = canonical["orchestrator"]["library_dir"]
    assert preflight == canonical


def test_runtime_inventory_covers_live_surfaces() -> None:
    inventory = build_inventory()
    assert inventory["canonical_config"] == "configs/orchestrated_overmind_all_features.toml"
    assert inventory["canonical_config_sha256"] == hashlib.sha256(Config.DEFAULT_CONFIG.read_bytes()).hexdigest()
    assert {"configs/orchestrated_overmind.toml", "configs/orchestrated_overmind_all_features.toml"} <= set(inventory["available_configs"])
    assert "orchestrator.direct.train.kind" in inventory["config_keys"]
    assert {"config_path", "config_sha256", "orchestrator"} <= set(inventory["runtime_config_keys"])
    assert "gradient_scheduled" in inventory["registries"]["train"]
    assert "untie_motif_weights" in inventory["registries"]["mutation"]
    assert inventory["console_scripts"]["app"] == "ardevo.main:main"
    assert inventory["console_scripts"]["runtime_inventory"] == "ardevo.tools.runtime_inventory:main"
    assert {row["path"] for row in inventory["run_paths"]} == {row["path"] for row in RUN_PATHS}
