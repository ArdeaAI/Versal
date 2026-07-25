import hashlib
import json

from ardevo.tools.runtime_inventory import DEFAULT_MANIFEST, RUN_PATHS, build_inventory, check_manifest
from ardevo.utils.config import Config

TOP_LEVEL_CONFIGS = {
    "configs/brute.toml",
    "configs/canary-lattice.toml",
    "configs/canary.toml",
    "configs/full.toml",
    "configs/full_cluster.toml",
    "configs/full_cluster-lattice.toml",
    "configs/preflight.toml",
    "configs/preflight-lattice.toml",
    "configs/smoke.toml",
}


def _effective(name: str) -> dict:
    config, _sources = Config._load_config_tree(Config.PROJECT_ROOT / "configs" / name)
    return config


def test_smoke_is_the_default_profile() -> None:
    assert Config.DEFAULT_CONFIG.name == "smoke.toml"
    config = Config().current
    assert config["config_path"] == str(Config.DEFAULT_CONFIG)
    assert config["schedule"]["tasks_per_rung"] == 1
    assert config["orchestrator"]["tasks"] == 18


def test_checked_in_runtime_inventory_is_current() -> None:
    assert check_manifest()
    assert json.loads(DEFAULT_MANIFEST.read_text()) == build_inventory()


def test_preflight_only_scales_the_canary_task_count() -> None:
    canary = _effective("canary.toml")
    preflight = _effective("preflight.toml")

    preflight["run"]["experiment"] = canary["run"]["experiment"]
    preflight["schedule"]["tasks_per_rung"] = canary["schedule"]["tasks_per_rung"]
    preflight["orchestrator"]["tasks"] = canary["orchestrator"]["tasks"]
    assert preflight == canary


def test_profiles_have_the_declared_scale_and_hardware() -> None:
    canary = Config(Config.PROJECT_ROOT / "configs" / "canary.toml").current
    assert canary["machine_env"] == "MonadMetal"
    assert canary["clearml_capture_streams"] is False
    assert canary["schedule"]["rungs"] == "all"
    assert canary["orchestrator"]["max_depth"] == 8
    assert canary["orchestrator"]["max_task_seconds"] == canary["orchestrator"]["max_total_task_seconds"] == 900
    assert canary["min_fixed_query_samples"] == 32
    assert canary["orchestrator"]["refine"]["mode"] == "always"
    assert canary["evolution"]["composition"]["max_initial_glue_values"] == 0

    smoke = Config(Config.PROJECT_ROOT / "configs" / "smoke.toml").current
    assert smoke["orchestrator"]["evolve"] == canary["orchestrator"]["evolve"]
    assert smoke["n_samples"] == 48
    assert smoke["orchestrator"]["max_depth"] == 1
    assert smoke["orchestrator"]["refine"]["mode"] == "decay"
    assert smoke["orchestrator"]["refine"]["budget_k"] == 4
    assert smoke["orchestrator"]["direct"]["pop_size"] == 16
    assert smoke["orchestrator"]["field"]["train_sites"] == 1024
    assert smoke["orchestrator"]["field"]["audit_sites"] == 4096
    assert smoke["orchestrator"]["field"]["verify_top_k"] == 2
    assert smoke["evolution"]["composition"]["pop_size"] == 24
    assert smoke["evolution"]["composition"]["max_initial_glue_values"] == 5_000_000

    brute = Config(Config.PROJECT_ROOT / "configs" / "brute.toml").current
    assert len(brute["schedule"]["rungs"]) == 1 and 1 <= brute["schedule"]["rungs"][0] <= 18  # intentionally retargetable
    assert brute["schedule"]["tasks_per_rung"] == brute["orchestrator"]["tasks"] == 2000
    assert brute["n_samples"] == 400
    assert brute["orchestrator"]["budgets"]["depth0"] == 2000
    assert brute["orchestrator"]["max_total_task_seconds"] == 7200

    lattice = Config(Config.PROJECT_ROOT / "configs" / "canary-lattice.toml").current
    assert lattice["machine_env"] == "LocalLatticeCUDA"
    assert lattice["orchestrator"]["direct"]["train"]["batched"] is True
    assert lattice["resources"]["device_fraction"] == 0.65
    assert lattice["tf32"] is False

    full = Config(Config.PROJECT_ROOT / "configs" / "full_cluster.toml").current
    assert full["machine_env"] == "ClusterCUDA"
    assert full["schedule"]["tasks_per_rung"] == 20
    assert full["orchestrator"]["tasks"] == 360
    assert full["campaign"] == {"seeds": [0, 1, 2], "cold_library": True}
    assert full["orchestrator"]["field"]["train_sites"] == 16384
    assert full["orchestrator"]["field"]["audit_sites"] == 65536
    assert full["orchestrator"]["field"]["verify_top_k"] == 8


def test_runtime_inventory_covers_live_surfaces() -> None:
    inventory = build_inventory()
    assert inventory["canonical_config"] == "configs/smoke.toml"
    assert inventory["canonical_config_sha256"] == hashlib.sha256(Config.DEFAULT_CONFIG.read_bytes()).hexdigest()
    assert set(inventory["available_configs"]) == TOP_LEVEL_CONFIGS
    assert "orchestrator.direct.train.kind" in inventory["config_keys"]
    assert {"config_path", "config_sha256", "orchestrator"} <= set(inventory["runtime_config_keys"])
    assert "gradient_scheduled" in inventory["registries"]["train"]
    assert "untie_motif_weights" in inventory["registries"]["mutation"]
    assert inventory["console_scripts"]["app"] == "ardevo.main:main"
    assert inventory["console_scripts"]["runtime_inventory"] == "ardevo.tools.runtime_inventory:main"
    assert {row["path"] for row in inventory["run_paths"]} == {row["path"] for row in RUN_PATHS}
