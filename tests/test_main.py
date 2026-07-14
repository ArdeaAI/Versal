"""main() runs the orchestrated trial only; a config that cannot drive it is a startup error."""

import json

import pytest

from ardevo.main import load_run_config, require_orchestrator
from ardevo.utils.config import Config


def test_missing_orchestrator_table_is_a_startup_error():
    with pytest.raises(SystemExit, match=r"\[orchestrator\]"):
        require_orchestrator({"orchestrator": {}, "schedule": {"kind": "interleave_rungs"}})


def test_default_config_drives_the_orchestrated_trial():
    config = Config()  # resolves the fast smoke profile
    require_orchestrator(config.current)
    assert config.current["orchestrator"].get("evolve")
    assert config.current["config_path"] == str(Config.DEFAULT_CONFIG)


def test_implicit_resume_uses_effective_run_snapshot(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source = b'extends = "../configs/missing-after-snapshot.toml"\n'
    (run_dir / "config.toml").write_bytes(source)
    effective = Config().current
    effective["seed"] = 19
    effective["orchestrator"]["library_dir"] = "results/matrix/cold"
    (run_dir / "config.effective.json").write_text(json.dumps(effective))

    restored = load_run_config(None, str(run_dir))

    assert restored.current["seed"] == 19
    assert restored.current["orchestrator"]["library_dir"] == "results/matrix/cold"
    assert restored.current["config_path"] == str(run_dir / "config.toml")
