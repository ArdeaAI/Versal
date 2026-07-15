"""main() runs the orchestrated trial only; a config that cannot drive it is a startup error."""

import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from ardevo import main as main_module
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


@pytest.mark.parametrize(
    ("clearml_default", "clearml_flag", "expected_clearml"),
    [(False, "--clearml", True), (True, "--no-clearml", False)],
)
def test_cli_machine_and_clearml_overrides_reach_pipeline_without_starting_trial(
    monkeypatch: pytest.MonkeyPatch,
    clearml_default: bool,
    clearml_flag: str,
    expected_clearml: bool,
) -> None:
    captured: dict[str, Any] = {"run_task": 0}
    current = {
        "machine_env": "MonadMetal",
        "clearml_run": clearml_default,
        "orchestrator": {"library_dir": "library", "evolve": ["direct"]},
    }

    def fake_load_run_config(config_path: str | None, resume: str | None):
        captured["config_path"] = config_path
        captured["resume"] = resume
        return SimpleNamespace(current=current)

    class FakePipeline:
        def __init__(self, config: dict[str, Any], load_data: bool = False) -> None:
            captured["pipeline_config"] = dict(config)
            captured["load_data"] = load_data

        def get_pipeline_info(self) -> str:
            return "{}"

        def add_trial(self, trial_class: type) -> None:
            captured["trial_class"] = trial_class

        def run_task(self) -> list[dict[str, Any]]:
            captured["run_task"] += 1
            return []

    monkeypatch.setattr(main_module, "load_run_config", fake_load_run_config)
    monkeypatch.setattr(main_module, "configure_precision", lambda _config: None)
    monkeypatch.setattr(main_module, "configure_assess_pool", lambda _config: None)
    monkeypatch.setattr(main_module, "Pipeline", FakePipeline)
    monkeypatch.setattr(main_module.Logger, "configure", lambda **_values: None)
    monkeypatch.setattr(main_module.Logger, "get_logger", lambda: SimpleNamespace(debug=lambda *_args: None))
    monkeypatch.setattr(
        sys,
        "argv",
        ["app", "--config", "configs/canary-lattice.toml", "--machine", "LocalLatticeCUDA", clearml_flag],
    )

    main_module.main()

    assert captured["config_path"] == "configs/canary-lattice.toml"
    assert captured["resume"] is None
    assert captured["pipeline_config"]["machine_env"] == "LocalLatticeCUDA"
    assert captured["pipeline_config"]["clearml_run"] is expected_clearml
    assert captured["load_data"] is False
    assert captured["trial_class"] is main_module.OrchestratedTrial
    assert captured["run_task"] == 1
