"""main() runs the orchestrated trial only; a config that cannot drive it is a startup error."""

import pytest

from ardevo.main import require_orchestrator
from ardevo.utils.config import Config


def test_missing_orchestrator_table_is_a_startup_error():
    with pytest.raises(SystemExit, match=r"\[orchestrator\]"):
        require_orchestrator({"orchestrator": {}, "schedule": {"kind": "interleave_rungs"}})


def test_default_config_drives_the_orchestrated_trial():
    config = Config()  # resolves configs/orchestrated_overmind.toml
    require_orchestrator(config.current)
    assert config.current["orchestrator"].get("evolve")
