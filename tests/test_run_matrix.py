"""run_matrix: config provenance stamping and the pure scorecard aggregation over
run_summary.json records (the tier ladder T0-T4 that operationalizes "meaningful result")."""

import hashlib
from pathlib import Path

from versal.tools.run_matrix import build_scorecard, flatten_rows, parse_rungs, rung_tier
from versal.utils.config import Config


def _summary(tasks: list[dict], **overrides: object) -> dict:
    summary: dict[str, object] = {
        "run_dir": "results/x",
        "status": "done",
        "config_path": "configs/canary.toml",
        "config_sha256": "deadbeef",
        "seed": 0,
        "library_dir": "library_recon",
        "tasks": tasks,
    }
    summary.update(overrides)
    return summary


def test_config_stamps_path_and_content_hash(tmp_path: Path) -> None:
    config_file = tmp_path / "tiny.toml"
    config_file.write_text("[run]\nseed = 7\n[orchestrator]\ntasks = 1\n")
    config = Config(conf_path=config_file)
    assert config.current["config_path"] == str(config_file)
    assert config.current["config_sha256"] == hashlib.sha256(config_file.read_bytes()).hexdigest()
    assert config.current["seed"] == 7


def test_flatten_rows_stamps_provenance_on_every_task() -> None:
    rows = flatten_rows(_summary([{"rung": 1, "task": "xor", "outcome": "evolved", "metric": 1.0, "new_library_keys": ["m1_a"]}], seed=2))
    assert len(rows) == 1
    row = rows[0]
    assert row["seed"] == 2 and row["config_sha256"] == "deadbeef" and row["run_status"] == "done"
    assert row["outcome"] == "evolved" and row["new_library_keys"] == 1


def test_rung_tiers_cover_the_ladder() -> None:
    assert rung_tier([]) == "T0"
    assert rung_tier([{"outcome": "failed", "metric": 0.1, "new_library_keys": 0}]) == "T1"
    assert rung_tier([{"outcome": "failed", "metric": 0.6, "new_library_keys": 0}]) == "T2"
    assert rung_tier([{"outcome": "failed", "metric": 0.6, "new_library_keys": 1}]) == "T3"  # wall stone admitted
    assert rung_tier([{"outcome": "refined", "metric": 0.97, "new_library_keys": 0}]) == "T4"
    assert rung_tier([{"outcome": "failed", "metric": 0.2, "new_library_keys": 0}], t2_floor=0.1) == "T2"


def test_scorecard_reports_unattempted_rungs_as_t0() -> None:
    rows = flatten_rows(
        _summary(
            [
                {"rung": 1, "task": "xor", "outcome": "evolved", "metric": 1.0, "new_library_keys": ["k"], "seconds": 1.5},
                {"rung": 3, "task": "two_spirals", "outcome": "failed", "metric": 0.83, "new_library_keys": [], "failure_stage": "time_budget", "seconds": 300.0},
            ]
        )
    )
    scorecard = build_scorecard(rows, parse_rungs("1-4"))
    assert scorecard["rungs"]["1"]["tier"] == "T4" and scorecard["rungs"]["1"]["solves"] == 1
    assert scorecard["rungs"]["3"]["tier"] == "T2" and scorecard["rungs"]["3"]["time_budget_hits"] == 1
    assert scorecard["rungs"]["2"]["tier"] == "T0" and scorecard["rungs"]["4"]["attempts"] == 0
    assert scorecard["tier_counts"] == {"T0": 2, "T1": 0, "T2": 1, "T3": 0, "T4": 1}


def test_parse_rungs_handles_ranges_and_lists() -> None:
    assert parse_rungs("1-3,7") == [1, 2, 3, 7]
    assert parse_rungs("18") == [18]
