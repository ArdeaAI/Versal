"""
Exercise the real trial checkpoint lifecycle with an offline task and idle workers.
"""

import json
import sys
import time
from pathlib import Path

from tests.test_hierarchical_loop import _config
from versal.evolution.evolver import close_assess_pools, create_assess_pool
from versal.evolution.multitask import PoolReport, task_entry
from versal.orchestrator import Orchestrator
from versal.tools.xor_repro import xor_task
from versal.trials import orchestrated_trial
from versal.utils.cancellation import cancelled
from versal.utils.shutdown import EscapeShutdown


def main():
    root = Path(sys.argv[1])
    resume = sys.argv[2] == "resume"
    config = _config()
    config.update(dataset="offline", n_samples=4, seed=0, live_status=False)
    config["orchestrator"] = {
        "tasks": 2,
        "library_dir": str(root / "library"),
        "max_depth": 0,
        "search_policy": "interleaved",
        "blind_query": True,
        "search_metric": "support_accuracy",
        "accept_metric": "support_accuracy",
        "evolve": ["direct"],
        "budgets": {"depth0": 1},
        "refine": {"budget_k": 0},
        "direct": {"pop_size": 4, "assess_workers": 2, "train": {"kind": "gradient", "steps": 1}},
    }
    config["schedule"] = {"kind": "round_robin", "rungs": [1], "tasks_per_rung": 2}
    if resume:
        config["resume"] = str(next((root / "results").glob("*_orchestrated")))
    setattr(orchestrated_trial.results, "DEFAULT_ROOT", root / "results")
    setattr(orchestrated_trial, "build_pool_report", lambda **_kwargs: PoolReport([task_entry(xor_task())], []))
    if not resume:
        original = Orchestrator._evolve

        def wait_for_interrupt(self, *args, **kwargs):
            (root / "ready").touch()
            while not cancelled():
                time.sleep(0.01)
            return original(self, *args, **kwargs)

        setattr(Orchestrator, "_evolve", wait_for_interrupt)
    with EscapeShutdown():
        create_assess_pool(2, str(root / "library"))
        try:
            result = orchestrated_trial.OrchestratedTrial(config).run()
        finally:
            close_assess_pools()
    (root / ("resumed.json" if resume else "stopped.json")).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
