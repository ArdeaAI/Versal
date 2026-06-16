"""Pipeline: a thin wrapper around a ClearML Task with graceful offline fallback.

Ported and slimmed from the sibling NEXUS infra. Drops the heavy DataLoader coupling:
ArdEVO trials load their own Icarus data, so `Pipeline` only owns the ClearML task,
the machine -> queue mapping, and trial orchestration.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import subprocess
from typing import TYPE_CHECKING, Any

from ardevo.utils.logging import Logger

if TYPE_CHECKING:
    from clearml import Task

# ClearML is optional and gated by the [run] clearml flag. Detect it without importing the
# value (the import happens lazily inside _create_task, where the task is actually built).
HAS_CLEARML = importlib.util.find_spec("clearml") is not None

VALID_MACHINE_ENVS = {"local", "LatticeCPU", "LatticeCUDA", "MonadCPU", "MonadMetal"}
QUEUE_BY_MACHINE = {
    "LatticeCPU": "lattice_cpu",
    "LatticeCUDA": "lattice_cuda",
    "MonadCPU": "local",
    "MonadMetal": "local",
}
logger = Logger.get_logger()


def get_current_branch() -> str:
    """Get the current git branch name, or an empty string if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


class Pipeline:
    """Owns the ClearML task, machine -> queue routing, and trial execution."""

    def __init__(self, config: dict[str, Any], load_data: bool = False) -> None:
        self.config = config
        self.clearml_run = bool(config.get("clearml_run", False)) and HAS_CLEARML
        self.project_name = config.get("project_name", "ardevo")
        self.experiment_name = config.get("experiment_name", "experiment")
        self.machine_env = config.get("machine_env", "local")
        self.dataset_name = config.get("dataset", "")
        self.repo = config.get("repo", "")
        self.task: Task | None = None
        self.trials: list[Any] = []

        if self.machine_env not in VALID_MACHINE_ENVS:
            raise ValueError(f"Invalid machine environment: {self.machine_env}")
        self._set_queue()
        self._create_task()

        if load_data:
            logger.warning("Pipeline.load_data is a no-op in ArdEVO; trials load their own data.")

    def _set_queue(self) -> None:
        self.queue = QUEUE_BY_MACHINE.get(self.machine_env, "local")
        if not self.clearml_run:
            self.queue = "local"

    def _create_task(self) -> None:
        if not self.clearml_run:
            return
        from clearml import Task

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = f"{self.experiment_name}_{self.machine_env}_{timestamp}"
        self.task = Task.init(
            project_name=self.project_name,
            task_name=task_name,
            task_type=Task.TaskTypes.custom,
            reuse_last_task_id=False,
            output_uri=self.config.get("output_uri", False),
        )
        self.task.connect(self.config.get("hyperparameters", {}))
        branch = get_current_branch()
        if self.repo and branch:
            self.task.set_repo(repo=self.repo, branch=branch)

    def get_pipeline_task(self) -> "Task | None":
        return self.task

    def add_trial(self, trial_class: type, **trial_kwargs: Any) -> None:
        """Instantiate a trial CLASS (not an instance) with config + task."""
        trial = trial_class(config=self.config, task=self.task, **trial_kwargs)
        self.trials.append(trial)

    def run_task(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if self.queue == "local" or not self.clearml_run:
            for trial in self.trials:
                results.append(trial.run())
        else:
            # A non-local queue is only reachable when clearml_run is true, so the task exists.
            assert self.task is not None
            # Enqueue to the remote agent and exit the local process; the cloned run
            # re-executes from the top and reaches the trial loop on the agent.
            self.task.execute_remotely(queue_name=self.queue, clone=True, exit_process=True)
            for trial in self.trials:
                results.append(trial.run())

        if self.task is not None:
            self.task.close()
        return results

    def get_pipeline_info(self) -> str:
        return json.dumps(
            {
                "project_name": self.project_name,
                "experiment_name": self.experiment_name,
                "machine_env": self.machine_env,
                "dataset": self.dataset_name,
                "clearml_run": self.clearml_run,
                "queue": self.queue,
                "repo": self.repo,
            },
            indent=4,
        )
