import argparse
from typing import Any

from ardevo.trials.orchestrated_trial import OrchestratedTrial
from ardevo.utils.config import Config
from ardevo.utils.logging import Logger
from ardevo.utils.pipelines import Pipeline


def require_orchestrator(config: dict[str, Any]) -> None:
    """The orchestrated trial is the only run mode; fail fast on a config that cannot drive it."""
    if not config.get("orchestrator"):
        raise SystemExit("config has no [orchestrator] table; the supported run config is configs/orchestrated_overmind.toml")


def configure_assess_pool(config: dict[str, Any]) -> None:
    """Create the direct strategy's process pool HERE, before the Pipeline builds the ClearML task.
    clearml.Task.init unconditionally patches os.fork/multiprocessing, and workers spawned afterward
    inherit CLEARML_PROC_MASTER_ID and stall attaching to the task instead of computing. Spawning the
    persistent workers first (clearml not yet imported) keeps them clean for the whole run."""
    orchestrator = config.get("orchestrator", {})
    workers = int(orchestrator.get("direct", {}).get("assess_workers", 0))
    if orchestrator and workers > 1:
        from ardevo.evolution.evolver import create_assess_pool

        create_assess_pool(workers, orchestrator.get("library_dir", "library"))


def main() -> None:
    parser = argparse.ArgumentParser(description="ArdEVO: the orchestrated overmind evolver on the Icarus ladder.")
    parser.add_argument("--config", type=str, default=None, help="Path to a run config (defaults to configs/orchestrated_overmind.toml).")
    parser.add_argument("--resume", type=str, default=None, help="Resume a run from its directory (e.g. results/<ts>_orchestrated).")
    args = parser.parse_args()

    config = Config(conf_path=args.config)
    require_orchestrator(config.current)
    if args.resume:
        config.current["resume"] = args.resume
    configure_assess_pool(config.current)
    logger = Logger.get_logger()

    pipe = Pipeline(config.current, load_data=False)
    logger.info("pipeline: %s", pipe.get_pipeline_info())
    pipe.add_trial(OrchestratedTrial)
    pipe.run_task()


if __name__ == "__main__":
    main()
