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
    from ardevo.utils.device import resolve_worker_count

    orchestrator = config.get("orchestrator", {})
    workers = resolve_worker_count(orchestrator.get("direct", {}).get("assess_workers", 0))
    if orchestrator and workers > 1:
        import torch

        from ardevo.evolution.evolver import create_assess_pool

        create_assess_pool(workers, orchestrator.get("library_dir", "library"))
        Logger.get_logger().info("assess pool: %d workers (1 torch thread each); main process torch threads: %d", workers, torch.get_num_threads())


def configure_precision(config: dict[str, Any]) -> None:
    """Opt-in TF32 for CUDA matmuls (`[run] tf32 = true`). TF32 truncates the mantissa during
    accumulation, so it changes numerics materially; the default stays full fp32."""
    if bool(config.get("tf32", False)):
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        Logger.get_logger().info("TF32 enabled for CUDA matmuls ([run] tf32 = true)")


def main() -> None:
    parser = argparse.ArgumentParser(description="ArdEVO: the orchestrated overmind evolver on the Icarus ladder.")
    parser.add_argument("--config", type=str, default=None, help="Path to a run config (defaults to configs/orchestrated_overmind.toml).")
    parser.add_argument("--resume", type=str, default=None, help="Resume a run from its directory (e.g. results/<ts>_orchestrated).")
    args = parser.parse_args()

    config = Config(conf_path=args.config)
    require_orchestrator(config.current)
    if args.resume:
        config.current["resume"] = args.resume
    configure_precision(config.current)
    configure_assess_pool(config.current)
    logger = Logger.get_logger()

    pipe = Pipeline(config.current, load_data=False)
    logger.info("pipeline: %s", pipe.get_pipeline_info())
    pipe.add_trial(OrchestratedTrial)
    pipe.run_task()


if __name__ == "__main__":
    main()
