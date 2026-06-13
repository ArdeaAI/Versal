import argparse
import os
from typing import Any

from ardevo.trials.continuous_trial import ContinuousTrial
from ardevo.trials.orchestrated_trial import OrchestratedTrial
from ardevo.trials.xor_trial import EvolutionTrial
from ardevo.utils.config import Config
from ardevo.utils.logging import Logger
from ardevo.utils.pipelines import Pipeline


def configure_torch_threads(config: dict[str, Any]) -> None:
    """When candidates are assessed on a thread pool, pin torch to one intra-op thread: the
    per-candidate kernels are tiny (widths 2-128), intra-op threading gains nothing there, and
    N workers x torch's default thread count would oversubscribe the cores. Process-global, so
    it lives at the single entry point, not in trial constructors."""
    if int(config.get("evolution", {}).get("parallel_assess", 0)) > 1:
        import torch

        torch.set_num_threads(1)


def configure_macro_resolver(config: dict[str, Any]) -> None:
    """Macro genes resolve frozen inner networks from the library at decode time. The orchestrated
    trial installs its live instance; flat runs that enable library-reading mutators point at the
    on-disk dir here."""
    from pathlib import Path

    library_dir = config.get("orchestrator", {}).get("library_dir") or config.get("library", {}).get("dir")
    if library_dir and Path(library_dir).exists():
        from ardevo.library import ModuleLibrary, macro_resolver
        from ardevo.substrate import set_macro_resolver

        set_macro_resolver(macro_resolver(ModuleLibrary(library_dir)))


def main() -> None:
    parser = argparse.ArgumentParser(description="ArdEVO: evolve network topologies on the Icarus ladder.")
    parser.add_argument("--config", type=str, default=None, help="Path to a config.toml (defaults to repo root).")
    parser.add_argument("--resume", type=str, default=None, help="Resume a continuous run from its run directory (e.g. results/<ts>_continuous).")
    args = parser.parse_args()

    config_path = args.config
    # A ClearML agent re-runs the entry point with no CLI args, so `--config` is lost. Recover the
    # config chosen at enqueue time from the task (stored in Pipeline._create_task) before loading it.
    if config_path is None and os.environ.get("CLEARML_TASK_ID"):
        from clearml import Task

        config_path = Task.get_task(task_id=os.environ["CLEARML_TASK_ID"]).get_parameter("General/config_path") or None

    config = Config(conf_path=config_path)
    config.current["config_path"] = config_path or str(Config.DEFAULT_CONFIG)
    if args.resume:
        config.current["resume"] = args.resume
    configure_torch_threads(config.current)
    logger = Logger.get_logger()

    # Entity level (the meta-GA over the search itself): when enabled, evolve the search-control knobs
    # on a small task batch and run the orchestrated trial under the tuned policy. Off by default and
    # skipped on resume, so the ordinary path is byte-identical.
    if config.current.get("entity", {}).get("enabled") and not args.resume:
        from ardevo.evolution.entity import run_entity_layer

        logger.info("entity layer enabled: evolving the search policy before the main run")
        config.current = run_entity_layer(config.current)
    configure_macro_resolver(config.current)

    pipe = Pipeline(config.current, load_data=False)
    logger.info("pipeline: %s", pipe.get_pipeline_info())

    # A populated [orchestrator] table selects the orchestrated recursive trial; [schedule] alone
    # selects the continuous multi-rung trial; otherwise a single-rung run.
    if config.current.get("orchestrator"):
        trial = OrchestratedTrial
    elif config.current.get("schedule"):
        trial = ContinuousTrial
    else:
        trial = EvolutionTrial
    pipe.add_trial(trial)
    pipe.run_task()


if __name__ == "__main__":
    main()
