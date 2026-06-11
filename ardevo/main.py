import argparse

from ardevo.trials.continuous_trial import ContinuousTrial
from ardevo.trials.orchestrated_trial import OrchestratedTrial
from ardevo.trials.xor_trial import EvolutionTrial
from ardevo.utils.config import Config
from ardevo.utils.logging import Logger
from ardevo.utils.pipelines import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="ArdEVO: evolve network topologies on the Icarus ladder.")
    parser.add_argument("--config", type=str, default=None, help="Path to a config.toml (defaults to repo root).")
    parser.add_argument("--resume", type=str, default=None, help="Resume a continuous run from its run directory (e.g. results/<ts>_continuous).")
    args = parser.parse_args()

    config = Config(conf_path=args.config)
    if args.resume:
        config.current["resume"] = args.resume
    logger = Logger.get_logger()

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
