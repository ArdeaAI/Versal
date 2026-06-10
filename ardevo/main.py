import argparse

from ardevo.trials.xor_trial import EvolutionTrial
from ardevo.utils.config import Config
from ardevo.utils.logging import Logger
from ardevo.utils.pipelines import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="ArdEVO: evolve network topologies on the Icarus ladder.")
    parser.add_argument("--config", type=str, default=None, help="Path to a config.toml (defaults to repo root).")
    args = parser.parse_args()

    config = Config(conf_path=args.config)
    logger = Logger.get_logger()

    pipe = Pipeline(config.current, load_data=False)
    logger.info("pipeline: %s", pipe.get_pipeline_info())

    pipe.add_trial(EvolutionTrial)
    pipe.run_task()


if __name__ == "__main__":
    main()
