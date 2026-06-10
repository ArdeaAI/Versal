"""Configuration for ArdEVO.

`config.toml` is the single source of truth. `Config` parses it and produces `current`,
a flat dict that the `Pipeline`/`Proctor` infra reads (scalar run settings plus a
`hyperparameters` dict for ClearML), while preserving the nested `evolution`/`substrate`/
`fitness` tables verbatim so the evolver factory can resolve operators from them.
"""

import json
import tomllib
from pathlib import Path
from typing import Any, ClassVar

PROJECT_ROOT = Path(__file__).parent.parent.parent


class Config:
    """Load and normalize `config.toml` into a runtime dict."""

    PROJECT_ROOT: ClassVar[Path] = PROJECT_ROOT
    TOML_FILE: ClassVar[Path] = PROJECT_ROOT / "pyproject.toml"
    DEFAULT_CONFIG: ClassVar[Path] = PROJECT_ROOT / "config.toml"

    def __init__(self, conf_path: Path | str | None = None) -> None:
        self.toml = self._load_toml()
        self.current = self._load_config(Path(conf_path) if conf_path else self.DEFAULT_CONFIG)

    @classmethod
    def _load_toml(cls) -> dict[str, Any]:
        try:
            with open(cls.TOML_FILE, "rb") as handle:
                return tomllib.load(handle)
        except (FileNotFoundError, tomllib.TOMLDecodeError) as exc:
            print(f"Error reading '{cls.TOML_FILE}': {exc}")
            return {}

    @classmethod
    def _load_config(cls, conf_path: Path) -> dict[str, Any]:
        if not conf_path.exists():
            raise FileNotFoundError(f"Configuration file '{conf_path}' not found.")
        with open(conf_path, "rb") as handle:
            raw = tomllib.load(handle)
        return cls._normalize_config(raw)

    @classmethod
    def _normalize_config(cls, raw: dict[str, Any]) -> dict[str, Any]:
        """Flatten the run settings the infra reads, and keep the operator tables intact."""
        run = raw.get("run", {})
        duration = run.get("duration", {})
        dataset = raw.get("dataset", {})
        substrate = raw.get("substrate", {})
        evolution = raw.get("evolution", {})
        fitness = raw.get("fitness", {})

        generations = duration.get("generations", 100)
        pop_size = evolution.get("pop_size", 64)

        normalized: dict[str, Any] = {
            # Run settings consumed by Pipeline/Proctor.
            "project_name": run.get("project", "ardevo"),
            "experiment_name": run.get("experiment", "experiment"),
            "clearml_run": run.get("clearml", False),
            "machine_env": run.get("machine", "local"),
            "seed": run.get("seed", 0),
            "log_level": run.get("log_level", "INFO"),
            "ml_lib": "torch",
            "repo": run.get("repo", ""),
            "experiment_type": "custom",
            "output_uri": run.get("output_uri", False),
            "dataset": dataset.get("source", ""),
            # Run shape the trial reads directly.
            "generations": generations,
            "rung": dataset.get("rung", 1),
            "n_samples": dataset.get("n_samples", 8),
            # Operator tables preserved verbatim for build_evolver.
            "substrate": substrate,
            "evolution": evolution,
            "fitness": fitness,
        }

        # Flat scalars for ClearML hyperparameter tracking (logging only; not the source of truth).
        normalized["hyperparameters"] = {
            "generations": generations,
            "pop_size": pop_size,
            "elitism": evolution.get("elitism", 1),
            "rung": normalized["rung"],
            "n_samples": normalized["n_samples"],
            "seed": normalized["seed"],
            "selection_kind": evolution.get("selection", {}).get("kind", "tournament"),
            "crossover_kind": evolution.get("crossover", {}).get("kind", "none"),
            "mutation_operators": ",".join(evolution.get("mutation", {}).get("operators", [])),
            "train_kind": evolution.get("train", {}).get("kind", "none"),
            "fitness_components": ",".join(fitness.get("components", [])),
        }
        return normalized

    def __getitem__(self, key: str) -> Any:
        return self.current.get(key, None)

    def __setitem__(self, key: str, value: Any) -> None:
        self.current[key] = value

    def get_project_info(self) -> str:
        project = self.toml.get("project", {})
        return json.dumps(
            {
                "name": str(project.get("name", "ardevo")),
                "version": str(project.get("version", "0.1.0")),
                "root": str(self.PROJECT_ROOT),
                "config": self.current,
            },
            indent=4,
        )

    def __str__(self) -> str:
        return self.get_project_info()
