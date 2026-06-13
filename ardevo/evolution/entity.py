"""The Entity level: a meta-GA that evolves the search's OWN parameters across task batches.

ArdEVO already has the structural hierarchy (orchestrator over compositions over modules, with
recursion and decomposition). What it lacks is an evolved layer that tunes HOW that search runs:
the operator rates, the novelty pressure, the budgets, the stall window are hand-set constants and
run history is never read back. An `EntityGenome` is a small chromosome of those search-control
knobs; a `MetaEvolver` runs the orchestrator on a task BATCH per Entity, scores each by the search
it induces, and evolves the population, carrying the champion's config forward.

This is the macro analogue of the per-genome self-adaptive operator rates (the micro level): there a
single genome evolves how it mutates; here the whole search evolves how it searches. Default-off
(`[entity] enabled = false`), so the ordinary orchestrated path is untouched. The MetaEvolver takes
its scorer by injection, so the core is pure and testable without a heavy run; production injects the
orchestrator-batch evaluator below.
"""

import copy
import random
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ardevo.utils.logging import Logger

logger = Logger.get_logger()


@dataclass(frozen=True)
class EntityGene:
    """One search-control knob the Entity evolves: a config path and its bounds."""

    path: tuple[str, ...]
    low: float
    high: float
    integer: bool = False

    @property
    def key(self) -> str:
        return "/".join(self.path)


# The Entity chromosome: the few knobs with the most leverage over the search, kept small on purpose
# (no premature breadth; widen once these earn it). Each is a real path into the config tables.
ENTITY_GENES: tuple[EntityGene, ...] = (
    EntityGene(("evolution", "novelty", "weight"), 0.0, 0.9),
    EntityGene(("evolution", "mutation", "enable_refinement_prob"), 0.0, 0.15),
    EntityGene(("orchestrator", "stall_generations"), 12, 60, integer=True),
    EntityGene(("orchestrator", "budgets", "depth0"), 80, 320, integer=True),
    EntityGene(("orchestrator", "decompose_solvability_floor"), 0.3, 0.85),
)


def _get_path(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _set_path(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = config
    for key in path[:-1]:
        nxt = node.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            node[key] = nxt
        node = nxt
    node[path[-1]] = value


@dataclass
class EntityGenome:
    """A point in search-policy space: a value per Entity gene, clamped to the gene's bounds."""

    values: dict[str, float]
    genes: tuple[EntityGene, ...] = ENTITY_GENES

    @classmethod
    def seed(cls, base_config: dict[str, Any], genes: tuple[EntityGene, ...] = ENTITY_GENES) -> "EntityGenome":
        """Start from the config's current knob values (or each gene's midpoint when unset)."""
        values: dict[str, float] = {}
        for gene in genes:
            current = _get_path(base_config, gene.path)
            values[gene.key] = float(current) if current is not None else (gene.low + gene.high) / 2.0
        return cls(values, genes)

    def _clamp(self, gene: EntityGene, value: float) -> float:
        bounded = min(gene.high, max(gene.low, value))
        return float(round(bounded)) if gene.integer else bounded

    def mutate(self, rng: random.Random, sigma: float = 0.2) -> "EntityGenome":
        """Gaussian perturbation scaled by each gene's range, clamped to bounds."""
        mutated = {}
        for gene in self.genes:
            step = rng.gauss(0.0, sigma) * (gene.high - gene.low)
            mutated[gene.key] = self._clamp(gene, self.values.get(gene.key, (gene.low + gene.high) / 2.0) + step)
        return EntityGenome(mutated, self.genes)

    def apply(self, base_config: dict[str, Any]) -> dict[str, Any]:
        """A deep copy of `base_config` with this Entity's knob values written into their paths."""
        config = copy.deepcopy(base_config)
        for gene in self.genes:
            value = self.values[gene.key]
            _set_path(config, gene.path, int(value) if gene.integer else value)
        return config

    def to_dict(self) -> dict[str, float]:
        return dict(self.values)


@dataclass
class MetaEvolver:
    """Evolve a population of EntityGenomes by the search each induces. `evaluate` is injected (the
    orchestrator-batch scorer in production, a synthetic objective in tests), so this stays pure."""

    base_config: dict[str, Any]
    evaluate: Callable[[EntityGenome], float]
    pop_size: int = 6
    generations: int = 4
    elitism: int = 1
    sigma: float = 0.2
    genes: tuple[EntityGene, ...] = ENTITY_GENES
    history: list[dict[str, Any]] = field(default_factory=list)

    def run(self, rng: random.Random) -> EntityGenome:
        seed = EntityGenome.seed(self.base_config, self.genes)
        population = [seed] + [seed.mutate(rng, self.sigma) for _ in range(self.pop_size - 1)]
        champion, champion_score = seed, float("-inf")
        for generation in range(self.generations):
            scored = sorted(((genome, self.evaluate(genome)) for genome in population), key=lambda pair: pair[1], reverse=True)
            if scored[0][1] > champion_score:
                champion, champion_score = scored[0]
            self.history.append({"generation": generation, "best_score": scored[0][1], "best": scored[0][0].to_dict()})
            logger.info("entity gen %d: best score %.4f", generation, scored[0][1])
            elites = [genome for genome, _score in scored[: self.elitism]]
            children = [elites[rng.randrange(len(elites))].mutate(rng, self.sigma) for _ in range(self.pop_size - len(elites))]
            population = elites + children
        return champion


def build_orchestrator_batch_evaluator(base_config: dict[str, Any], tasks: list[Any], *, seed: int = 0) -> Callable[[EntityGenome], float]:
    """Production scorer: run the orchestrator on a fixed task BATCH under an Entity's config and score
    it by the mean accept metric the search achieves (an Entity is as good as the search it induces).
    Each evaluation uses a throwaway library so Entities never pollute the real one or each other."""
    from ardevo.evolution.loop import HierarchicalLoop
    from ardevo.evolution.registry import build_loop
    from ardevo.library import ModuleLibrary
    from ardevo.orchestrator import Orchestrator

    def evaluate(entity: EntityGenome) -> float:
        config = entity.apply(base_config)
        loop = build_loop(config)
        if not isinstance(loop, HierarchicalLoop):
            raise ValueError('the entity layer requires [evolution] loop = "hierarchical"')
        with tempfile.TemporaryDirectory(prefix="ardevo_entity_") as scratch:
            library = ModuleLibrary(Path(scratch) / "lib")
            loop.attach_library(library)
            state = loop.fresh_state(random.Random(seed))
            orchestrator = Orchestrator(config, loop, library, state)
            for task in tasks:
                orchestrator.solve(task)
            metrics = [attempt.metric for attempt in orchestrator.attempts]
        return sum(metrics) / len(metrics) if metrics else 0.0

    return evaluate


def run_entity_layer(config: dict[str, Any]) -> dict[str, Any]:
    """Evolve the search policy on a small task batch, then return `config` with the champion Entity's
    knobs written in. Called from `main` only when `[entity] enabled`; the heavy orchestrated trial
    then runs once under the tuned policy."""
    from ardevo.evolution.multitask import build_pool_report

    table = config.get("entity", {})
    schedule_cfg = config.get("schedule", {})
    rungs_cfg = schedule_cfg.get("rungs", [1, 2, 3, 4, 5])
    rungs = list(range(1, 19)) if rungs_cfg == "all" else [int(rung) for rung in rungs_cfg]
    report = build_pool_report(
        source=config["dataset"],
        rungs=rungs,
        n_samples=int(config["n_samples"]),
        support_fraction=float(config.get("support_fraction", 0.8)),
        tasks_per_rung=int(table.get("batch_size", 4)),
        shuffle=bool(schedule_cfg.get("shuffle", True)),
        seed=int(config.get("seed", 0)),
    )
    batch = [entry.task for entry in report.entries[: int(table.get("batch_size", 4))]]
    if not batch:
        logger.warning("entity layer: no tasks in the batch; running with the base config")
        return config
    evaluator = build_orchestrator_batch_evaluator(config, batch, seed=int(config.get("seed", 0)))
    meta = MetaEvolver(
        config,
        evaluator,
        pop_size=int(table.get("pop_size", 6)),
        generations=int(table.get("generations", 4)),
        elitism=int(table.get("elitism", 1)),
        sigma=float(table.get("sigma", 0.2)),
    )
    champion = meta.run(random.Random(int(config.get("seed", 0))))
    logger.info("entity layer champion: %s", champion.to_dict())
    return champion.apply(config)
