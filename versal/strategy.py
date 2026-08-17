"""Config-driven search strategy registry and stable public facade."""

from typing import Any, Callable

from versal.evolution.registry import Registry, build_evolver
from versal.strategy_common import StrategyPreflight, StrategyResult, StrategyRuntime, comp_size_metrics
from versal.strategy_composition import CompositionStrategy
from versal.strategy_direct import DirectStrategy
from versal.strategy_field import FieldStrategy
from versal.strategy_grammar import GrammarStrategy

__all__ = [
    "CompositionStrategy",
    "DirectStrategy",
    "EVOLVE_STRATEGY",
    "FieldStrategy",
    "GrammarStrategy",
    "StrategyPreflight",
    "StrategyResult",
    "StrategyRuntime",
    "build_strategies",
    "comp_size_metrics",
]

EVOLVE_STRATEGY: Registry = Registry("evolve_strategy")


@EVOLVE_STRATEGY.register("composition")
def _build_composition(config: dict[str, Any]) -> "CompositionStrategy":
    return CompositionStrategy(blind_query=bool(config.get("orchestrator", {}).get("blind_query", False)))


@EVOLVE_STRATEGY.register("field")
def _build_field(config: dict[str, Any]) -> "FieldStrategy":
    overlay = dict(config)
    table = config.get("orchestrator", {}).get("field", {}) or {}
    evolution = {key: value for key, value in config.get("evolution", {}).items() if key != "loop"}
    for key in ("pop_size", "elitism", "assess_workers", "mutation", "train", "evaluate", "novelty", "halving_stages", "halving_keep"):
        if key in table:
            evolution[key] = table[key]
    overlay["evolution"] = evolution
    overlay["library_dir"] = config.get("orchestrator", {}).get("library_dir", "library")
    return FieldStrategy(
        evolver=build_evolver(overlay),
        train_sites=max(1, int(table.get("train_sites", 4096))),
        audit_sites=max(1, int(table.get("audit_sites", 16384))),
        verify_top_k=max(1, int(table.get("verify_top_k", 5))),
        verify_chunk_size=max(1, int(table.get("verify_chunk_size", 32768))),
        blind_query=bool(config.get("orchestrator", {}).get("blind_query", False)),
    )


@EVOLVE_STRATEGY.register("direct")
def _build_direct(config: dict[str, Any]) -> "DirectStrategy":
    overlay = dict(config)
    table = config.get("orchestrator", {}).get("direct", {})
    evolution = {key: value for key, value in config.get("evolution", {}).items() if key != "loop"}
    evolution["pop_size"] = int(table.get("pop_size", 48))
    evolution["elitism"] = int(table.get("elitism", 2))
    evolution["assess_workers"] = table.get("assess_workers", 0)  # "auto" resolves in build_evolver
    overlay["library_dir"] = config.get("orchestrator", {}).get("library_dir", "library")
    # Single-task structure growth usually wants a stronger inner trainer (and sometimes a
    # different mutation recipe) than the composition loop's glue fitting; both are overridable.
    for overridable in ("mutation", "train", "evaluate", "novelty", "halving_stages", "halving_keep"):
        if overridable in table:
            evolution[overridable] = table[overridable]
    overlay["evolution"] = evolution
    max_flat_outputs = table.get("max_flat_outputs", 0)
    max_init_genes = table.get("max_init_genes", 0)
    for name, value in (("max_flat_outputs", max_flat_outputs), ("max_init_genes", max_init_genes)):
        if value != "adaptive" and int(value) < 0:
            raise ValueError(f"[orchestrator.direct] {name} must be non-negative or 'adaptive'")
    return DirectStrategy(
        evolver=build_evolver(overlay),
        max_flat_outputs=max_flat_outputs if max_flat_outputs == "adaptive" else int(max_flat_outputs),
        max_init_genes=max_init_genes if max_init_genes == "adaptive" else int(max_init_genes),
        structured_grid=bool(table.get("structured_grid", False)),
        blind_query=bool(config.get("orchestrator", {}).get("blind_query", False)),
    )


@EVOLVE_STRATEGY.register("grammar")
def _build_grammar(config: dict[str, Any]) -> "GrammarStrategy":
    table = config.get("orchestrator", {}).get("grammar", {}) or {}
    return GrammarStrategy(
        direct=_build_direct(config),
        blind_query=bool(config.get("orchestrator", {}).get("blind_query", False)),
        max_productions=max(1, int(table.get("max_productions", 12))),
        candidates_per_production=max(1, int(table.get("candidates_per_production", 3))),
        mutation_steps=max(0, int(table.get("mutation_steps", 2))),
        module_sizes=tuple(int(size) for size in table.get("module_sizes", [3, 4])),
        composition_sizes=tuple(int(size) for size in table.get("composition_sizes", [2, 3, 4])),
        min_lineage_support=max(2, int(table.get("min_lineage_support", 2))),
        per_entry_cap=max(1, int(table.get("per_entry_cap", 5000))),
    )


@EVOLVE_STRATEGY.register("routed")
def _build_routed(config: dict[str, Any]) -> Any:
    # Lazy import (the train.py pattern): routing pulls in torch-heavy machinery only when configured.
    from versal.routing import build_routed_strategy

    return build_routed_strategy(config)


def build_strategies(config: dict[str, Any]) -> list[tuple[str, Callable[..., StrategyResult]]]:
    """Resolve `[orchestrator] evolve` (default: composition only, the pre-strategy behavior)."""
    names = [str(name) for name in config.get("orchestrator", {}).get("evolve", ["composition"])]
    return [(name, EVOLVE_STRATEGY.get(name)(config)) for name in names]
