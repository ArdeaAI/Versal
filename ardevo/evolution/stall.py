"""Stall strategies: what the direct search does when it plateaus WITH budget remaining.

The phase-6 direct loop, with novelty on, runs the full budget and never REACTS to a plateau (the
fitness-plateau stop is disabled so novelty can keep exploring). That spends the budget on exploration
but never escalates the operator mix toward the depth/recursion two_spirals needs, and the orchestrated
direct path is depth-timid relative to the proven standalone recipe (add_deep_node 0.12 vs 0.25).

A stall strategy is a registered transform on the mutation pipeline, fired ONCE when the plateau is
detected (or the half-budget mark is reached). `adapt_rates` raises the depth/recursion operator
probabilities for the remaining generations, so structure growth accelerates exactly when warranted
without slowing tasks that solve early. Selected by `[orchestrator] stall_strategy`; the boost values
live in `[orchestrator.stall]`. No key (or "none") keeps the phase-6 behavior byte-identical.
"""

from typing import Any, Callable

from ardevo.evolution.mutation import MutationPipeline
from ardevo.evolution.registry import Registry

STALL_STRATEGY: Registry = Registry("stall_strategy")

# A stall strategy maps the current mutation pipeline to the one used for the rest of a stalled search.
StallStrategy = Callable[[MutationPipeline], MutationPipeline]


@STALL_STRATEGY.register("adapt_rates")
def _build_adapt_rates(config: dict[str, Any]) -> StallStrategy:
    """Boost the operator probabilities named under `[orchestrator.stall]` (operator -> probability)."""
    table = config.get("orchestrator", {}).get("stall", {})
    boosts = {str(name): float(value) for name, value in table.items() if name != "strategy"}

    def escalate(pipeline: MutationPipeline) -> MutationPipeline:
        return pipeline.with_boosted_rates(boosts)

    return escalate


def build_stall_strategy(config: dict[str, Any]) -> StallStrategy | None:
    """Resolve `[orchestrator] stall_strategy` (None or "none" = no escalation, the phase-6 default)."""
    name = config.get("orchestrator", {}).get("stall_strategy")
    if not name or name == "none":
        return None
    return STALL_STRATEGY.get(name)(config)
