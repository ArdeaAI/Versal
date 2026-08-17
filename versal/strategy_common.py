"""Shared strategy contracts and structural bookkeeping."""

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable

from versal.evolution.composition import CompositionGenome
from versal.evolution.evolver import Assessed
from versal.evolution.genome import Genome, InnovationTracker
from versal.evolution.loop import AssessedComposition, HierarchicalLoop, HierarchicalState
from versal.library import ModuleLibrary
from versal.utils.resources import StageDecision, StageFootprint

if TYPE_CHECKING:
    from versal.topology import TopologyTabuSession


@dataclass(frozen=True)
class StrategyPreflight:
    eligible: bool
    representation: str
    footprint: StageFootprint | None = None
    decision: StageDecision | None = None
    reason: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyRuntime:
    """Everything a strategy needs from the orchestrator, bundled so signatures stay stable."""

    loop: HierarchicalLoop
    library: ModuleLibrary
    state: HierarchicalState
    accept_threshold: float
    metric_of: Callable[[Any], float]  # reads .metrics; works for Assessed and AssessedComposition
    stall_factory: Callable[[int], Callable[[int, Any], bool]]
    on_generation: Callable[[str, int, Any, float], None] | None = None  # (strategy, gen, best, mean)
    accepts: Callable[[Any], bool] | None = None
    deadline_exceeded: Callable[[], bool] | None = None
    shutdown_requested: Callable[[], bool] | None = None
    topology_tabu: "TopologyTabuSession | None" = None
    deadline: float | None = None

    def accepted(self, item: Any) -> bool:
        return self.accepts(item) if self.accepts is not None else self.metric_of(item) >= self.accept_threshold

    def should_stop(self) -> bool:
        return self.deadline_exceeded is not None and self.deadline_exceeded()

    def should_shutdown(self) -> bool:
        return self.shutdown_requested is not None and self.shutdown_requested()


@dataclass
class StrategyResult:
    """A search result with separate support, reporting, and admission evidence."""

    strategy: str
    metric: float
    generations_used: int
    champion_comp: AssessedComposition | None = None
    champion_genome: Genome | None = None
    report_candidate_comp: AssessedComposition | None = None
    report_candidate_genome: Genome | None = None
    report_candidate_routed: Any | None = None
    report_candidate_metrics: dict[str, float] = field(default_factory=dict)
    champion_metrics: dict[str, float] = field(default_factory=dict)
    report_metrics: dict[str, float] = field(default_factory=dict)
    report_attempted: bool = False
    validation_status: str = "not_run"
    validation_metrics: dict[str, float] = field(default_factory=dict)
    champion_routed: Any | None = None
    seed_metric: float | None = None
    size_metrics: dict[str, float] = field(default_factory=dict)
    resource_metrics: dict[str, float] = field(default_factory=dict)
    strategy_metrics: dict[str, float] = field(default_factory=dict)
    field_template: dict[str, Any] | None = None
    representation: str | None = None
    skip_reason: str | None = None

    @property
    def has_report_candidate(self) -> bool:
        """Whether a support-selected payload is available for held-out reporting."""

        return (
            self.champion_comp is not None
            or self.champion_genome is not None
            or self.report_candidate_comp is not None
            or self.report_candidate_genome is not None
            or self.report_candidate_routed is not None
        )

    @property
    def has_admissible_champion(self) -> bool:
        """Whether this result carries an executable payload that can satisfy the solve contract."""

        return self.champion_comp is not None or self.champion_genome is not None or self.champion_routed is not None


def _module_size_metrics(champion: Genome, population: list[Assessed]) -> dict[str, float]:
    """Return champion and final-population structural sizes."""
    metrics = {
        "champion_nodes": float(len(champion.nodes)),
        "champion_connections": float(sum(connection.enabled for connection in champion.connections)),
        "champion_complexity": float(champion.complexity()),
    }
    if population:
        node_counts = sorted(len(member.genome.nodes) for member in population)
        connection_counts = sorted(sum(connection.enabled for connection in member.genome.connections) for member in population)
        metrics["pop_median_nodes"] = float(node_counts[len(node_counts) // 2])
        metrics["pop_max_nodes"] = float(node_counts[-1])
        metrics["pop_median_connections"] = float(connection_counts[len(connection_counts) // 2])
        metrics["pop_max_connections"] = float(connection_counts[-1])
    return metrics


def comp_size_metrics(comp: Any) -> dict[str, float]:
    """Return composition shell size; referenced modules are priced separately."""
    return {"champion_modules": float(len(comp.module_ids)), "champion_complexity": float(comp.complexity())}


def _restamp_genome(source: Genome, tracker: InnovationTracker) -> Genome:
    """Move a grammar seed into the receiving run's innovation namespace."""

    id_map = {node_id: tracker.new_node_id() for node_id in sorted(source.nodes)}
    nodes = {id_map[node_id]: replace(node, id=id_map[node_id]) for node_id, node in source.nodes.items()}
    groups = {group for conn in source.connections if (group := conn.tie_group) is not None}
    tie_groups = {group: tracker.new_marker() for group in sorted(groups)}
    connections = [
        replace(
            conn,
            in_id=id_map[conn.in_id],
            out_id=id_map[conn.out_id],
            innovation=tracker.innovation(id_map[conn.in_id], id_map[conn.out_id], conn.recurrent),
            tie_group=tie_groups[conn.tie_group] if conn.tie_group is not None else None,
        )
        for conn in source.connections
    ]
    macros = [
        replace(
            macro,
            input_node_ids=tuple(id_map[node_id] for node_id in macro.input_node_ids),
            output_node_ids=tuple(id_map[node_id] for node_id in macro.output_node_ids),
            innovation=tracker.new_marker(),
        )
        for macro in source.macros
    ]
    return Genome(nodes, connections, macros, source.refine_steps, dict(source.operator_rates))


def _restamp_composition(source: CompositionGenome, tracker: InnovationTracker) -> CompositionGenome:
    id_map = {node_id: tracker.new_node_id() for node_id in sorted(source.nodes)}
    nodes = {id_map[node_id]: replace(node, id=id_map[node_id]) for node_id, node in source.nodes.items()}
    edges = [replace(edge, in_id=id_map[edge.in_id], out_id=id_map[edge.out_id], innovation=tracker.innovation(id_map[edge.in_id], id_map[edge.out_id])) for edge in source.edges]
    return CompositionGenome(nodes=nodes, edges=edges)
