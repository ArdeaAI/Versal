"""Loops: the top-level generational strategies, selectable from config like any other stage.

`flat` is the phase-1 single-population Evolver, untouched. `hierarchical` is the recursive
composition loop: a per-task population of `CompositionGenome`s co-evolves with ONE shared live
module population. Compositions are assembled (live refs resolve to species champions, library refs
inline admitted entries), trained (glue + unfrozen live inner copies), evaluated, and scored;
fitness then flows DOWN as attribution to the module species and library entries each composition
referenced. Modules reproduce with the existing flat-stage operators every `advance_every`
composition generations, using attribution as their fitness.

Ownership is the forgetting fix: compositions and their glue are per task, the module pool is
shared, library entries are frozen. Champion-only module writeback keeps one task's gradient from
silently shredding a module other tasks rely on.
"""

import random
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

from ardevo.dataset.icarus import EncodedTask, Level0Encoder
from ardevo.evaluation import evaluate
from ardevo.evolution.composition import (
    AssemblyContext,
    CompMutationContext,
    CompMutationPipeline,
    ComposedNet,
    CompositionAssemblyError,
    CompositionGenome,
    RefSpec,
    assemble,
    minimal_composition,
    writeback_composition,
)
from ardevo.evolution.evolver import Evolver
from ardevo.evolution.genome import Genome, InnovationTracker, genome_from_dict, genome_to_dict, make_acyclic
from ardevo.evolution.mutation import MutationContext
from ardevo.evolution.registry import Registry, _bind_prefixed, build_evolver
from ardevo.evolution.train import _writeback
from ardevo.library import MODULE, ModuleLibrary
from ardevo.utils.logging import Logger

logger = Logger.get_logger()

LOOP: Registry = Registry("loop")

FLOOR_FITNESS = -1e9


@LOOP.register("flat")
def _build_flat(config: dict[str, Any]) -> "FlatLoop":
    return FlatLoop(evolver=build_evolver(config))


@LOOP.register("hierarchical")
def _build_hierarchical_entry(config: dict[str, Any]) -> "HierarchicalLoop":
    return build_hierarchical(config)


@dataclass
class FlatLoop:
    """Thin shim so `[evolution] loop = "flat"` resolves; the Evolver behaves exactly as before."""

    evolver: Evolver


@dataclass
class LiveModule:
    """One member of the shared module population. Fitness is ATTRIBUTED, never measured directly:
    a module is as good as the compositions that use it."""

    genome: Genome
    fitness: float = 0.0


@dataclass
class HierarchicalState:
    """The durable, cross-task state of the hierarchical loop (per-task comp populations are not
    kept: accepted solutions live in the library, which is the point)."""

    modules: list[LiveModule]
    module_innovations: InnovationTracker
    comp_innovations: InnovationTracker
    rng: random.Random
    generation: int = 0
    species_members: dict[int, list[int]] = field(default_factory=dict)
    species_champion_index: dict[int, int] = field(default_factory=dict)
    species_champions: dict[int, Genome] = field(default_factory=dict)
    module_species_history: list[dict[int, int]] = field(default_factory=list)
    repaired_refs: int = 0


@dataclass
class AssessedComposition:
    comp: CompositionGenome
    metrics: dict[str, float]
    fitness: float
    net: ComposedNet | None  # None when assembly failed (floor fitness)


@dataclass(frozen=True)
class CompTaskSpec:
    """Everything the loop needs to evolve compositions against one task."""

    encoded: EncodedTask
    encoder: Level0Encoder
    n_inputs: int
    input_specs: list[tuple[str, int]]  # (bank signature, width) per INPUT node
    bank_columns: dict[str, list[int]]
    output_ref: str
    output_width: int


@dataclass
class HierarchicalLoop:
    evolver: Evolver  # supplies the bound module-stage operators + train/evaluate/fitness stages
    comp_pop_size: int
    comp_elitism: int
    # Selection ops come from the shared SELECTION registry (genome-type agnostic), so the list
    # element type stays loose here.
    comp_selection_op: Callable[..., list[Any]]
    comp_crossover_op: Callable[..., CompositionGenome]
    comp_crossover_rate: float
    comp_mutation: CompMutationPipeline
    max_inline_depth: int
    glue_scale: float | None
    module_pop_size: int
    module_elitism: int
    in_ports: int
    out_ports: int
    advance_every: int
    writeback_mode: str  # "champion" | "none"
    attribution_mode: str  # "max" | "mean"
    decay: float
    offspring_discount: float
    seed_fraction: float
    library: ModuleLibrary | None = None

    def attach_library(self, library: ModuleLibrary | None) -> None:
        self.library = library

    # --- state ------------------------------------------------------------------------------------

    def fresh_state(self, rng: random.Random) -> HierarchicalState:
        # Seed minimal genomes first so the tracker baseline is well-defined, then graft library
        # modules (matching port shape) over the front of the population: cross-run reuse.
        genomes = [self.evolver.init_op(self.in_ports, self.out_ports, rng=rng) for _ in range(self.module_pop_size)]
        tracker = InnovationTracker.from_genomes(genomes)
        if self.library is not None and self.seed_fraction > 0.0:
            from ardevo.library import graft

            wanted = int(self.seed_fraction * self.module_pop_size)
            for index, entry in enumerate(self.library.query(entry_type=MODULE, input_width=self.in_ports, output_width=self.out_ports, limit=wanted)):
                genomes[index] = graft(entry, tracker)
        modules = [LiveModule(genome=genome) for genome in genomes]
        state = HierarchicalState(modules=modules, module_innovations=tracker, comp_innovations=InnovationTracker(_next_node_id=0), rng=rng)
        self._speciate_only(state)
        return state

    def _speciate_only(self, state: HierarchicalState) -> None:
        """Refresh species membership/champions WITHOUT reproducing (used at seed and restore)."""
        genomes = [module.genome for module in state.modules]
        fitnesses = [module.fitness for module in state.modules]
        plans = self.evolver.speciate(genomes, fitnesses, rng=state.rng, elitism=self.module_elitism, pop_size=self.module_pop_size)
        state.module_species_history.append({plan.species_id: len(plan.members) for plan in plans})
        state.species_members = {plan.species_id: list(plan.members) for plan in plans}
        state.species_champion_index = {}
        state.species_champions = {}
        for plan in plans:
            champion = max(plan.members, key=lambda index: fitnesses[index])
            state.species_champion_index[plan.species_id] = champion
            state.species_champions[plan.species_id] = state.modules[champion].genome

    # --- ref plumbing -------------------------------------------------------------------------------

    def ref_catalog(self, state: HierarchicalState) -> list[RefSpec]:
        catalog = [RefSpec(f"live:{species_id}", self.in_ports, self.out_ports) for species_id in sorted(state.species_champions)]
        if self.library is not None:
            for entry in self.library.query():
                in_width = sum(item["width"] for item in entry.io["inputs"])
                catalog.append(RefSpec(f"library:{entry.key}", in_width, entry.io["output"]["width"]))
        return catalog

    def _repair_refs(self, comp: CompositionGenome, state: HierarchicalState) -> CompositionGenome:
        """Re-point MODULE nodes whose live species died to a random living species (logged)."""
        from dataclasses import replace

        live = sorted(state.species_champions)
        if not live:
            return comp
        child = None
        for node_id in comp.module_ids:
            node = comp.nodes[node_id]
            if node.ref.startswith("live:") and int(node.ref.removeprefix("live:")) not in state.species_champions:
                if child is None:
                    child = comp.clone()
                replacement = live[state.rng.randrange(len(live))]
                child.nodes[node_id] = replace(node, ref=f"live:{replacement}")
                state.repaired_refs += 1
        return comp if child is None else child

    def _context(self, state: HierarchicalState) -> AssemblyContext:
        def live_resolver(species_id: str) -> Genome:
            genome = state.species_champions.get(int(species_id))
            if genome is None:
                raise CompositionAssemblyError(f"live species {species_id} no longer exists")
            return genome

        return AssemblyContext(bank_columns={}, live_resolver=live_resolver, library=self.library, max_inline_depth=self.max_inline_depth)

    # --- assessment ---------------------------------------------------------------------------------

    def _assess(self, comp: CompositionGenome, spec: CompTaskSpec, state: HierarchicalState, *, train: bool) -> AssessedComposition:
        ctx = self._context(state)
        ctx.bank_columns = dict(spec.bank_columns)
        try:
            net = assemble(comp, ctx, spec.n_inputs)
        except CompositionAssemblyError as error:
            logger.debug("composition floored at assembly: %s", error)
            return AssessedComposition(comp=comp, metrics={}, fitness=FLOOR_FITNESS, net=None)
        if train:
            # Train ops must NOT write back here: their genome-level writeback expects a flat Genome.
            # Glue lands on the comp genes below; module weights flow via the champion policy.
            comp, _module = self._train(comp, net, spec, state)
            comp = writeback_composition(comp, net)
        metrics = self.evolver.evaluate_op(comp, net, _CompositionEvalAdapter(spec))
        fitness = self.evolver.fitness(comp, metrics)  # CompositionGenome duck-types complexity/hidden_ids
        return AssessedComposition(comp=comp, metrics=metrics, fitness=fitness, net=net)

    def _train(self, comp: CompositionGenome, net: ComposedNet, spec: CompTaskSpec, state: HierarchicalState) -> tuple[CompositionGenome, ComposedNet]:
        self.evolver.train_op(comp, net, spec.encoded, rng=state.rng, writeback=False)
        return comp, net

    # --- attribution and writeback --------------------------------------------------------------------

    def _attribute(self, assessed: list[AssessedComposition], state: HierarchicalState) -> None:
        per_ref: dict[str, list[float]] = {}
        for item in assessed:
            if item.fitness <= FLOOR_FITNESS:
                continue
            for ref in set(item.comp.refs()):
                per_ref.setdefault(ref, []).append(item.fitness)
        attribution = {ref: (max(values) if self.attribution_mode == "max" else sum(values) / len(values)) for ref, values in per_ref.items()}

        referenced_species: set[int] = set()
        for ref, value in attribution.items():
            if ref.startswith("live:"):
                species_id = int(ref.removeprefix("live:"))
                if species_id in state.species_members:
                    referenced_species.add(species_id)
                    champion = state.species_champion_index.get(species_id)
                    for index in state.species_members[species_id]:
                        if index == champion:
                            state.modules[index].fitness = value
                        else:
                            state.modules[index].fitness *= self.decay
            elif ref.startswith("library:") and self.library is not None:
                self.library.bump_stats(ref.removeprefix("library:"), value)
        for species_id, members in state.species_members.items():
            if species_id not in referenced_species:
                for index in members:
                    state.modules[index].fitness *= self.decay

    def _module_writeback(self, assessed: list[AssessedComposition], state: HierarchicalState) -> None:
        """Champion-only Lamarckianism: ONLY the generation's best composition writes its trained
        live-module weights back, so one bad gradient run cannot shred a shared module."""
        if self.writeback_mode != "champion":
            return
        best = max(assessed, key=lambda item: item.fitness)
        if best.net is None or best.fitness <= FLOOR_FITNESS:
            return
        for ref, inner in best.net.inner_modules.items():
            if not ref.startswith("live:"):
                continue
            species_id = int(ref.removeprefix("live:"))
            champion_genome = state.species_champions.get(species_id)
            if champion_genome is None:
                continue
            updated = _writeback(champion_genome, inner)
            state.species_champions[species_id] = updated
            champion_index = state.species_champion_index.get(species_id)
            if champion_index is not None and champion_index < len(state.modules):
                state.modules[champion_index].genome = updated

    # --- reproduction -----------------------------------------------------------------------------------

    def _module_context(self, state: HierarchicalState) -> MutationContext:
        return MutationContext(innovations=state.module_innovations, activations=self.evolver.activations, default_activation=self.evolver.default_activation)

    def advance_modules(self, state: HierarchicalState) -> None:
        """One module generation: speciate on attributed fitness, then reproduce per species with
        the SAME registered operators the flat loop uses."""
        genomes = [module.genome for module in state.modules]
        fitnesses = [module.fitness for module in state.modules]
        plans = self.evolver.speciate(genomes, fitnesses, rng=state.rng, elitism=self.module_elitism, pop_size=self.module_pop_size)
        state.module_species_history.append({plan.species_id: len(plan.members) for plan in plans})

        ctx = self._module_context(state)
        next_modules: list[LiveModule] = []
        members_map: dict[int, list[int]] = {}
        champion_index: dict[int, int] = {}
        champions: dict[int, Genome] = {}
        for plan in plans:
            members = sorted(plan.members, key=lambda index: fitnesses[index], reverse=True)
            champions[plan.species_id] = state.modules[members[0]].genome
            species_fitness = fitnesses[members[0]]
            new_indices: list[int] = []
            for index in members[: plan.n_elites]:
                new_indices.append(len(next_modules))
                next_modules.append(state.modules[index])
            if new_indices:
                champion_index[plan.species_id] = new_indices[0]
            if plan.n_offspring > 0:
                species_genomes = [genomes[index] for index in members]
                species_fitnesses = [fitnesses[index] for index in members]
                parents = self.evolver.selection_op(species_genomes, species_fitnesses, rng=state.rng, count=2 * plan.n_offspring)
                for k in range(plan.n_offspring):
                    child = self.evolver.crossover_op(parents[2 * k], parents[2 * k + 1], rng=state.rng)
                    child = self.evolver.mutation(child, ctx, rng=state.rng)
                    child = make_acyclic(child)
                    new_indices.append(len(next_modules))
                    # Offspring inherit a discounted species fitness until a composition references them.
                    next_modules.append(LiveModule(genome=child, fitness=species_fitness * self.offspring_discount))
            members_map[plan.species_id] = new_indices
        state.modules = next_modules
        state.species_members = members_map
        state.species_champion_index = champion_index
        state.species_champions = champions

    def _reproduce_comps(self, assessed: list[AssessedComposition], spec: CompTaskSpec, state: HierarchicalState) -> list[AssessedComposition]:
        ordered = sorted(assessed, key=lambda item: item.fitness, reverse=True)
        # Elites keep their genes but are RE-EVALUATED (no gradient, no drift): their live refs
        # resolve against modules that moved since last generation, so cached fitness goes stale.
        next_generation = [self._assess(item.comp, spec, state, train=False) for item in ordered[: self.comp_elitism]]
        n_offspring = max(self.comp_pop_size - len(next_generation), 0)
        if n_offspring == 0:
            return next_generation
        comps = [item.comp for item in ordered]
        fitnesses = [item.fitness for item in ordered]
        parents = self.comp_selection_op(comps, fitnesses, rng=state.rng, count=2 * n_offspring)
        comp_ctx = CompMutationContext(innovations=state.comp_innovations, ref_catalog=self.ref_catalog(state))
        for k in range(n_offspring):
            if state.rng.random() < self.comp_crossover_rate:
                child = self.comp_crossover_op(parents[2 * k], parents[2 * k + 1], rng=state.rng)
            else:
                child = parents[2 * k].clone()
            child = self.comp_mutation(child, comp_ctx, rng=state.rng)
            child = self._repair_refs(child, state)
            next_generation.append(self._assess(child, spec, state, train=True))
        return next_generation

    # --- the per-task drive ---------------------------------------------------------------------------

    def run_task(
        self,
        spec: CompTaskSpec,
        state: HierarchicalState,
        *,
        budget: int,
        stop: Callable[[int, AssessedComposition], bool] | None = None,
        seed_comps: list[CompositionGenome] | None = None,
        on_generation: Callable[[int, AssessedComposition, float], None] | None = None,
    ) -> AssessedComposition:
        """Evolve a composition population against one task for up to `budget` generations."""
        population: list[CompositionGenome] = [comp.clone() for comp in (seed_comps or [])][: self.comp_pop_size]
        while len(population) < self.comp_pop_size:
            population.append(minimal_composition(spec.input_specs, spec.output_ref, spec.output_width, state.comp_innovations, state.rng, glue_scale=self.glue_scale))
        population = [self._repair_refs(comp, state) for comp in population]

        assessed = [self._assess(comp, spec, state, train=True) for comp in population]
        self._attribute(assessed, state)
        self._module_writeback(assessed, state)
        best = max(assessed, key=lambda item: item.fitness)

        for generation in range(budget):
            if stop is not None and stop(generation, best):
                break
            if self.advance_every > 0 and generation % self.advance_every == self.advance_every - 1:
                self.advance_modules(state)
            assessed = self._reproduce_comps(assessed, spec, state)
            self._attribute(assessed, state)
            self._module_writeback(assessed, state)
            generation_best = max(assessed, key=lambda item: item.fitness)
            if generation_best.fitness > best.fitness:
                best = generation_best
            if on_generation is not None:
                mean = sum(item.fitness for item in assessed) / len(assessed)
                on_generation(generation, generation_best, mean)
            state.generation += 1
        return best


class _CompositionEvalAdapter:
    """The minimal Adapter surface the evaluate stage needs (`evaluate(module)`); decode is owned
    by the loop because it requires per-candidate assembly contexts."""

    def __init__(self, spec: CompTaskSpec) -> None:
        self.encoded = spec.encoded
        self.encoder = spec.encoder
        self.n_inputs = spec.n_inputs
        self.n_outputs = spec.output_width

    def evaluate(self, module: ComposedNet) -> dict[str, float]:
        return evaluate(module, self.encoded, self.encoder)


# --- state serialization ------------------------------------------------------------------------------


def state_to_dict(state: HierarchicalState) -> dict[str, Any]:
    return {
        "generation": state.generation,
        "modules": [{"genome": genome_to_dict(module.genome), "fitness": module.fitness} for module in state.modules],
        "module_innovations": state.module_innovations.to_dict(),
        "comp_innovations": state.comp_innovations.to_dict(),
        "species_members": {str(species_id): members for species_id, members in state.species_members.items()},
        "species_champion_index": {str(species_id): index for species_id, index in state.species_champion_index.items()},
        "species_champions": {str(species_id): genome_to_dict(genome) for species_id, genome in state.species_champions.items()},
        "module_species_history": [{str(species_id): count for species_id, count in snapshot.items()} for snapshot in state.module_species_history],
        "repaired_refs": state.repaired_refs,
    }


def state_from_dict(data: dict[str, Any], rng: random.Random) -> HierarchicalState:
    return HierarchicalState(
        modules=[LiveModule(genome=genome_from_dict(item["genome"]), fitness=float(item["fitness"])) for item in data["modules"]],
        module_innovations=InnovationTracker.from_dict(data["module_innovations"]),
        comp_innovations=InnovationTracker.from_dict(data["comp_innovations"]),
        rng=rng,
        generation=int(data["generation"]),
        species_members={int(species_id): [int(index) for index in members] for species_id, members in data["species_members"].items()},
        species_champion_index={int(species_id): int(index) for species_id, index in data["species_champion_index"].items()},
        species_champions={int(species_id): genome_from_dict(genome) for species_id, genome in data["species_champions"].items()},
        module_species_history=[{int(species_id): int(count) for species_id, count in snapshot.items()} for snapshot in data["module_species_history"]],
        repaired_refs=int(data.get("repaired_refs", 0)),
    )


# --- factory ---------------------------------------------------------------------------------------------


def build_hierarchical(config: dict[str, Any]) -> HierarchicalLoop:
    from ardevo.evolution import composition, selection

    evolution = config.get("evolution", {})
    comp_cfg = evolution.get("composition", {})
    modules_cfg = evolution.get("modules", {})

    comp_selection_cfg = comp_cfg.get("selection", {})
    comp_selection_op = partial(
        selection.SELECTION.get(comp_selection_cfg.get("kind", "tournament")),
        **{k: v for k, v in comp_selection_cfg.items() if k != "kind"},
    )
    comp_crossover_cfg = comp_cfg.get("crossover", {})
    comp_crossover_op = partial(
        composition.COMP_CROSSOVER.get(comp_crossover_cfg.get("kind", "none")),
        **{k: v for k, v in comp_crossover_cfg.items() if k not in ("kind", "rate")},
    )
    comp_mutation_cfg = comp_cfg.get("mutation", {})
    mutators: list[Callable[..., CompositionGenome]] = [
        partial(composition.COMP_MUTATION.get(name), **_bind_prefixed(comp_mutation_cfg, name)) for name in comp_mutation_cfg.get("operators", [])
    ]

    return HierarchicalLoop(
        evolver=build_evolver(config),
        comp_pop_size=int(comp_cfg.get("pop_size", 16)),
        comp_elitism=int(comp_cfg.get("elitism", 2)),
        comp_selection_op=comp_selection_op,
        comp_crossover_op=comp_crossover_op,
        comp_crossover_rate=float(comp_crossover_cfg.get("rate", 0.2)),
        comp_mutation=CompMutationPipeline(mutators),
        max_inline_depth=int(comp_cfg.get("max_inline_depth", 4)),
        glue_scale=float(comp_cfg["glue_scale"]) if "glue_scale" in comp_cfg else None,
        module_pop_size=int(modules_cfg.get("pop_size", evolution.get("pop_size", 32))),
        module_elitism=int(modules_cfg.get("elitism", evolution.get("elitism", 1))),
        in_ports=int(modules_cfg.get("in_ports", 4)),
        out_ports=int(modules_cfg.get("out_ports", 2)),
        advance_every=int(modules_cfg.get("advance_every", 3)),
        writeback_mode=str(modules_cfg.get("writeback", "champion")),
        attribution_mode=str(modules_cfg.get("attribution", "max")),
        decay=float(modules_cfg.get("decay", 0.95)),
        offspring_discount=float(modules_cfg.get("offspring_discount", 0.9)),
        seed_fraction=float(modules_cfg.get("seed_fraction", 0.25)),
    )
