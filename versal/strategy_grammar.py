"""Grammar-derived seed search strategy."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

from versal.dataset.icarus import Task
from versal.evolution.composition import CompositionGenome
from versal.evolution.genome import Genome
from versal.evolution.loop import CompTaskSpec
from versal.strategy_common import StrategyResult, StrategyRuntime, _restamp_composition
from versal.strategy_composition import CompositionStrategy


@dataclass
class GrammarStrategy:
    """Search programs assembled only from motifs independently rediscovered by evolution."""

    direct: Callable[..., StrategyResult]
    blind_query: bool = False
    max_productions: int = 12
    candidates_per_production: int = 3
    mutation_steps: int = 2
    module_sizes: tuple[int, ...] = (3, 4)
    composition_sizes: tuple[int, ...] = (2, 3, 4)
    min_lineage_support: int = 2
    per_entry_cap: int = 5000
    name: str = "grammar"
    _library_keys: tuple[str, ...] = field(default=(), init=False, repr=False)
    _grammar: Any = field(default=None, init=False, repr=False)

    def _programs(self, runtime: StrategyRuntime) -> list[Any]:
        from versal.grammar import crossover_program, mutate_program, rebuild_grammar, seed_program

        keys = tuple(runtime.library.keys())
        if self._grammar is None or keys != self._library_keys:
            self._grammar = rebuild_grammar(
                runtime.library,
                module_sizes=self.module_sizes,
                composition_sizes=self.composition_sizes,
                min_lineage_support=self.min_lineage_support,
                per_entry_cap=self.per_entry_cap,
            )
            self._library_keys = keys
        productions = sorted(self._grammar.productions, key=lambda item: (-item.mdl_gain, -item.support, item.key))[: self.max_productions]
        programs: list[Any] = []
        seen: set[str] = set()
        for production in productions:
            seed = seed_program(production)
            candidates = [seed]
            for _ in range(self.candidates_per_production - 1):
                candidate = seed
                for _step in range(self.mutation_steps):
                    candidate = mutate_program(candidate, self._grammar, rng=runtime.state.rng)
                candidates.append(candidate)
            for candidate in candidates:
                key = repr(candidate.to_dict())
                if key not in seen:
                    seen.add(key)
                    programs.append(candidate)
        # Aligned crossover is useful only once mutation has produced multi-production parents.
        parents = list(programs)
        for left, right in zip(parents[::2], parents[1::2]):
            child = crossover_program(left, right, self._grammar, rng=runtime.state.rng)
            key = repr(child.to_dict())
            if key not in seen:
                seen.add(key)
                programs.append(child)
        return programs

    @staticmethod
    def _composition_compatible(comp: CompositionGenome, spec: CompTaskSpec) -> bool:
        if len(comp.output_ids) != 1 or comp.nodes[comp.output_ids[0]].in_width != spec.output_width:
            return False
        for node_id in comp.input_ids:
            node = comp.nodes[node_id]
            columns = spec.bank_columns.get(node.ref)
            if columns is None or len(columns) != node.out_width:
                return False
        return True

    def __call__(
        self,
        task: Task,
        spec: CompTaskSpec,
        runtime: StrategyRuntime,
        *,
        budget: int,
        seed_comps: list | None = None,
    ) -> StrategyResult:
        from versal.grammar import GrammarError, compile_program

        module_seeds: list[Genome] = []
        comp_seeds: list[CompositionGenome] = []
        for program in self._programs(runtime):
            try:
                compiled = compile_program(program, self._grammar, library=runtime.library, rng=runtime.state.rng)
            except (GrammarError, KeyError, ValueError):
                continue
            if isinstance(compiled, Genome) and len(compiled.input_ids) == spec.n_inputs and len(compiled.output_ids) == spec.output_width:
                module_seeds.append(compiled)
            elif isinstance(compiled, CompositionGenome) and self._composition_compatible(compiled, spec):
                comp_seeds.append(_restamp_composition(compiled, runtime.state.comp_innovations))
        if not module_seeds and not comp_seeds:
            return StrategyResult(strategy=self.name, metric=0.0, generations_used=0, champion_metrics={"grammar_productions": float(len(self._grammar.productions))})

        results: list[StrategyResult] = []
        used = 0
        if module_seeds:
            allocation = budget if not comp_seeds else max(1, budget // 2)
            result = self.direct(task, spec, runtime, budget=allocation, seed_genomes=module_seeds)
            used += result.generations_used
            results.append(result)
            if runtime.accepted(SimpleNamespace(metrics=result.champion_metrics)):
                result.strategy = self.name
                result.generations_used = used
                return result
        remaining = max(0, budget - used)
        if comp_seeds and remaining > 0:
            result = CompositionStrategy(blind_query=self.blind_query)(task, spec, runtime, budget=remaining, seed_comps=[*(seed_comps or []), *comp_seeds])
            used += result.generations_used
            results.append(result)
        winner = max(results, key=lambda item: item.metric)
        winner.strategy = self.name
        winner.generations_used = used
        winner.champion_metrics["grammar_productions"] = float(len(self._grammar.productions))
        winner.champion_metrics["grammar_programs"] = float(len(module_seeds) + len(comp_seeds))
        return winner
