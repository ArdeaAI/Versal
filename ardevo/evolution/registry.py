"""Operator registries and the evolver factory: the heart of the lego-block design.

Each evolutionary stage (init, selection, crossover, mutation, train, fitness) is a function
registered by name in a `Registry`. `build_evolver` reads `config.toml`, resolves the chosen
operators, binds their parameters, and assembles a thin `Evolver`. Adding a behavior means
registering one function and naming it in config; the loop never changes.
"""

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

if TYPE_CHECKING:
    from ardevo.evolution.evolver import Evolver

T = TypeVar("T")


class Registry(Generic[T]):
    """A name -> implementation table populated by the `@register` decorator."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(item: T) -> T:
            self._items[name] = item
            return item

        return decorator

    def get(self, name: str) -> T:
        if name not in self._items:
            available = ", ".join(sorted(self._items)) or "(none registered)"
            raise KeyError(f"unknown {self.kind} operator {name!r}; available: {available}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)


def _bind_prefixed(table: dict[str, Any], name: str) -> dict[str, Any]:
    """Collect `{name}_{param}` keys into `{param: value}` (e.g. add_node_prob -> {prob: ...})."""
    prefix = f"{name}_"
    return {key[len(prefix) :]: value for key, value in table.items() if key.startswith(prefix)}


def build_loop(config: dict[str, Any]) -> Any:
    """Resolve the configured top-level loop strategy (`[evolution] loop`, default "hierarchical")."""
    from ardevo.evolution import loop

    kind = config.get("evolution", {}).get("loop", "hierarchical")
    return loop.LOOP.get(kind)(config)


def build_evolver(config: dict[str, Any]) -> "Evolver":
    """Assemble an `Evolver` from the nested `[evolution]`/`[substrate]`/`[fitness]` config tables."""
    # Local imports keep the operator modules (which import Registry from here) free of a cycle,
    # while still triggering their @register side effects.
    from ardevo.evolution import crossover, evaluate, fitness, init, mutation, selection, speciation, train
    from ardevo.evolution.evolver import Evolver

    evolution = config.get("evolution", {})
    substrate = config.get("substrate", {})
    fitness_cfg = config.get("fitness", {})

    activations = substrate.get("available_activations", ["tanh", "relu", "sigmoid", "identity"])
    default_activation = substrate.get("default_activation", "tanh")

    init_cfg = evolution.get("init", {})
    init_op = partial(
        init.INIT.get(init_cfg.get("kind", "minimal")),
        default_activation=default_activation,
        **{k: v for k, v in init_cfg.items() if k != "kind"},
    )

    selection_cfg = evolution.get("selection", {})
    selection_op = partial(
        selection.SELECTION.get(selection_cfg.get("kind", "tournament")),
        **{k: v for k, v in selection_cfg.items() if k != "kind"},
    )

    crossover_cfg = evolution.get("crossover", {})
    crossover_op = partial(
        crossover.CROSSOVER.get(crossover_cfg.get("kind", "none")),
        **{k: v for k, v in crossover_cfg.items() if k not in ("kind", "rate")},
    )

    mutation_cfg = evolution.get("mutation", {})
    operator_names = mutation_cfg.get("operators", [])
    mutation_pipeline: mutation.MutationPipeline | mutation.AdaptiveMutationPipeline
    if bool(mutation_cfg.get("self_adaptive", False)):
        # Lever F: rates become per-genome strategy genes. Each operator's configured `prob` seeds its
        # starting rate; the log-normal step size and clamp bounds are the only new knobs.
        specs = [(name, mutation.MUTATION.get(name), _bind_prefixed(mutation_cfg, name)) for name in operator_names]
        mutation_pipeline = mutation.AdaptiveMutationPipeline(
            specs,
            learning_rate=float(mutation_cfg.get("self_adaptive_learning_rate", 0.1)),
            min_rate=float(mutation_cfg.get("self_adaptive_min", 0.001)),
            max_rate=float(mutation_cfg.get("self_adaptive_max", 1.0)),
        )
    else:
        mutators = [partial(mutation.MUTATION.get(name), **_bind_prefixed(mutation_cfg, name)) for name in operator_names]
        mutation_pipeline = mutation.MutationPipeline(mutators)

    train_cfg = evolution.get("train", {})
    train_kind = train_cfg.get("kind", "none")
    train_population_op = None
    sequential_params = {k: v for k, v in train_cfg.items() if k in ("steps", "lr", "writeback", "weight_decay")}
    if train_kind in train.TRAIN_POPULATION.names() and bool(train_cfg.get("batched", True)):
        from ardevo.utils.device import resolve_compute_device

        # Population trainers batch a whole generation; single-candidate calls (resume re-scoring,
        # task switches) and the batch program's serial fallback still need a sequential op. Bind
        # the SAME kind's sequential form when one exists (gradient_refine must keep deep
        # supervision on refine genomes everywhere), else `gradient`. The compute device resolves
        # from the run config unless the table pins one, so a LatticeCUDA run lands population
        # training on cuda with zero config edits; the padding budget widens on GPU (CIFAR-scale
        # n fits comfortably once the batch stacks compact [P, n, h] columns).
        population_params = {k: v for k, v in train_cfg.items() if k not in ("kind", "batched")}
        if "device" not in population_params:
            population_params["device"] = str(resolve_compute_device(config))
        if "max_padded_nodes" not in population_params:
            population_params["max_padded_nodes"] = 1024 if population_params["device"] == "cpu" else 4096
        train_population_op = partial(train.TRAIN_POPULATION.get(train_kind), **population_params)
        sequential_kind = train_kind if train_kind in train.TRAIN.names() else "gradient"
        train_op = partial(train.TRAIN.get(sequential_kind), **sequential_params)
    elif train_kind in train.TRAIN_POPULATION.names():
        # `batched = false`: the kill-switch back to the pool-only path. Population-only kinds
        # degrade to their sequential form (`gradient` when no same-name TRAIN op exists).
        sequential_kind = train_kind if train_kind in train.TRAIN.names() else "gradient"
        train_op = partial(train.TRAIN.get(sequential_kind), **sequential_params)
    else:
        train_op = partial(
            train.TRAIN.get(train_kind),
            **{k: v for k, v in train_cfg.items() if k != "kind"},
        )

    evaluate_cfg = evolution.get("evaluate", {})
    evaluate_op = partial(
        evaluate.EVALUATE.get(evaluate_cfg.get("kind", "standard")),
        **{k: v for k, v in evaluate_cfg.items() if k != "kind"},
    )

    components = [(fitness.FITNESS.get(name), float(fitness_cfg.get(f"w_{name}", 1.0))) for name in fitness_cfg.get("components", [])]
    # `[fitness] objectives` names the Pareto vector (need not overlap `components`); absent = scalar-only.
    objective_components = [(name, fitness.FITNESS.get(name)) for name in fitness_cfg.get("objectives", [])]
    aggregator = fitness.FitnessAggregator(components, objective_components)

    # `[evolution.novelty]`: absent table (or enabled = false) means None, the byte-identical off
    # switch; the Evolver's post-assess hook then never computes a descriptor. A bare table header
    # parses to {} and must mean ON with defaults, so gate on presence, never on dict truthiness.
    novelty_cfg = evolution.get("novelty")
    novelty_config = None
    if novelty_cfg is not None and bool(novelty_cfg.get("enabled", True)):
        from ardevo.evolution.novelty import NoveltyConfig

        novelty_config = NoveltyConfig(
            k=int(novelty_cfg.get("k", 15)),
            archive_cap=int(novelty_cfg.get("archive_cap", 256)),
            probe_rows=int(novelty_cfg.get("probe_rows", 64)),
        )

    # Speciation is stateful (it remembers a representative per species), so its registry entries are
    # factories: resolve the kind, then build the configured speciator instance.
    speciation_cfg = evolution.get("speciation", {})
    speciate = speciation.SPECIATION.get(speciation_cfg.get("kind", "none"))(**{k: v for k, v in speciation_cfg.items() if k != "kind"})

    from ardevo.utils.device import resolve_worker_count

    return Evolver(
        pop_size=int(evolution.get("pop_size", 64)),
        elitism=int(evolution.get("elitism", 1)),
        assess_workers=resolve_worker_count(evolution.get("assess_workers", 0)),
        library_dir=str(config.get("library_dir", "library")),
        seed=int(config.get("seed", 0)),
        init_op=init_op,
        selection_op=selection_op,
        crossover_op=crossover_op,
        mutation=mutation_pipeline,
        train_op=train_op,
        train_population_op=train_population_op,
        fitness=aggregator,
        evaluate_op=evaluate_op,
        speciate=speciate,
        activations=activations,
        default_activation=default_activation,
        novelty=novelty_config,
        halving_stages=[float(fraction) for fraction in evolution.get("halving_stages", [])],
        halving_keep=float(evolution.get("halving_keep", 0.5)),
    )
