# ArdEVO

Playground for evolutionary algo ML testing

The purpose of this is to test different ENAS methods for topology itself.

There is a custom dataset I made that goes through MANY rungs across modalities. The idea is to single in on a
search algorithm that seems to be able to generalize over the different rungs of the dataset, growing the topology
in minimum complexity needed as we go up each difficulty rung until we find at least an ideal algorithm, potentially
modified by me to something novel, that can be used across every rung and produce a significant score.

The dataset is `https://huggingface.co/datasets/Ardea/Icarus-dataset` and its vendored runtime lives at
`ardevo/dataset/icarus.py`. NOTE THAT the pole and double-pole rungs, which are normally continuous RL-type
problems, have been transformed into time-series differentiable tasks. You also need to pay SPECIAL ATTENTION to
the types in the metadata of that dataset that allow it to describe itself.

We want to use ClearML for this as well as much as we can make use of it. I already have my config set up for that.

## The workflow

There is ONE run mode: the orchestrated overmind evolver. The canonical configuration is
`configs/orchestrated_overmind_all_features.toml`, also the default when `uv run app` gets no `--config`.
Other files in `configs/` are retained as paper-reproduction, diagnostic, smoke, and ablation arms.

```bash
uv sync --group dev
uv run app                                        # the orchestrated overmind run (all 18 rungs)
uv run app --config configs/preflight.toml        # 180-task MonadMetal/ClearML production canary
uv run app --resume results/<ts>_orchestrated     # continue a run from its rolling checkpoint
uv run render --overmind                          # re-render the library + the routed model portrait
uv run motif_census --render                      # mine recurring motifs -> library/motifs.json + atlas
uv run rung_doctor --rungs 1-18 --n-tasks 2       # probe rung loadability/shapes without a run
uv run library_gc --dry-run                       # sweep unreferenced tombstones (prunes dead router vertices)
uv run benchmark                                  # measured throughput truths
uv run benchmark --calibrate-compute --calibration-only --compute-profile results/compute_policy.json
uv run runtime_inventory --check                  # verify config, registry, CLI, and runtime-path inventory
```

Set `[run] clearml = true` to track in ClearML; it degrades gracefully offline. Machine env maps to a queue:
`MonadMetal`/`MonadCPU`/`local` run locally, `LatticeCPU`/`LatticeCUDA` enqueue remotely.

The search grows a network *topology* from nothing and lets structural mutations add nodes/edges. A per-generation
`train` step tunes each candidate's weights by gradient before scoring, so **evolution searches structure and the
gradient owns the weights** (pure weight-evolution stalls even on XOR; random weight mutation *fights* the
gradient, so it is left out when training is on).

## The orchestrated ladder

Every scheduled task runs an escalation ladder. **The library is the memory and the DSL**: every solved thing must
end up in it as a reusable entry, and it survives across runs (delete `library/` to start the search space cold).

1. **LOOKUP**: the persistent, file-based **library** (`library/`) is queried by structural I/O signature
   (value_type + axes + widths, NEVER rung or task name). A stored solution that still clears the accept
   threshold is reused at zero generations: a solved task STAYS solved. With `[orchestrator.refine] budget_k > 0`
   (LEARN MODE), a hit spends a small per-entry-decaying budget of evolution seeded from the stored solution
   trying to strictly beat it (metric, then robustness, then LOWER complexity); a failed refinement returns the
   original hit, so a task never regresses.
2. **EVOLVE**: the strategy ladder (`[orchestrator] evolve = ["routed", "grammar", "direct", "composition"]`, config order,
   first success wins, unspent budget rolls on):
   - **routed**: the overmind (below). A zero-shot clear costs 0 generations.
   - **grammar**: typed graph programs induced only from motifs rediscovered in independent library lineages and
     admitted by a positive MDL gain. Empty or interface-incompatible grammars cost zero generations.
   - **direct**: the flat recipe on the task's REAL I/O; grows structure (the two-spirals class), routes
     TIME-axis tasks through the stepped recurrent substrate, stamps grid coordinates so the geometry operators
     (`add_local_node`, `add_local_connection`, `add_shared_motif`: a structural convolution prior) bite, and
     admits TASK-SHAPED modules. `assess_workers` process-pools the per-genome work across cores.
   - **composition**: a per-task population of compositions (small DAGs whose MODULE nodes reference live module
     species or library entries, wired by trainable linear GLUE) co-evolves with ONE shared mini-model
     population; fitness flows DOWN as attribution, and only the champion writes module weights back.
3. **DECOMPOSE**: on a stall, registered operators (`output_slices`, `input_subsets`, `time_windows`,
   `spatial_patches` for grid->grid) split the task into valid subtasks and the orchestrator RECURSES on each
   (depth-capped); a solvability gate probes subtasks before committing budget. Accepted parts become frozen
   entries; the parent re-evolves seeded with a port-wired composition over them.
4. **ADMIT**: winners persist with provenance and a weight-robustness score. The archive admission
   (`[library] admission = "archive"`) niches entries by (io shape, behavior descriptor) so behaviorally DIVERSE
   stepping stones coexist instead of collapsing to the top-k by metric. Live module refs are snapshotted as
   frozen entries, so nothing in the library ever dangles. Recursion comes from the library, not new types:
   level 1 is mini-models, level 2 compositions of them, level 3+ compositions of compositions.

A depth-0 FAILURE above `[orchestrator.wall] min_metric` still leaves a trace: the best champion is shelved as a
below-bar STEPPING STONE and the next attempt on that signature warm-starts from it, so assaults on a hard family
accumulate trained weights and structure across attempts instead of restarting.

The run is observable end to end: `run_summary.json` gets a row for EVERY task and a rolling resume
`checkpoint.json` lands at the run root each task (library stats flush at the same boundary). New runs also pin
`config.toml` and the CLI-adjusted `config.effective.json`, each with a SHA256 sidecar; implicit `--resume` loads
those run-local snapshots rather than today's repository default.

## The overmind (routed substrate)

The `routed` strategy (`ardevo/routing.py`) is a sparse mixture-of-experts whose experts are FROZEN library
entries as vertices on a shared `d_model` bus, wired by trainable per-vertex adapters and a task-conditioned
top-k gate. Routing is a bounded `max_steps` unroll, so module-to-module pathways INCLUDING CYCLES are legal and
termination is structural. Per task it tries zero-shot first, then trains only adapters/gate/heads; across tasks
the state persists under `<library_dir>/router/` and resumes like the library.

**Routed wins distill or die** (`distill = true`): a router-space win only counts when its dominant pathway
builds into a `CompositionGenome` and verifies at the accept bar; the verified composition is admitted through
the ordinary rails and becomes a new routable vertex at the next sync. On a cold library, routed short-circuits
at zero cost, so evolution populates the vertex set first.

`uv run render --overmind` draws the whole routed model to `library/images/overmind.png` as a top-down flow
grid: input-adapter band up top, every expert a fully-embedded cell, output heads across the bottom, edges
widthed by observed lifetime gate traffic. Per-entry renders are recursive and dark: nested networks draw fully
inside translucent callout boxes green-lined to their footprints, and every render failure degrades to a labeled
opaque box, never an exception.

## Motif census

`uv run motif_census --render` mines RECURRING substructures across entries (exact permutation-canonical
fingerprints over labeled dataflow graphs, ESU enumeration up to k=5), which is how a spontaneously evolved
skip/gating/attention-like motif gets noticed. Motifs rank by support grouped by intrinsic diversity class
(`recurrent`/`gated`/`macro`/`mixed`/`uniform-*`), compositions mine with refs collapsed to level classes, and
the reuse census reports the growing vocabulary (who is built FROM whom). Report: `library/motifs.json`; atlas:
`library/images/motifs.png`.

## Substrate expressibility (grown, never pre-allocated)

- `aggregation = "product"` nodes: multiplicative gating, second-order interactions.
- True RECURRENCE: time-delayed edges + a stepped `RecurrentGraphNet` over TIME-axis tasks (`ardevo/temporal.py`),
  trained by plain BPTT through the ordinary gradient operator; inert under the flat decode.
- RECURSIVE DEPTH (TRM): `decode_refine` re-applies a tiny network to a static input `refine_steps` times,
  threading a latent and an answer back through recurrent edges, deep-supervised by `gradient_refine`
  (`refine_steps = 1` is byte-identical to feedforward).
- Library reuse three ways: grafted into the module pool at every lookup miss (`absorb_top_k`), inlined as
  unfrozen evolvable structure (`add_library_module`), or embedded as a single FROZEN MACRO NODE
  (`add_macro_node`): a whole library network behind one gene, the way an LSTM cell is a network inside a node.

## Lego-block evolution

Every stage of the generational loop is an independent, registered operator selected and tuned from
`configs/orchestrated_overmind_all_features.toml`. The loop runs: **select -> crossover -> mutate -> train -> evaluate ->
fitness -> replace**, with speciation shaping how offspring are allocated. To experiment, change a `kind`,
reorder `[evolution.mutation].operators`, retune a weight, or register one new function in the matching
registry; the loop itself never changes.

`runtime_inventory.json` is the machine-generated source of truth for the complete registry and configuration
surface. Refresh it with `uv run runtime_inventory --write` whenever a registered operator, config key, console
script, or documented persistent path changes; CI-style checks should use `--check`.

| Stage | Config section | Registered options |
|---|---|---|
| loop | `[evolution] loop` | `hierarchical` (compositions + shared module pool) |
| evolve strategy | `[orchestrator] evolve` | `routed` (the overmind), `grammar`, `direct`, `composition` |
| init | `[evolution.init]` | `minimal`, `factored`, `sparse`, `cppn` |
| selection | `[evolution.selection]` | `tournament`, `truncation`, `nsga2` |
| crossover | `[evolution.crossover]` | `none`, `neat` |
| mutation | `[evolution.mutation]` | NEAT connection/node growth; rich/deep/hinted/relation/local growth; recurrence; activation/aggregation changes; library/macro reuse; shared-motif tie/untie; remove, toggle, and prune/regrow; refinement-depth and weight mutation |
| train | `[evolution.train]` | `none`, `gradient`, `gradient_scheduled`, `gradient_refine`; population trainers `gradient_batched`, `gradient_refine`, `gradient_scheduled` |
| evaluate | `[evolution.evaluate]` | `standard`, `weight_samples`, `augmented_vote`, `hybrid` |
| speciation | `[evolution.speciation]` | `none`, `neat` (compatibility threshold auto-targets a species count) |
| schedule | `[schedule]` | `random`, `round_robin`, `interleave_rungs`, `regret` |
| comp mutation | `[evolution.composition.mutation]` | `add_module_node`, `switch_ref`, `add_comp_edge`, `toggle_comp_edge`, `perturb_glue` (crossover: `none`, `comp_neat`) |
| decompose | `[orchestrator] decompose` | `output_slices`, `input_subsets`, `time_windows`, `spatial_patches` (grid->grid bands) |
| library admission | `[library] admission` | `accept_all` (legacy), `default` (floors + flat per-signature cap), `archive` (open-ended QD: per-(io, behavior-niche) diversity) |
| fitness | `[fitness]` | support/query accuracy and loss variants, sample accuracy/loss, robustness, novelty, connection cost, and complexity/hidden penalties |

Notes from getting this to actually grow useful topologies: `add_rich_node` only widens a layer, so depth-needing
tasks (two-spirals) require `add_deep_node`; `toggle_connection` is the only operator that prunes, so a complexity
penalty needs it to simplify; and `neat` speciation auto-adjusts its threshold (a fixed one fractures the
population into singletons and starves reproduction). The torch substrate is vectorized (level-wise matmuls).

Hardware choices are measured, not inferred from the machine name. The calibration command above compares the
real process-pool path with scheduled population training on the available CPU, Metal, or CUDA device, verifies
trained-weight drift, and writes a hardware-fingerprinted policy only when an alternative is at least 15% faster.
Stale or absent profiles preserve the process-pool path. Population training uses adaptive microbatches and
falls back to the semantically identical serial operator on recognized allocator failures. Library I/O is cached
and hot stats writes are deferred to one flush per task; structural admissions stay immediately durable.

On the prior art: CoDeepNEAT's two-population idea survives here as compositions-referencing-species and fitness
attribution; WANN's weight-agnostic insight survives as the robustness metric and the `weight_samples`/`hybrid`
evaluate stage. Neither is implemented literally. evox was evaluated and skipped (fixed-vector optimization
only); tensorneat's padded-tensor trick was ported natively as `gradient_batched`.

## Project structure

```
ardevo/
├── dataset/
│   └── icarus.py       # vendored Icarus runtime (generated; edit upstream, not here)
├── evolution/
│   ├── genome.py       # NEAT-style Genome (node/connection genes incl. aggregation + recurrence) + DAG helpers
│   ├── registry.py     # Registry + build_evolver / build_loop factories
│   ├── init.py         # population-seeding operators (+ grid coordinate stamping)
│   ├── selection.py    # parent-selection operators
│   ├── crossover.py    # recombination operators
│   ├── mutation.py     # structural + weight mutators (incl. aggregation flips, recurrent edges, library grafts)
│   ├── train.py        # weight-optimization operators (gradient / gradient_refine / gradient_batched / none)
│   ├── evaluate.py     # metrics operators (standard / weight_samples / hybrid robustness scoring)
│   ├── fitness.py      # fitness components + weighted aggregator
│   ├── composition.py  # CompositionGenome: the recursive representation (modules + glue), assembly, operators
│   ├── loop.py         # LOOP registry: HierarchicalLoop (attribution, champion writeback)
│   ├── multitask.py    # the task pool: TaskEntry facts + defensive rung-by-rung loading (SkippedRung rows)
│   ├── schedule.py     # task scheduler operators (which pool task the run faces next)
│   └── evolver.py      # the thin generational loop (steppable EvolverState; assess_many batching seam)
├── substrate.py        # decode a genome into a torch GraphNet / RecurrentGraphNet / RefineGraphNet
├── substrate_batched.py# BatchedGraphNet: the whole population as one padded tensor program
├── temporal.py         # TemporalEncoder + adapter: rebuild the TIME axis for the stepped substrate
├── evaluation.py       # score a substrate on a Task via the Icarus encoder/loss
├── structured.py       # support-only variable-grid canvas, learned shape rule, exact/baseline metrics
├── decompose.py        # DECOMPOSE registry: split a Task into valid subtasks with port wiring specs
├── library.py          # the persistent search space: module/composition entries, signatures, grafting, GC
├── orchestrator.py     # the escalation-ladder policy: lookup -> refine -> evolve -> decompose -> admit
├── strategy.py         # EVOLVE_STRATEGY registry: routed / grammar / direct / composition
├── grammar.py          # independently-supported motif grammar, program variation, compilation
├── routing.py          # the overmind: sparse MoE over frozen library entries, distillation, persistence
├── motifs.py           # motif census: canonical substructure mining across the library
├── checkpoint.py       # serialize/restore orchestrated runs for --resume
├── results.py          # per-run local artifacts (stats, speciation chart)
├── rendering.py        # recursive dark network renders + the overmind portrait
├── tools/
│   ├── rung_doctor.py      # uv run rung_doctor: probe rung loadability/shapes without a run
│   ├── net_gallery.py      # uv run render: re-render the library (+ --overmind portrait)
│   ├── library_gc.py       # uv run library_gc: mark-and-sweep unreferenced tombstones
│   ├── motif_census.py     # uv run motif_census: the motif report + atlas
│   ├── bench_throughput.py # uv run benchmark: measured throughput truths
│   └── runtime_inventory.py # deterministic canonical-surface manifest/check
├── trials/
│   └── orchestrated_trial.py # OrchestratedTrial(Proctor): the run loop, observability, resume
├── utils/
│   ├── config.py       # canonical all-features TOML -> runtime dict
│   ├── pipelines.py    # ClearML task + machine->queue + trial orchestration
│   ├── proctor.py      # base trial: logging, device, artifacts
│   └── logging.py      # Rich logger / console
└── main.py             # Config -> Pipeline -> OrchestratedTrial -> run_task
```

## Development

```bash
uv run ruff check . && uv run ruff format --check .   # lint + format (line length 180)
uv run ty check                                       # type check (Astral 'ty', not mypy)
uv run pytest tests/ -v                               # tests (offline; synthetic fixtures, no network)
```
