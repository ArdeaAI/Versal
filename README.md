# ArdEVO

Playground for evolutionary algo ML testing

The purpose of this is to test different ENAS methods for topology itself.

There is a custom dataset I made that goes through MANY rungs across modalities. The idea is to single in on a
search algorithm that seems to be able to generalize over the different rungs of the dataset, growing the topology
in minimum complexity needed as we go up each difficulty rung until we find at least an ideal algorithm, potentially
modified by me to something novel, that can be used across every rung and produce a significant score.

You can see an example of how to get tasks from the dataset in `ardevo/dataset/loader.py` and the dataset is from: `https://huggingface.co/datasets/Ardea/Icarus-dataset`. NOTE THAT the pole and double-pole rungs, which are noramlly continous RL-type problems has been transformed
into time-series differentiable tasks. You also need to pay SPECIAL ATTENTION to the types in the metadata of that dataset that allow it to describe itself.

We want to use ClearML for this as well as much as we can make use of it. I already have my config set up for that.

## Growing topologies across the Icarus rungs

The search grows a network *topology* from nothing and lets
structural mutations add nodes/edges. A per-generation `train` step tunes each candidate's weights by gradient before
scoring, so **evolution searches structure and the gradient owns the weights** (pure weight-evolution stalls even on
XOR; and random weight mutation *fights* the gradient, so it is left out when training is on).

```bash
uv sync --group dev
uv run app                                       # default config.toml
uv run app --config configs/rung1_xor.toml       # rung 1: XOR
uv run app --config configs/rung2_parity.toml    # rung 2: parity (function-fit)
uv run app --config configs/rung3_two_spirals.toml  # rung 3: two-spirals (generalization; slow)
uv run app --config configs/continuous_ladder.toml  # ALL rungs interleaved into one growing topology
uv run app --config configs/continuous_ladder.toml --resume results/<ts>_continuous  # pick up where it left off
uv run app --config configs/continuous_ladder_batched.toml  # same ladder, population-batched training
uv run app --config configs/orchestrated_ladder.toml        # the recursive hierarchical orchestrated evolver
uv run app --config configs/orchestrated_ladder.toml --resume results/<ts>_orchestrated
```

What each rung shows (see `configs/` for tuned, runnable settings):

| Rung | Task | Data | What "solve" means | Result |
|---|---|---|---|---|
| 1 | `xor` | 4/4 (full table) | 100% on all 4 (needs >=1 hidden node) | **query 1.0**, grows 1 hidden node |
| 2 | `parity.n4` | 13/3 of 16 | fit the function (support 1.0) | **support 1.0**, grows hidden; query is noise* |
| 3 | `two_spirals` | 194/192 | held-out generalization (query) | **query 1.0**, grows 0 -> ~23 hidden nodes (~gen 100) |

*Parity is the canonical anti-generalization function: fitting the 13 support points tells you nothing about the 3
held-out points (even a perfect hand-built net scores ~0.0-0.33 on that query), so query is judged only where the
split is meaningful. The achievement on parity is growing a minimal topology that *fits the function*.

Set `[run] clearml = true` to track in ClearML; it degrades gracefully offline. Machine env maps to a queue:
`MonadMetal`/`MonadCPU`/`local` run locally, `LatticeCPU`/`LatticeCUDA` enqueue remotely.

## One topology across the whole ladder (continuous run)

`configs/continuous_ladder.toml` drives a single continuous run that randomly interleaves tasks across
many rungs and keeps **one** population alive across the switches, so the topology grows to be good at
all of them. Every Icarus rung is a differentiable supervised `(input -> output)` task (the
`INTERACTIVE` flag on rungs 4-5 is provenance, not a separate scoring regime), so the same gradient
inner loop scores all 18 rungs; the default config starts with the fast rungs 1-5 and `[schedule] rungs`
is the single knob to widen coverage (set `rungs = "all"` for the full ladder).

The interface is **grown, never pre-allocated** (the minimum-complexity thesis applied to I/O):

- Inputs live in **descriptor-keyed banks** (one per `value_type` + `axes` signature) that widen on
  demand; a node is never overloaded with semantically different values (a bit and a coordinate never
  share a node). Each input node is stamped with its raw axis-index `coordinate`.
- Each task grows its **own disjoint output head** on first encounter, scored via a thin head-slicing
  wrapper so the shared `GraphNet` decode and the evaluate/train path are unchanged.
- The shared hidden body persists across switches; the geometry-biased mutators (`add_local_node`,
  `add_local_connection`, `add_shared_motif`) read the coordinates to grow local receptive fields.
- A **complexity penalty is mandatory** so the shared body cannot grow unbounded (the trial refuses to
  run without one).

State is saved every `checkpoint_every` generations into its own `results/<ts>_continuous/gen_<NNNNNN>/`
(model, stats, net, speciation, plus a resumable `checkpoint.json`); `--resume <run_dir>` continues from
the latest checkpoint bit-for-bit. `stats.json` records the champion's accuracy on every rung seen so far
(the "good at all rungs" signal).

## The orchestrated ladder: recursive hierarchical evolution

The continuous run exposed the monolith's failure mode: ONE shared body retrained per task forgets
everything it learned before (gen-200 stats: 1.0 on the active task, chance on every prior rung).
The orchestrated run (`configs/orchestrated_ladder.toml`) replaces the shared mutable champion with
an escalation ladder per task:

1. **LOOKUP**: a persistent, file-based **library** (`library/`) is queried by structural I/O
   signature (value_type + axes + widths, NEVER rung or task name). A stored solution that still
   clears the accept threshold is reused at zero generations: a solved task STAYS solved.
2. **EVOLVE**: a per-task population of **compositions** (small DAGs whose MODULE nodes reference
   live module species or library entries, wired by trainable linear GLUE) co-evolves with ONE
   shared mini-model population. Composition fitness flows DOWN as attribution to the modules and
   entries it referenced; only the champion composition writes module weights back.
3. **DECOMPOSE**: on a stall, registered decompose operators (`output_slices`, `input_subsets`,
   `time_windows`) split the task into fully valid subtasks and the orchestrator RECURSES on each
   (depth-capped). Accepted parts become frozen library entries; the parent re-evolves seeded with
   a port-wired composition over them.
4. **ADMIT**: winners persist with provenance and a weight-robustness score (mean minus std of
   accuracy under shared-weight samples, the signal that predicts a module composes well). Live
   module refs are snapshotted as frozen entries, so nothing in the library ever dangles.

Recursion comes from the library, not from new types: an entry can itself be a composition, so
level 1 is mini-models, level 2 is compositions of them, level 3+ is compositions of compositions.
The library survives across runs (delete `library/` to start the search space cold). When a task
admits novel library entries, its checkpoint (`task_<NNNN>/`), net image, speciation chart, attempts
ledger, library growth, and module-pool stats land in stats.json and ClearML (`Orchestrator/*`,
`Modules/*`, `Fitness/comp_*`, `Robustness/*`).

Net images are recursive and dark (`ardevo/rendering.py`): the host network keeps its compact
layout with green hexagon footprints where nested networks plug in, and each referenced library
entry draws fully inside a translucent callout box packed across the top of the frame, green-lined
to its footprint, nested to depth 4 (anything unresolvable degrades to a labeled opaque box).
Re-render everything after the fact:

```bash
uv run render             # one library/images/<key>.png per entry (--gallery adds a contact sheet)
```

New substrate primitives make layered complexity EXPRESSIBLE while staying grown-from-minimal:
`aggregation = "product"` nodes (multiplicative gating, second-order interactions) and true
RECURRENCE (time-delayed edges + a stepped `RecurrentGraphNet` over TIME-axis tasks via
`ardevo/temporal.py`, trained by plain BPTT through the existing gradient operator). An exact
running-parity machine is a 2-hidden-node genome: one product node times the previous state, one
accumulator with a self-loop.

Phase 4 turned the run data into upgrades. The EVOLVE step is now a strategy ladder
(`[orchestrator] evolve = ["composition", "direct"]`): `direct` runs the proven flat phase-1 recipe
on the task's REAL I/O (the fix for the two-spirals class of failures that burned 68% of all
generations), routes TIME-axis tasks through the stepped recurrent substrate (recurrent genes
finally execute in the orchestrated path), and admits TASK-SHAPED modules; `composition` verifies
its champion against CURRENT module state before anything is admitted. Found networks now flow back
into the search three ways: grafted into the module pool at every lookup miss (`absorb_top_k`),
inlined as unfrozen evolvable structure (`add_library_module`), or embedded as a single FROZEN
MACRO NODE (`add_macro_node`): a whole library network behind one gene, the way an LSTM cell is a
network inside a node, with crossover/speciation treating the placement atomically. The library
gained a quality gate (`[library] admission = "default"`: metric/robustness floors plus a
per-signature cap that tombstones, never deletes), readmission-refreshing dedupe, and width-tolerant
queries. Decompose failures now say WHERE they died (`failure_stage`, per-stage counters), skipped
rungs are loud (stats + console + `ardevo/tools/rung_doctor.py`), and compositions can carry
factored rank-r glue (`glue_rank_threshold`) so wide rungs do not explode entry sizes. Honest
throughput note from `uv run benchmark` (`ardevo/tools/bench_throughput.py`): thread-parallel assessment and stacked sample-eval
measured SLOWER at current kernel sizes and ship default-off; the partitioned `gradient_batched`
trainer (1.5x CPU / 2.1x MPS at pop 48) is the lever that pays, and the direct strategy can use it.

On the prior art: CoDeepNEAT's two-population idea survives here as composition-genomes-referencing-
species and fitness attribution; WANN's weight-agnostic insight survives as the robustness metric
and the `weight_samples`/`hybrid` evaluate stage. Neither is implemented literally. evox was
evaluated and skipped (fixed-vector optimization only); tensorneat's padded-tensor trick was ported
natively as `gradient_batched` (one tensor program trains the whole generation, MPS-friendly,
semantics-preserving with sequential fallback).

## Lego-block evolution

Every stage of the generational loop is an independent, registered operator selected and tuned from `config.toml`.
The loop runs: **select → crossover → mutate → train → evaluate → fitness → replace**, with speciation shaping how
offspring are allocated. To experiment, change a `kind`, reorder `[evolution.mutation].operators`, retune a weight, or
register one new function in the matching registry; the loop itself never changes.

| Stage | Config section | Registered options |
|---|---|---|
| loop | `[evolution] loop` | `flat` (the phase-1 Evolver), `hierarchical` (compositions + shared module pool) |
| init | `[evolution.init]` | `minimal` |
| selection | `[evolution.selection]` | `tournament`, `truncation` |
| crossover | `[evolution.crossover]` | `none`, `neat` |
| mutation | `[evolution.mutation]` | `add_rich_node` (width), `add_deep_node` (depth), `add_local_node` / `add_local_connection` / `add_shared_motif` (coordinate-aware locality), `add_connection`, `toggle_connection` (prune), `add_node`, `mutate_activation`, `mutate_aggregation` (sum/product flip), `add_recurrent_connection` (time-delayed memory), `add_library_module` (graft a stored mini-model), `perturb_weights` |
| train | `[evolution.train]` | `none`, `gradient` (params `steps`, `lr`, `writeback`, `weight_decay`), `gradient_batched` (whole generation in one tensor program; `device`, `max_padded_nodes`) |
| evaluate | `[evolution.evaluate]` | `standard`, `weight_samples` (weight-agnostic scoring), `hybrid` (trained metrics + robustness) |
| speciation | `[evolution.speciation]` | `none`, `neat` (compatibility threshold auto-targets a species count) |
| schedule | `[schedule]` | `random`, `round_robin`, `interleave_rungs` (continuous run only; picks the next task) |
| comp mutation | `[evolution.composition.mutation]` | `add_module_node`, `switch_ref`, `add_comp_edge`, `toggle_comp_edge`, `perturb_glue` (crossover: `none`, `comp_neat`) |
| decompose | `[orchestrator] decompose` | `output_slices`, `input_subsets`, `time_windows` |
| fitness | `[fitness]` | `support_accuracy`, `query_accuracy`, `negative_support_loss`, `negative_query_loss`, `mean_sample_accuracy`, `max_sample_accuracy`, `weight_robustness`, `negative_mean_sample_loss`, `complexity_penalty`, `hidden_penalty` |

Notes from getting this to actually grow useful topologies: `add_rich_node` only widens a layer, so depth-needing
tasks (two-spirals) require `add_deep_node`; `toggle_connection` is the only operator that prunes, so a complexity
penalty needs it to simplify; and `neat` speciation auto-adjusts its threshold (a fixed one fractures the population
into singletons and starves reproduction). The torch substrate is vectorized (level-wise matmuls) for speed.

## Project structure

Built in phase 1:

```
ardevo/
├── dataset/
│   ├── icarus.py       # vendored Icarus runtime (generated; edit upstream, not here)
│   └── loader.py       # load one rung's Task from the Hub
├── evolution/
│   ├── genome.py       # NEAT-style Genome (node/connection genes incl. aggregation + recurrence) + DAG helpers
│   ├── registry.py     # Registry + build_evolver / build_loop factories
│   ├── init.py         # population-seeding operators
│   ├── selection.py    # parent-selection operators
│   ├── crossover.py    # recombination operators
│   ├── mutation.py     # structural + weight mutators (incl. aggregation flips, recurrent edges, library grafts)
│   ├── train.py        # weight-optimization operators (gradient / gradient_batched / none)
│   ├── evaluate.py     # metrics operators (standard / weight_samples / hybrid robustness scoring)
│   ├── fitness.py      # fitness components + weighted aggregator
│   ├── composition.py  # CompositionGenome: the recursive representation (modules + glue), assembly, operators
│   ├── loop.py         # LOOP registry: FlatLoop + HierarchicalLoop (attribution, champion writeback)
│   ├── multitask.py    # grow-the-I/O substrate: descriptor-keyed banks, output heads, head-slicing
│   ├── schedule.py     # task scheduler operators for the continuous run
│   └── evolver.py      # the thin generational loop (steppable EvolverState; assess_many batching seam)
├── substrate.py        # decode a genome into a torch GraphNet / RecurrentGraphNet (SubstrateModule base)
├── substrate_batched.py# BatchedGraphNet: the whole population as one padded tensor program
├── temporal.py         # TemporalEncoder + adapter: rebuild the TIME axis for the stepped substrate
├── evaluation.py       # score a substrate on a Task via the Icarus encoder/loss
├── decompose.py        # DECOMPOSE registry: split a Task into valid subtasks with port wiring specs
├── library.py          # the persistent search space: module/composition entries, signatures, grafting
├── orchestrator.py     # the escalation-ladder policy: lookup -> evolve -> decompose/recurse -> admit
├── checkpoint.py       # serialize/restore continuous + orchestrated runs for --resume
├── results.py          # per-run local artifacts (stats, model, speciation chart)
├── rendering.py        # recursive dark network renders: nested networks as callout boxes over the host
├── tools/
│   ├── rung_doctor.py      # uv run rung_doctor: probe rung loadability/shapes without a run
│   ├── net_gallery.py      # uv run render: re-render the library to library/images/<key>.png
│   └── bench_throughput.py # uv run benchmark: measured throughput truths
├── trials/
│   ├── xor_trial.py        # EvolutionTrial(Proctor): runs + logs one rung
│   ├── continuous_trial.py # ContinuousTrial(Proctor): one topology across interleaved rungs
│   └── orchestrated_trial.py # OrchestratedTrial(Proctor): the recursive orchestrated evolver
├── utils/
│   ├── config.py       # config.toml -> runtime dict
│   ├── pipelines.py    # ClearML task + machine->queue + trial orchestration
│   ├── proctor.py      # base trial: logging, device, artifacts
│   └── logging.py      # Rich logger / console
└── main.py             # Config -> Pipeline -> add_trial -> run_task
```

Planned for later phases (documented here, not built yet): `spatial_patches` decomposition (the ARC
grid shape) and the ARC-AGI-3 interactive harness trial (episodes -> TIME-axis Tasks through the
SAME orchestrator; the task layer already keeps signatures descriptor-only and `pool_from_tasks`
accepts tasks that never touched the Icarus dataset), stepped/temporal composition assembly,
iterative refinement on static tasks (`settle_steps`), composition-level speciation, a bandit
chooser over decompose operators, batched training for compositions, `analysis/` (cross-modal
metrics), `baselines/` (gradient-only trainer), and `paper/`.

## Development

```bash
uv run ruff check . && uv run ruff format --check .   # lint + format (line length 180)
uv run ty check                                       # type check (Astral 'ty', not mypy)
uv run pytest tests/ -v                               # tests (offline; synthetic XOR fixture)
```
