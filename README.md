# ArdEVO

Playground for evolutionary algo ML testing

The purpose of this is to test different ENAS methods for topology itself.

There is a custom dataset I made that goes through MANY rungs across modalities. The idea is to single in on a
search algorithm that seems to be able to generalize over the different rungs of the dataset, growing the topology
in minimum complexity needed as we go up each difficulty rung until we find at least an ideal algorithm, potentially
modified by me to something novel, that can be used across every rung and produce a significant score.

You can see an example of how to get tasks from the dataset in `ardevo/dataset/loader.py` and the dataset is from: `https://huggingface.co/datasets/Ardea/Icarus-dataset`

I want to explore what we can do with `https://github.com/RobertTLange/evosax`, though I think we might have to come up with how to use that for topologies themselves instead of weights/hyperparameters. I also want to look into the different things mentioned in `https://github.com/rtu715/NAS-Bench-360`

To start out with, keep this small. For phase 1, get the modular structure set up and a topology-evolution algorithm working on the first rung, which is just XOR.

We want to use ClearML for this as well as much as we can make use of it. I already have my config set up for that.

## Growing topologies across the Icarus rungs

The search grows a network *topology* from nothing (inputs + bias wired straight to a linear output) and lets
structural mutations add nodes/edges. A per-generation `train` step tunes each candidate's weights by gradient before
scoring, so **evolution searches structure and the gradient owns the weights** (pure weight-evolution stalls even on
XOR; and random weight mutation *fights* the gradient, so it is left out when training is on). We do not use
evosax/JAX in the loop; it stays installed for a future `cmaes` train operator on the non-differentiable higher rungs.

```bash
uv sync --group dev
uv run app                                       # default config.toml
uv run app --config configs/rung1_xor.toml       # rung 1: XOR
uv run app --config configs/rung2_parity.toml    # rung 2: parity (function-fit)
uv run app --config configs/rung3_two_spirals.toml  # rung 3: two-spirals (generalization; slow)
uv run app --config configs/continuous_ladder.toml  # ALL rungs interleaved into one growing topology
uv run app --config configs/continuous_ladder.toml --resume results/<ts>_continuous  # pick up where it left off
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

## Lego-block evolution

Every stage of the generational loop is an independent, registered operator selected and tuned from `config.toml`.
The loop runs: **select → crossover → mutate → train → evaluate → fitness → replace**, with speciation shaping how
offspring are allocated. To experiment, change a `kind`, reorder `[evolution.mutation].operators`, retune a weight, or
register one new function in the matching registry; the loop itself never changes.

| Stage | Config section | Registered options |
|---|---|---|
| init | `[evolution.init]` | `minimal` |
| selection | `[evolution.selection]` | `tournament`, `truncation` |
| crossover | `[evolution.crossover]` | `none`, `neat` |
| mutation | `[evolution.mutation]` | `add_rich_node` (width), `add_deep_node` (depth), `add_local_node` / `add_local_connection` / `add_shared_motif` (coordinate-aware locality), `add_connection`, `toggle_connection` (prune), `add_node`, `mutate_activation`, `perturb_weights` |
| train | `[evolution.train]` | `none`, `gradient` (params `steps`, `lr`, `writeback`, `weight_decay`; future: `cmaes` via evosax) |
| speciation | `[evolution.speciation]` | `none`, `neat` (compatibility threshold auto-targets a species count) |
| schedule | `[schedule]` | `random`, `round_robin`, `interleave_rungs` (continuous run only; picks the next task) |
| fitness | `[fitness]` | `support_accuracy`, `query_accuracy`, `negative_support_loss`, `negative_query_loss`, `complexity_penalty` |

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
│   ├── genome.py       # NEAT-style Genome (node/connection genes) + DAG helpers
│   ├── registry.py     # Registry + build_evolver factory
│   ├── init.py         # population-seeding operators
│   ├── selection.py    # parent-selection operators
│   ├── crossover.py    # recombination operators
│   ├── mutation.py     # structural + weight mutators
│   ├── train.py        # weight-optimization operators (gradient / none)
│   ├── fitness.py      # fitness components + weighted aggregator
│   ├── multitask.py    # grow-the-I/O substrate: descriptor-keyed banks, output heads, head-slicing
│   ├── schedule.py     # task scheduler operators for the continuous run
│   └── evolver.py      # the thin generational loop (steppable EvolverState for the continuous run)
├── substrate.py        # decode a genome into a torch GraphNet (SubstrateModule base)
├── evaluation.py       # score a substrate on a Task via the Icarus encoder/loss
├── checkpoint.py       # serialize/restore the continuous run for --resume
├── results.py          # per-run local artifacts (stats, model, net, speciation)
├── trials/
│   ├── xor_trial.py        # EvolutionTrial(Proctor): runs + logs one rung
│   └── continuous_trial.py # ContinuousTrial(Proctor): one topology across interleaved rungs
├── utils/
│   ├── config.py       # config.toml -> runtime dict
│   ├── pipelines.py    # ClearML task + machine->queue + trial orchestration
│   ├── proctor.py      # base trial: logging, device, artifacts
│   └── logging.py      # Rich logger / console
└── main.py             # Config -> Pipeline -> add_trial -> run_task
```

Planned for later phases (documented here, not built yet): `models/` (checkpoints, evolved/sgd substrates),
`analysis/` (cross-modal metrics, visualization), `baselines/` (gradient-only trainer), `scripts/` (evaluate,
figures), `paper/`, additional per-rung trials, an evosax-backed `cmaes` train operator, speciation / fitness
sharing, recurrent topologies, and richer crossover.

## Development

```bash
uv run ruff check . && uv run ruff format --check .   # lint + format (line length 180)
uv run ty check                                       # type check (Astral 'ty', not mypy)
uv run pytest tests/ -v                               # tests (offline; synthetic XOR fixture)
```
