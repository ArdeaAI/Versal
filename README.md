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

## Phase 1: XOR (current)

Phase 1 evolves a network *topology* from nothing to solve the rung-1 XOR task. We deliberately do **not** use
evosax/JAX in the loop here: evosax is a fixed-vector optimizer, while growing a graph from minimal complexity is a
structural (NEAT-style) search that we own directly in PyTorch. evosax stays installed and is slated to return later
as a pluggable weight optimizer (the `train` stage) for the non-differentiable higher rungs (rungs 4-5 are interactive
policy rollouts).

A candidate starts as inputs + a bias wired straight to a linear output. Structural mutations add nodes and
connections; a per-generation `train` step tunes each candidate's weights by backprop before scoring. Because XOR is
not linearly separable, a winning run must *grow* at least one hidden node to break 75% and reach 100% query accuracy.

```bash
uv sync --group dev
uv run app                       # evolve on rung-1 XOR (offline by default)
uv run app --config config.toml  # explicit config path
```

Set `[run] clearml = true` in `config.toml` to track fitness / accuracy / complexity in ClearML; it degrades
gracefully offline. Machine env maps to a queue: `MonadMetal`/`MonadCPU`/`local` run locally, `LatticeCPU`/
`LatticeCUDA` enqueue remotely (push to GitHub first, since the agent clones the repo).

## Lego-block evolution

Every stage of the generational loop is an independent, registered operator selected and tuned from `config.toml`.
The loop runs: **select → crossover → mutate → train → evaluate → fitness → replace**. To experiment, change a
`kind`, reorder `[evolution.mutation].operators`, retune a weight, or register one new function in the matching
registry; the loop itself never changes.

| Stage | Config section | Registered options (phase 1) |
|---|---|---|
| init | `[evolution.init]` | `minimal` |
| selection | `[evolution.selection]` | `tournament`, `truncation` |
| crossover | `[evolution.crossover]` | `none`, `neat` |
| mutation | `[evolution.mutation]` | `perturb_weights`, `add_connection`, `add_node`, `mutate_activation`, `toggle_connection` |
| train | `[evolution.train]` | `none`, `gradient` (future: `cmaes` via evosax) |
| fitness | `[fitness]` | `query_accuracy`, `complexity_penalty`, `negative_query_loss` |

The `train` stage defaults to `gradient`: tuning a fresh topology's weights before scoring lets structural growth
pay off immediately, which stands in for the speciation that would otherwise protect new innovations (deferred).

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
│   └── evolver.py      # the thin generational loop
├── substrate.py        # decode a genome into a torch GraphNet
├── evaluation.py       # score a substrate on a Task via the Icarus encoder/loss
├── trials/
│   └── xor_trial.py    # EvolutionTrial(Proctor): runs + logs one rung
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
