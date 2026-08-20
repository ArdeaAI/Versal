# Versal (Versatile Evolution of Reusable Structure for Adaptive Learning)

Versal is a Python 3.12 research system for evolutionary neural architecture search. It searches
for network and composition structure, trains candidate weights with gradient descent, and keeps
accepted structures in a persistent library so later tasks can reuse or extend them.

![alt text](paper/images/evolved_net.png "Evolved Neural Network")

**Toward task-general intelligence with persistent, compounding neuroevolution.**

Human intelligence is not a single faculty, and intelligence itself is neither uniquely human nor
confined to one substrate. Trying to specify every useful mechanism by hand tends to produce narrow
or jagged systems. Versal starts from a different premise: search for structure, preserve what works,
and let later searches build on the accumulated result.

The long-term aim is a system that is not taught a vocabulary of solutions, but can develop the
machinery it needs to read, hear, see, reason, and adapt. The present project is an early, measurable
step toward that aim, not a claim of general intelligence or consciousness. It is a research system
for testing whether one evolutionary process can cross task modalities while retaining and reusing its discoveries.

The broader research program, which I have three systems working together planned for, ultimately asks whether this road can support genuine agency,
objectness, selfhood, and conscious experience. Versal currently supplies no operational measure of
those properties; benchmark capability and subjective experience are separate claims.
The supported runtime works across the 18-rung
[Icarus dataset](https://huggingface.co/datasets/Ardea/Icarus-dataset). The ladder spans Boolean,
temporal, image, scientific, audio, and structured-grid tasks. Versal handles them through a shared
data contract—typed tensors, semantic axes, masks, and structural widths—rather than branching on
benchmark names or rung numbers.

This is an experimental research codebase, not a claim of general intelligence, agency, or
consciousness.

## How a run works

The complete local method is defined by [`configs/canary.toml`](configs/canary.toml). Plain
`uv run app` loads the smaller [`configs/smoke.toml`](configs/smoke.toml) overlay, which preserves
the method while reducing its populations, training, data, and recursion.

For each scheduled task, Versal:

1. Materializes one revision-pinned Icarus task and constructs a support-only search view.
2. Tests structurally compatible library entries. A successful hit may receive a bounded,
   non-regressing refinement attempt.
3. For a miss, optionally decomposes an oversized task before flat search when no enabled native
   representation can handle it safely.
4. Runs the configured shared-budget strategy ladder:
   `routed → grammar → field → direct → composition`.
5. Validates provisional support winners by refitting them on reduced support folds. The real query
   split remains inaccessible.
6. If the ladder does not produce an accepted parent, optionally decomposes the task and recursively
   solves the resulting parts.
7. Evaluates support-selected report candidates on held-out query data only after their support
   decision. Those values cannot affect search, validation, or library admission.
8. Admits a reusable winner, or retains a sufficiently useful loser as a stepping stone, then
   advances lifecycle state and writes a resumable task boundary.

The five strategies have different jobs:

- **Routed** selects and combines frozen library experts, then tries to distill a useful route into
  a reusable composition.
- **Grammar** turns structures rediscovered in independent lineages into new candidate graph
  programs.
- **Field** evolves resolution-independent programs over compatible spatial fields.
- **Direct** evolves a task-shaped network from genomes and optional warm starts.
- **Composition** evolves graphs of reusable modules connected by trainable or fixed mappings.

All five share one generation budget. Unused generations carry forward, and the first executable
candidate to clear the support and cross-validation gates stops the ladder.

## Quick start

Install runtime and development dependencies, then run the default smoke profile:

```bash
uv sync --group dev
uv run app
```

Run the complete local profile explicitly:

```bash
uv run app --config configs/canary.toml
```

The default canary selects two tasks from every rung and schedules 36 attempts. Task references are
selected across dataset shards and pinned to an immutable dataset revision. Only the scheduled task
is decoded into memory.

The learned library and run records are durable local state. Give a run a separate relative library
path when it must start cold:

```bash
uv run app --config configs/canary.toml --library-dir library/cold-canary
uv run app --resume results/<timestamp>_orchestrated
```

During an interactive run, press **Escape** to request a cooperative stop. Versal finishes the
current safe optimizer or generation boundary, restores the terminal, records status `stopped`, and
writes the checkpoint needed by `--resume`. Ctrl-C remains the immediate interruption path.

## Run profiles

The checked-in profiles inherit the canary method and override explicit scale or hardware choices.

| Profile | Purpose | Scheduled scale |
|---|---|---:|
| `smoke.toml` | Fast end-to-end health check | 1 task/rung, 18 attempts |
| `canary.toml` | Complete local method | 2 tasks/rung, 36 attempts |
| `brute.toml` | Repeated deep search on one editable rung | Set in the profile |
| `preflight.toml` | Workstation confidence run | 10 tasks/rung, 180 attempts |
| `full.toml` | Long adaptive local campaign arm | 10 tasks/rung, 180 attempts/seed |
| `full_cluster.toml` | Multi-seed adaptive cluster campaign | 20 tasks/rung, 360 attempts/seed |
| `canary-lattice.toml` | CUDA parity overlay for the canary | Inherits the canary schedule |
| `full_cluster-lattice.toml` | Local-CUDA version of the cluster arm | 20 tasks/rung, 360 attempts/seed |

The brute profile is intentionally easy to retarget through `schedule.rungs`. Repeated attempts can
reuse exact hits, refine stored structures, seed searches from stepping stones, and build deeper
compositions instead of restarting from an empty search state.

## Persistent state and reporting

Every new run creates `results/<timestamp>_orchestrated/`. Its durable boundary includes:

- the source and fully merged effective configuration;
- the pinned task-pool manifest and dataset provenance;
- the rolling evolutionary, scheduler, topology, and attempt checkpoint;
- `run_summary.json`, with one record per attempted root task;
- `rung_summary.csv`, `run_report.json`, and `run_report.md` derived from that summary;
- task-level checkpoints and renders when a new library structure is admitted.

The library stores modules, compositions, routing state, grammar state, lifecycle metadata, and
network renders. `library/images/overmind.png` preserves live and retired routing history, while
`library/images/overmind_pruned.png` shows only the current live set.

Support and held-out values are separate reporting rails. A missing held-out value remains missing
with an explicit reason; it is never silently converted to zero.

## Useful commands

Run and compare experiments:

```bash
uv run app [--config <profile>]
uv run run_matrix --seeds 0,1,2 --cold
uv run ablation_suite --dry-run
```

Inspect a run or the learned library:

```bash
uv run run_report results/<run>
uv run rung_doctor --rungs 1-18
uv run render --overmind
uv run motif_census --render
uv run motif_census --run results/<run> --discover
uv run benchmark
uv run cppn_spike --offline
```

Maintain and verify persistent state:

```bash
uv run library_gc --dry-run
uv run router_migrate --library <v1-library> --output <new-library>
uv run experiment_archive list
uv run runtime_inventory --check
```

`library_gc --dry-run` previews unreachable retired entries without deleting them. External archive
restoration is hash-verified and staged before installation; it refuses a nonempty destination.

## ClearML and hardware selection

ClearML is optional. Set `[run] clearml = true`, pass `--clearml`, or use a hardware overlay that
enables it. Offline runs remain fully functional. Versal reports Python logs and deliberate
artifacts, but leaves Rich's transient stdout/stderr display local unless
`clearml_capture_streams = true`.

Machine labels select local CPU, MPS, CUDA, or remote queue behavior without changing the method.
For example:

```bash
uv run app --config configs/smoke.toml --machine LocalLatticeCUDA --clearml
uv run app --config configs/canary-lattice.toml
uv run app --config configs/preflight.toml --machine LocalLatticeCPU
```

## Project map

```text
versal/evolution/       genomes, operators, populations, schedules, and compositions
versal/dataset/         vendored Icarus contract plus streaming task materialization
versal/trials/          supported orchestrated trial
versal/tools/           reports, diagnostics, campaigns, rendering, and maintenance
versal/utils/           configuration, devices, display, shutdown, resources, and ClearML
configs/                complete method plus scale, campaign, and hardware overlays
tests/                  offline regression and integration coverage
library/                persistent learned state (gitignored)
results/                resumable run state and reports (gitignored)
runtime_inventory.json  generated registry, CLI, config, and persistent-path contract
```

Do not edit `versal/dataset/icarus.py`; it is vendored/generated. Add evolutionary behavior through
the registries and select it in configuration instead of hard-coding it into the control loop.

## Development

```bash
uv run ruff check .
uv run ruff format .
uv run ty check
uv run pytest tests/ -v
uv run runtime_inventory --check
```

Tests are offline and fixture-driven. Run a focused case with, for example:

```bash
uv run pytest tests/test_substrate.py::test_decode_forward_shape -v
```

When reporting a new experiment, preserve the exact configuration and task manifest, distinguish
missing measurements from valid zeroes, and keep held-out outcomes separate from support fitting.
