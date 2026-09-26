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
2. Tests structurally compatible library entries, checking the protected winner for this exact support task first. A successful hit resumes its populations for bounded, non-regressing refinement.
3. For a miss, optionally decomposes an oversized task before flat search when no enabled native
   representation can handle it safely.
4. Interleaves eligible `routed`, `grammar`, `spatial`, `direct`, and `composition` processes using the configured budget shares. A turn advances one population generation, one router training slice, one distillation attempt, or a bounded grammar preparation slice. Preparation uses scheduling credit without reporting a generation.
5. Validates provisional support winners by refitting them on reduced support folds. A complete, consistent, unmasked Boolean input domain instead receives exhaustive support verification: withholding one truth-table row would ask a different learning question. The real query split remains inaccessible to search and admission.
6. If search does not produce an accepted parent, optionally decomposes the task and recursively
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
- **Spatial** grows sparse neuron circuits over raw typed tensors, with evolved connection rules, optional repeated placements, and independently evolved parameter sharing. It supports sequence, classification, and tensor mapping tasks without choosing an architecture from the rung number.
- **Direct** evolves a task-shaped network from genomes and optional warm starts.
- **Composition** evolves graphs of reusable modules connected by trainable or fixed mappings.

All five share one generation budget. The first accepted executable switches the search to a shared refinement allowance (`orchestrator.refine.budget_k`, normally 24). Exact support-task revisits resume those same populations, random streams, species, novelty history, candidate deduplication history, and router optimizer moments. Ineligible strategies spend no generation credits and can become eligible as the library grows. Router slices and distillation are separately charged; per-strategy evaluations, optimizer updates, and elapsed time are recorded because a generation is not an equal amount of compute across representations.

An accepted task winner is protected separately from the diversity archive. Replacement compares acceptance score, recursively expanded complexity, then weight robustness; perfect accuracy cannot be traded for a smaller graph. Compact native winners remain available as parents even when diversity selection prefers another individual. Exchanges are bounded and reference immutable executable snapshots. Grammar still requires independent lineages: copies exchanged during one task do not manufacture rediscovery evidence.

When refinement deduplication is enabled, interleaved search compares actual candidate parameters, including live module weights. A previously tried topology can be fitted again from new inherited weights; unchanged candidates are skipped. A lifetime ban on topology alone would block continued fitting and can prevent later pruning. The ladder comparator retains its original topology-only rule.

Set `orchestrator.search_policy = "ladder"` to retain the original sequential policy for comparison. That policy stops at the first accepted strategy and refines only the stored representation. The live profiles select `"interleaved"` through `canary.toml`.

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
writes the checkpoint needed by `--resume`. The first Ctrl-C requests the same cooperative save, including redirected runs. A second Ctrl-C forces exit with status 130. Assessment workers leave signal handling to the main process.

## Run profiles

The checked-in profiles inherit the canary method and override explicit scale or hardware choices.

| Profile | Purpose | Scheduled scale |
|---|---|---:|
| `smoke.toml` | Fast end-to-end health check | 1 task/rung, 18 attempts |
| `canary.toml` | Complete local method | 2 tasks/rung, 36 attempts |
| `brute.toml` | Repeated deep search on one editable rung | Set in the profile |
| `preflight.toml` | Workstation confidence run | 10 tasks/rung, 180 attempts |
| `full.toml` | Long adaptive local campaign arm | 10 tasks/rung, 180 attempts/seed |
| `fullish_light.toml` | Editable local campaign with adaptive resource limits | Set in the profile |
| `xor_repro.toml` | Offline cold-library XOR reproducibility check | 100 exact-task encounters/seed |
| `full_cluster.toml` | Multi-seed adaptive cluster campaign | 20 tasks/rung, 360 attempts/seed |
| `canary-lattice.toml` | CUDA parity overlay for the canary | Inherits the canary schedule |
| `full_cluster-lattice.toml` | Local-CUDA version of the cluster arm | 20 tasks/rung, 360 attempts/seed |

The brute profile is intentionally easy to retarget through `schedule.rungs`. Repeated attempts can
reuse exact hits, refine stored structures, seed searches from stepping stones, and build deeper
compositions instead of restarting from an empty search state.

## XOR reproducibility

The offline check constructs all four XOR inputs locally and uses a separate empty library per seed. All five strategies remain configured, including spatial graph recipes on the Boolean tensor contract. Grammar preparation shares the schedule and waits for independently supported productions. The profile retains the direct/composition populations and training allowances from `fullish_light.toml`, pins computation to one CPU thread per process, and batches small direct candidates for practical runtime.

```bash
uv run xor_repro --output results/xor-check --seeds 0,1,2,3,4,5,6,7,8,9
uv run xor_repro --output results/xor-ladder --policy ladder --seeds 0,1,2,3,4,5,6,7,8,9
```

For a quicker convergence check, add `--stop-at-minimum` to finish each seed when perfect support reaches expanded complexity 5 or less. The default continues through all 100 encounters to check continued stability. Query accuracy remains report-only in either mode.

Each seed must reach perfect support and query accuracy, preserve that accuracy, never increase the accepted perfect winner's expanded complexity, and finish at complexity **5 or less** within 100 encounters. This is an empirical reproducibility target under the repository's structural cost, not a proof of a unique mathematical minimum. Activation choices, including `sin`, remain evolved.

Each seed directory saves `trajectory.json`, `summary.json`, `predictions.json`, the final executable and dependency closure in `final_payloads.json`, the effective configuration, source hashes, and a task-boundary checkpoint. A failed target produces a nonzero exit status and retains its evidence. To test interruption at a completed boundary:

```bash
uv run xor_repro --output results/xor-resume --seeds 0 --encounters 50
uv run xor_repro --output results/xor-resume --seeds 0 --encounters 100 --resume
```

Compare this with a separate uninterrupted seed-0 run. Timings may differ; candidate identities, predictions, structural trajectories, and saved search state should agree. Held-out XOR covers the same finite four-input domain, so a perfect query score here is not evidence of generalization beyond that domain.

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

The terminal keeps the best support score fixed at the left of its live status line, with the current activity updating beside it. Completed result boxes stay in scrollback; add `--verbose` to retain individual stage messages too. Network PNGs use a dark, organic style with small nodes, fine connections, and compact legends. Overmind portraits retain usage percentages and stone/retired labels; the recorded JSON holds the detailed experiment data.

Interleaved populations live in compressed immutable snapshots under `library/search/`, indexed by the full support data contract, configuration, and seed. Inactive tasks stay on disk. Rolling checkpoints capture the snapshot index and RNG state; garbage collection protects the current incumbents and population dependencies. Resume requires the corresponding library as well as the run directory.

Support and held-out values are separate reporting rails. A missing held-out value remains missing
with an explicit reason; it is never silently converted to zero.
For accepted solutions, both headline scores describe the selected executable. A live router can score perfectly while its distilled composition fails; that failure reports the router score, distilled score, and acceptance threshold separately. Validation status, candidate identities, shared work, and expanded complexity remain present on unchanged library hits.

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

Spatial recipes are versioned module payloads. A minimal recipe connects raw input values and a bias to output ports; it supplies no convolution stencil, pooled statistics, attention block, or hand-authored hidden circuit. Registered mutations add/prune neurons and edges, change index maps, group connected neurons, repeat or split circuits, and tie or untie their parameters. Shape bindings are inferred from support descriptors and dimensions; query target shapes are used only for evaluation. Irregular shapes that cannot be represented by a consistent binding are declined explicitly.

Compositions and routes can execute stored spatial modules, and spatial recipes can embed frozen modules or compositions. Direct search can expand small recipes or use them as macros; large recipes remain compact. Grammar mines scalar circuit examples from their actual bindings, keeping the original entry lineage so repeated copies cannot supply independent evidence. Candidate comparisons count every executed neuron, connection, and referenced placement; recipe size and parameter count are separate diagnostics. Resource limits govern execution size, not eligibility by task name or rung.

Grammar induction keeps an indexed, resumable cursor and caches complete evidence per immutable entry, lineage, and mining parameters. Each preparation turn processes at most 128 work items with a 100 ms slice target, checking the stop/deadline between items. Entry loading, indexing, and publication are indivisible work items, so the slice target is not a hard real-time guarantee. New evidence is published only as a complete snapshot. Timing rows distinguish preparation, an explicit skip, and a strategy that was not reached before the budget ended.

The legacy `field` strategy remains registered for old configurations and `field_template` artifacts. New profiles select `spatial`; an existing run resumed from its saved effective configuration keeps its original strategy selection.
