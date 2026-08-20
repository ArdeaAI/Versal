# Quick persistence ablation

This is the only last-minute ablation proposed for the paper. It tests the central persistence claim without changing the canary’s search strategies, populations, training, validation, or blind-query semantics.

## Design

Five mechanism-positive rungs are scheduled with two unique tasks each: 6, 9, 10, 12, and 15. Fifteen attempts create three cycles:

1. attempts 1-5: first task from each rung, allowing cold discoveries;
2. attempts 6-10: a distinct task from each rung, probing within-rung transfer;
3. attempts 11-15: exact revisits of cycle 1, probing retention and revisit cost.

The matched control sets `fresh_per_task = true`, disables module absorption, and therefore removes cross-task library, router, and shared-module state. Tasks, budgets, validation, and reporting remain matched, but candidate trajectories are not paired because the control resets its per-task random state.

## Commands

Run serially on the single MPS device and do not overlap these with another training job:

```bash
uv run ablation_suite \
  --manifest configs/ablations/quick_reuse_manifest.toml \
  --output results/ablations/quick_reuse_20260817 \
  --compute mps --dry-run

uv run ablation_suite \
  --manifest configs/ablations/quick_reuse_manifest.toml \
  --output results/ablations/quick_reuse_20260817 \
  --compute mps --arms full_memory --run-index 0

uv run ablation_suite \
  --manifest configs/ablations/quick_reuse_manifest.toml \
  --output results/ablations/quick_reuse_20260817 \
  --compute mps --arms no_cross_task_memory --run-index 0
```

The filtered `--run-index 0` selects seed 0 for each named arm while preserving its global run identity.

If seed 0 passes the activation gate, finish the remaining resumable suite serially:

```bash
uv run ablation_suite \
  --manifest configs/ablations/quick_reuse_manifest.toml \
  --output results/ablations/quick_reuse_20260817 \
  --compute mps --max-parallel 1
```

## Seed-0 activation gate

Continue only if:

- both arms finish with 15 task rows, 10 unique pool references, and 5 exact revisits;
- canonical task manifests match;
- full memory admits at least 3 durable entries during cycle 1;
- full memory produces at least 2 `library_hit` or `refined` outcomes during cycle 3;
- held-out coverage is at least 80% in both arms; and
- each arm finishes within 15 minutes with at most one deadline hit.

Otherwise stop and report that the quick probe did not activate or cleanly measure persistence.

### Gate outcome on 2026-08-17

The full-memory seed-0 arm was stopped after attempt 6. Its cold cycle admitted four durable entries, so the memory lever activated, but attempts 3 and 6 both reached the task deadline. Because the arm had already exceeded the predeclared allowance of one deadline hit, completing the matched arm could not make the seed-0 pair pass the gate. No ablation result was added to the manuscript; the interrupted diagnostic artifacts are not part of this public bundle.

## Paper claim gate

Describe exploratory revisit-cost amortization only if full memory is faster in at least two of three seeds, its paired median reduction is at least 20%, effective generations are lower, and mean held-out quality is no worse by more than 0.02. Claim transfer to distinct within-rung tasks only if the cycle-2 direction is consistent across seeds without quality loss. Mixed results mean the probe found no reliable advantage; a slower full-memory arm is a negative result.
