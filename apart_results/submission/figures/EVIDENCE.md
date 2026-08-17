# Apart evidence figures

Generate both publication figures from the repository root:

```bash
uv run python ai/for_apart/make_evidence_figures.py
```

The script verifies the SHA-256 digest of every source before reading it and asserts the headline
counts used in annotations. PDF timestamps are removed and PNG metadata is fixed.

## `clean_canary.pdf` / `.png`

Two tasks per Icarus rung from the completed cold seed-0 canary. The left panel plots the mean best
support score against the task-appropriate held-out mean. For ARC, task-appropriate held-out means
exact task success (0/2); the separate gray diamond gives literal cell accuracy (0.6695). The
cross-rung task-appropriate held-out mean is 0.6099. The right panel plots total task-seconds on a
log scale. Orange triangles mark rungs containing deadline-limited attempts and their labels show
the count out of two. All 36 held-out queries were evaluated; 15 attempts reached their task budget.

Source: `results/20260817_031349_orchestrated/run_summary.json`, SHA-256
`deeafeb90e790fbc8f72fc1bfadc10d07eebf95e5f2cb41d9647b5d3dd155e43`.

## `persistence_xor.pdf` / `.png`

The left panel is the cumulative count of library hits plus refinements in the historical cold
400-encounter run. It ends at 234/400 overall and 168/333 after excluding XOR; annotations mark the
first observed level-2 through level-5 admissions. These are observational counts from an earlier
software state, not a matched speedup estimate.

The right panel follows admitted solutions in one 200-encounter XOR run. Its eight admissions have
recorded complexity `18 -> 16 -> 10 -> 12 -> 10 -> 7 -> 6 -> 5`; the run comprises one fresh
evolution, seven refinements, and 192 library hits with support and held-out score 1.0 throughout.
The final admitted topology has five nodes and four enabled edges. This is repeated refinement of
one fixed task, not 200 independent solves or a multi-seed convergence result.

Sources:

- `ai/archive/20260706_flagship/results/run_summary.json`, SHA-256
  `04838c39f752fa423cac70f796faf349511eed3682c45bdf71349c0bb8198288`.
- `ai/archive/20260816_02_hackathon/results/20260816_185653_orchestrated/run_summary.json`, SHA-256
  `3f4071a5e649b1488781bd61643f3f5f03e320ab718b1d7a13aaed0c4564b0e7`.
