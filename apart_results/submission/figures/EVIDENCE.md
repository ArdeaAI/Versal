# Apart evidence figures

Generate both publication figures from the repository root:

```bash
uv run python apart_results/submission/source/make_evidence_figures.py
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

Source: `apart_results/20260817_031349_orchestrated/run_summary.json`, SHA-256
`d7f25295ba8c40a5a7bbf704612ae8c8d1f00656331e17a1b341ad414b885c51`.

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

- `apart_results/evidence/historical/20260706_flagship/run_summary.json`, SHA-256
  `6a880b1ba6e1a94040ab577a18d2042fb04b3253c0f135f8f029eac81039e1f2`.
- `apart_results/evidence/historical/20260816_xor/run_summary.json`, SHA-256
  `17a358bde5f130884251b38208b080dffb15f2965918b4fba0254cab14cb8ce1`.
