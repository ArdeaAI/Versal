# Run report

## Provenance

This report was derived from `/Users/sinjhin/WS/A/subworkdir/VERSAL/results/20260817_031349_orchestrated` using report schema 1. It reads only the durable summary, pinned configs and task manifest, and optional library metadata.

- `run_summary.json`: `deeafeb90e790fbc8f72fc1bfadc10d07eebf95e5f2cb41d9647b5d3dd155e43` (102,272 bytes)
- `run_manifest.json`: `dd78eade854d5bbe935079d7f6a66ff6c0744f0e3643a59b8b73c2ce1a98389b` (781 bytes)
- `config.toml`: `26e698cc2708a92eadebff326fab0b8ff6eaa12252ec04e3204684510fb30f9d` (9,751 bytes)
- `config.effective.json`: `7abb6aae5ef02da457dc374d375b94c9a0cc4b73ab14c3c7cb30538d40f44ef6` (10,721 bytes)
- `task_pool.json`: `9fd250d93c05b1747ae78d540717bc30c9d5278da9dad103e7c879eef48ade72` (7,884 bytes)
- Dataset `Ardea/Icarus-dataset` at revision `412029ed1b86072a08f47102959d3ebdc9dee766`; selection `shard_round_robin_v1`.
- Code commit `043e1e253d2345515ae70ee15b55c012b7e2bfc7`; clean worktree.
- Starting library: 0 entries; content hash `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Executive summary

Status `done`; 36/36 tasks; 1,183 generations; 6702.8 recorded task-seconds. Held-out query accuracy covered 36 tasks (mean 0.6471, maximum 1.0000); support accuracy is reported separately (mean 0.8870, maximum 1.0000). The pool contains 34 unique reference(s); 2 completed attempt(s) are revisits.
First-occurrence-only quality covers 34/34 query evaluations (mean 0.6420, maximum 1.0000).

## Per-rung results

| Rung | Tasks | Query coverage | Held-out max | Held-out mean | Winning task | Support max | Support mean | Accepted | Failed | Admissions | Seconds |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 2/2 (100.0%) | 1.0000 | 1.0000 | xor | 1.0000 | 1.0000 | 0 | 2 | 2 | 46.6 |
| 2 | 2 | 2/2 (100.0%) | 1.0000 | 0.7308 | parity.n8.b3 | 1.0000 | 1.0000 | 0 | 2 | 3 | 62.7 |
| 3 | 2 | 2/2 (100.0%) | 0.4688 | 0.4688 | two_spirals | 0.5670 | 0.5644 | 0 | 2 | 2 | 66.9 |
| 4 | 2 | 2/2 (100.0%) | 1.0000 | 1.0000 | pole.b13 | 1.0000 | 1.0000 | 0 | 2 | 1 | 49.7 |
| 5 | 2 | 2/2 (100.0%) | 1.0000 | 1.0000 | double_pole.b32 | 1.0000 | 1.0000 | 0 | 2 | 1 | 26.6 |
| 6 | 2 | 2/2 (100.0%) | 0.5000 | 0.5000 | mnist.b292 | 1.0000 | 1.0000 | 2 | 0 | 2 | 7.9 |
| 7 | 2 | 2/2 (100.0%) | 0.1000 | 0.1000 | cifar100.b26 | 0.9500 | 0.8469 | 0 | 2 | 0 | 681.8 |
| 8 | 2 | 2/2 (100.0%) | 0.6154 | 0.5000 | ecg.b1285 | 1.0000 | 0.9902 | 0 | 2 | 0 | 680.7 |
| 9 | 2 | 2/2 (100.0%) | 1.0000 | 0.8462 | satellite.b14767 | 1.0000 | 0.9706 | 1 | 1 | 1 | 349.6 |
| 10 | 2 | 2/2 (100.0%) | 0.8462 | 0.6538 | ninapro.b2 | 1.0000 | 1.0000 | 1 | 1 | 1 | 187.2 |
| 11 | 2 | 2/2 (100.0%) | 0.0000 | 0.0000 | spherical.b639 | 1.0000 | 0.6863 | 0 | 2 | 0 | 683.1 |
| 12 | 2 | 2/2 (100.0%) | 0.9965 | 0.9945 | cosmic.b201 | 0.9963 | 0.9941 | 2 | 0 | 1 | 421.9 |
| 13 | 2 | 2/2 (100.0%) | 0.6745 | 0.6644 | darcy_flow.b14 | 0.7010 | 0.6861 | 0 | 2 | 0 | 680.5 |
| 14 | 2 | 2/2 (100.0%) | 0.5006 | 0.4708 | psicov.b22 | 0.8623 | 0.8106 | 0 | 2 | 0 | 681.0 |
| 15 | 2 | 2/2 (100.0%) | 0.9842 | 0.9837 | fsd50k.b7177 | 0.9885 | 0.9857 | 2 | 0 | 2 | 47.3 |
| 16 | 2 | 2/2 (100.0%) | 0.8825 | 0.8355 | deepsea.b41 | 0.9684 | 0.9412 | 0 | 2 | 1 | 680.5 |
| 17 | 2 | 2/2 (100.0%) | 0.3077 | 0.2308 | pgm.train.b8877 | 0.7451 | 0.4902 | 0 | 2 | 0 | 982.7 |
| 18 | 2 | 2/2 (100.0%) | 0.8182 | 0.6695 | arc.test.9bebae7a | 1.0000 | 1.0000 | 0 | 2 | 2 | 366.0 |

### One-sentence rung summaries

- Rung 1 covered 2/2 held-out queries (mean 1.0000, best 1.0000); the mean support-to-query gap was 0.0000, with 2 failure(s).
- Rung 2 covered 2/2 held-out queries (mean 0.7308, best 1.0000); the mean support-to-query gap was 0.2692, with 2 failure(s).
- Rung 3 covered 2/2 held-out queries (mean 0.4688, best 0.4688); the mean support-to-query gap was 0.0957, with 2 failure(s).
- Rung 4 covered 2/2 held-out queries (mean 1.0000, best 1.0000); the mean support-to-query gap was 0.0000, with 2 failure(s).
- Rung 5 covered 2/2 held-out queries (mean 1.0000, best 1.0000); the mean support-to-query gap was 0.0000, with 2 failure(s).
- Rung 6 covered 2/2 held-out queries (mean 0.5000, best 0.5000); the mean support-to-query gap was 0.5000, with 0 failure(s).
- Rung 7 covered 2/2 held-out queries (mean 0.1000, best 0.1000); the mean support-to-query gap was 0.7469, with 2 failure(s).
- Rung 8 covered 2/2 held-out queries (mean 0.5000, best 0.6154); the mean support-to-query gap was 0.4902, with 2 failure(s).
- Rung 9 covered 2/2 held-out queries (mean 0.8462, best 1.0000); the mean support-to-query gap was 0.1244, with 1 failure(s).
- Rung 10 covered 2/2 held-out queries (mean 0.6538, best 0.8462); the mean support-to-query gap was 0.3462, with 1 failure(s).
- Rung 11 covered 2/2 held-out queries (mean 0.0000, best 0.0000); the mean support-to-query gap was 0.6863, with 2 failure(s).
- Rung 12 covered 2/2 held-out queries (mean 0.9945, best 0.9965); the mean support-to-query gap was -0.0004, with 0 failure(s).
- Rung 13 covered 2/2 held-out queries (mean 0.6644, best 0.6745); the mean support-to-query gap was 0.0217, with 2 failure(s).
- Rung 14 covered 2/2 held-out queries (mean 0.4708, best 0.5006); the mean support-to-query gap was 0.3398, with 2 failure(s).
- Rung 15 covered 2/2 held-out queries (mean 0.9837, best 0.9842); the mean support-to-query gap was 0.0020, with 0 failure(s).
- Rung 16 covered 2/2 held-out queries (mean 0.8355, best 0.8825); the mean support-to-query gap was 0.1057, with 2 failure(s).
- Rung 17 covered 2/2 held-out queries (mean 0.2308, best 0.3077); the mean support-to-query gap was 0.2594, with 2 failure(s).
- Rung 18 covered 2/2 held-out queries (mean 0.6695, best 0.8182); the mean support-to-query gap was 0.3305, with 2 failure(s).

## Generalization gaps

Across tasks with both measurements, mean support minus held-out query accuracy was 0.2399. Held-out query accuracy is the primary reported outcome; support accuracy measures search-time fitting and is not a substitute.

## Strategy usage

Selected/admission paths:

- `composition`: 3 task record(s)
- `direct`: 10 task record(s)
- `field`: 1 task record(s)
- `routed`: 21 task record(s)

Held-out evaluated paths:

- `composition`: 3 held-out evaluation(s)
- `direct`: 6 held-out evaluation(s)
- `lookup`: 1 held-out evaluation(s)
- `routed`: 26 held-out evaluation(s)

## Decomposition, refinement, and library behavior

The ledger records 0 decomposition-marked task(s), 0 refinement outcome(s), and 1 library-hit outcome(s). Library size moved from 0 to 18 entries, peaked at 19, admitted 19 entries, and removed 1 during GC.
Routing was exercised on 34 task(s): 4 distilled result(s), 14 undistillable result(s), and mean recorded distillation gap 0.6427.

## Timing and resource events

Recorded task time totals 6702.8 seconds, including 15 deadline-marked task(s).

- `composition`: 107.6 seconds
- `cross_validation`: 56.5 seconds
- `decompose`: 0.4 seconds
- `decompose_first`: 14.0 seconds
- `direct`: 4037.4 seconds
- `field`: 1571.0 seconds
- `grammar`: 235.1 seconds
- `routed`: 331.4 seconds
- `task_pool_discovery`: 55.1 seconds
- `task_materialization`: 10.9 seconds

Resource metric events:

- `composition_resource_device_budget_bytes`: 13
- `composition_resource_device_required_bytes`: 13
- `composition_resource_glue_values`: 13
- `composition_resource_host_budget_bytes`: 13
- `composition_resource_host_required_bytes`: 13
- `composition_stage_activation_bytes`: 13
- `composition_stage_candidate_bytes`: 13
- `composition_stage_optimizer_bytes`: 13
- `composition_stage_population_bytes`: 13
- `composition_stage_total_bytes`: 13
- `composition_stage_transfer_bytes`: 13
- `composition_stage_work_operations`: 13
- `direct_resource_activation_bytes`: 26
- `direct_resource_candidate_bytes`: 26
- `direct_resource_optimizer_bytes`: 26
- `direct_resource_population_bytes`: 26
- `direct_resource_total_bytes`: 26
- `direct_resource_transfer_bytes`: 26
- `direct_resource_work_operations`: 26
- `field_resource_activation_bytes`: 7
- `field_resource_candidate_bytes`: 7
- `field_resource_optimizer_bytes`: 7
- `field_resource_population_bytes`: 7
- `field_resource_total_bytes`: 7
- `field_resource_work_operations`: 7
- `routed_distill_resource_device_budget_bytes`: 17
- `routed_distill_resource_device_required_bytes`: 17
- `routed_distill_resource_glue_values`: 17
- `routed_distill_resource_host_budget_bytes`: 17
- `routed_distill_resource_host_required_bytes`: 17

## Limitations

- Missing held-out values are excluded from aggregates and rendered as N/A; valid zeroes are retained.
- The report is observational and cannot establish causality or matched-baseline superiority.
- Live end-state library metadata is index-level; the immutable starting identity hashes payload files without decoding them.
- Attempt-weighted aggregates include configured revisits; first-occurrence aggregates count each rung/task identity once.
