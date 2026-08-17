# Paper claim audit

This file separates publishable observations from interpretations that the available experiments do not establish.

## Primary cold canary

- **Publish:** one clean seed-0 run completed 36 attempts over all 18 rungs and 34 unique task references; all 36 received held-out evaluation, including 15 deadline-marked attempts.
- **Publish:** the cross-rung task-appropriate held-out mean was 0.6099; literal sample/cell accuracy was 0.6471. For ARC, literal cell accuracy was 0.6695 but exact held-out success was 0/2.
- **Publish:** the run recorded 1,183 generations and 6,702.8 task-seconds; 7 outcomes were evolved, 1 was a library hit, and 28 were nonaccepted.
- **Publish:** the library moved from 0 to 18 indexed records (17 live) and contained level-3 compositions.
- **Do not infer:** benchmark superiority, statistical reliability, curriculum-independent performance, or a solve rate from outcome labels.

Evidence: `results/20260817_031349_orchestrated/run_summary.json` (`deeafeb…155e43`), `run_manifest.json` (`dd78ea…98389b`), `config.effective.json` (`7abb6a…44ef6`), `task_pool.json` (`9fd250…ade72`), and `library_canary_clean_seed0/index.json` (`11b1d6…19143`).

## Cosmic field reuse

- **Publish:** a field admitted on `cosmic.b432` was later retrieved for distinct `cosmic.b201`, where the field-attributed held-out score was 0.9965.
- **Publish:** both tasks were 256×256, so this is same-resolution, cross-task reuse.
- **Publish:** the lookup/refinement task took 344.1 seconds and did not improve the incumbent.
- **Do not publish:** `cross_resolution_reuse_count = 1`. The generated counter is a false positive because stored field metadata changes the `io` dictionary used by the comparison.
- **Do not attribute:** the original `cosmic.b432` task’s 0.9924 query score to the newly admitted field; that task’s report champion was routed.

Evidence: `library_canary_clean_seed0/entries/m1_3c8730a6e281.json` (`a951e4…119ad`) plus the two Cosmic task rows in the clean run summary.

## Historical persistence

- **Publish as historical observation:** the July 6 cold, rungs 1-6 run recorded 217 hits and 17 refinements in 400 encounters; excluding XOR, 168/333 encounters were hits or refinements. Its retained library reached level 5.
- **Do not infer:** causal speedup or improved generalization. There is no matched no-memory arm, task identity and outcome confound runtime, and the run predates current metric/admission semantics.

Evidence: `ai/archive/20260706_flagship/results/run_summary.json` (`04838c…8288`) and `library/index.json` (`b63e85…f6ea6`).

## XOR reproducibility and compression

- **Publish:** one fixed XOR task repeated 200 times produced 1 evolution, 7 refinements, and 192 hits, with support and held-out accuracy 1.0 throughout. Recorded accepted complexity ended at 5 nodes and 4 enabled edges.
- **Publish with qualification:** three archived seed-0 launches admitted the same initial content-addressed XOR payload in two generations, demonstrating deterministic fixed-seed reproducibility.
- **Do not publish:** 200 independent solves, multi-seed replication, mathematical minimality, or an active sine implementation.
- **Reason:** the final five-node payload was not preserved; renders omit activation identity; the preserved accepted XOR entries containing sine have sine off the active input-output path.

Evidence: `ai/archive/20260816_02_hackathon/results/20260816_185653_orchestrated/run_summary.json` (`3f4071…4b0e7`) and the archived task render.

## Digital-mind framing

- **Publish:** “digital mind” is an operational description of a persistent, inspectable organization of learned modules, compositions, routes, and lineage.
- **State explicitly:** the experiments are not evidence of consciousness, sentience, subjective welfare, preferences, selfhood, moral patienthood, general intelligence, or alignment.
- **State explicitly:** no preference, distress, flourishing, introspection, persona, or self-report assay was performed.

## Pending causal test

The checked-in quick probe compares full memory against no cross-task memory over three cycles and three seeds. Until it is completed, compounding and transfer remain hypotheses supported by observational mechanism traces. The full-memory arm must actually admit at least three durable entries and produce at least two hits/refinements on exact revisits before the comparison is interpretable.
