# Paper TODO: Evidence Before Submission

Companion to the purpose-led conference core and technical supplement, revised 2026-07-15. **P0** items block submission, **P1** items block a strong empirical paper, and **P2** items are follow-up work. Checked tooling is not evidence.

## Current boundary

- [x] **Purpose-led rewrite.** The paper motivates persistent neuroevolution from the difficulty of hand-specifying task-general intelligence, while keeping consciousness and strong recursive self-improvement outside the empirical claims.
- [x] **Two editions.** The concise conference core and extended technical report have independent evidence pins and deterministic outputs.
- [x] **Historical/current split.** July 12 remains the broad pre-change characterization; July 15 is the current-method one-task canary. No causal comparison is made.
- [x] **Literal metrics.** Query accuracy is primary; support is separate; valid zeroes remain zero; unavailable values remain `N/A`; ARC is not described as an exact-grid solve.
- [x] **Current canary integrated.** The paper records 18/18 tasks, 720 generations, 6,925.5 task-seconds, 16/18 query coverage, mean query 0.6626, mean support 0.8312, 9 evolved/2 decomposed/7 failed, and a 23-entry level-6 library.

## P0 empirical program

- [ ] **Run the multi-seed full method.** Complete at least three cold-library seeds with a pinned executable revision and immutable effective configs. Report overall and per-rung means, confidence intervals, query coverage, support-query gaps, cost, and library growth.
- [ ] **Run the causal ablations.** At minimum: full, no library reuse, no routing, no decomposition, no hierarchy, and no refinement. Keep task samples and compute envelopes matched.
- [ ] **Run external baselines.** Compare fixed MLP plus Adam, canonical NEAT, WANN-style search, and random structural search under comparable evaluation budgets.
- [ ] **Run a fresh-per-task control.** Isolate the compounding claim by solving the same stream against disposable library state.
- [ ] **Add exact structured metrics.** Record exact-grid accuracy, predicted shape, represented-cell coverage, and trivial/copy baselines for ARC and other structured outputs.
- [ ] **Repeat the two-spirals study.** Freeze one configuration, run multiple seeds, and finish the scheduled-training/generative probe. Do not treat the cross-generation historical table as a controlled curve.

## P1 mechanism studies

- [ ] **Routed handoff A/B.** Compare executable handoff on/off on a mature, fixed library; report router score, distilled score, gap, recovery, query outcome, and cost.
- [ ] **Lifecycle run.** Exercise route-edge expiry, vertex eviction, entry inactivity retirement, dependency protection, and garbage collection at campaign scale. Compare full and pruned overmind states.
- [ ] **Topology-tabu accounting.** Record unique topologies, duplicate skips, retry exhaustion, compute avoided, and whether novelty/complexity distributions change.
- [ ] **Router residency benchmark.** Measure eager versus v2 peak memory and throughput on matched signatures; archive numerical-parity output and non-destructive v1 migration evidence.
- [ ] **Motif counterfactuals.** Run null-graph ranking and matched knockout/recovery controls on a library with independent lineage roots. Use only `observed`, `functional`, and `replicated` labels defined by the protocol.
- [ ] **Deadline calibration.** Quantify soft overshoot by stage and candidate width; distinguish deadline reach from resource decline and graceful shutdown.

## P0 paper and artifact readiness

- [ ] **Pin a release revision.** Every final campaign must include the executable Git commit and dirty-state status in its evidence bundle.
- [ ] **Freeze public evidence.** Choose a durable artifact deposit, restore it from scratch, verify all hashes, and document the archive version.
- [ ] **License decision.** Finalize code, dataset, and artifact licenses.
- [ ] **Statistical pass.** Replace single-seed language and historical point estimates only after the frozen campaign lands; audit every number against the manifest.
- [ ] **Page and layout pass.** Keep the conference argument within the venue's main-content limit. Render and inspect every page of both editions, including tables, figures, references, and supplement transitions.
- [ ] **Checklist and anonymization.** Answer every official checklist item, remove template instructions, confirm author list and AI-assistance disclosure, and verify submission-mode redaction.
- [ ] **Citation pass.** Re-check every retained citation and add page/venue details required for camera-ready.

## Final claims audit

Before submission, search the manuscripts and generated TeX for these prohibited or stale claims:

- July 12 described as the current method or July 15 described as a controlled improvement;
- ARC described as solved, exact, or shape-correct without corresponding fields;
- runtime support acceptance described as held-out success;
- motif frequency described as architectural invention;
- router layout or regression parity described as campaign performance;
- current lifecycle, duplicate suppression, graceful shutdown, or archive deduplication described as canary-measured when they were only test-backed;
- recursive library improvement described as code, objective, evaluator, or curriculum self-modification;
- benchmark capability described as evidence of qualia, objectness, selfhood, sentience, or consciousness;
- full-cluster, ablation, or baseline tooling described as completed results.

## Recommended order

1. Freeze a revision and run full, no-library, and fresh-per-task arms first.
2. Run the remaining P0 ablations and external baselines in the same campaign envelope.
3. Add structured metrics and rerun the upper ladder before making any ARC capability claim.
4. Complete the routed, lifecycle, tabu, router-memory, and motif studies.
5. Recompute figures and tables from the final evidence, verify both editions, then complete venue and release tasks.
