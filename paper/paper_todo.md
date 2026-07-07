# paper_todo.md: from draft to publishable

Companion to `paper/paper.md` (draft of 2026-07-06, polish pass completed later the same day). The
draft is honest about its evidence, which means the gap to publishable is enumerable. This file is
that enumeration: a checklist with the plan baked in. Items marked **P0** block any submission;
**P1** block a strong submission; **P2** are polish/stretch. No em dashes anywhere in the paper;
keep it that way.

**2026-07-06 status: everything below that does not require a long run is done** (figures, citation
verification, related-work sweep, numbers/claims/terminology audits, dataset documentation,
reproducibility + compute appendices, LaTeX conversion, artifact freeze, runner + provenance).
`ai/instructions_for_publish.md` holds the exact commands and integration protocol for every
remaining run-gated item; work from that file.

The one structural advantage we have: every mechanism is a config flip that is byte-identical when
off, so almost every ablation below is a one-line config change plus a run. The bottleneck is
run-hours and seeds, not code.

---

## 1. Statistical hardening (P0)

The draft's every number is seed 0 on one machine. Nothing comparative survives review without
distributions.

- [ ] **Seed sweep the gate ladder.** Re-run G0, G1, G1-warm, and the G2 probe at seeds {0,1,2,3,4}
      under FROZEN configs (branch or tag the exact configs; the draft's cross-generation table
      currently spans evolving configurations and says so). Report mean/max/CI per arm plus paired
      per-attempt-index win rates. The existing diag_g0/g1/g2 TOMLs are the arms; only `seed` varies.
- [x] ~~Finish the flagship run~~ DONE: a completed cold 400-task run over rungs 1-6 exists
      (`results/20260706_024726_orchestrated`, status done) and now anchors S7.2.
- [ ] **Repeat the headline run at 3+ seeds, cold.** Report per-rung solve rate with CIs,
      cost-to-first-solve vs cost-on-revisit curves, refinement economics
      (attempts/improvements/decayed skips), and library growth over task index.
- [ ] **Cold vs warm A/B.** Same seed, same config, one arm starts with `library/` deleted, one with
      the mature library. This is THE compounding claim isolated; the archived warm run (76 hits/112
      tasks) previews it but was interrupted and config-drifted.
- [ ] **Independent-attempt arm.** The gate runs are 20 accumulated assaults (wall ledger on). Add a
      control with `[orchestrator.wall] ledger = false` so attempts are independent samples; report
      both. This also doubles as the wall-ledger ablation.
- [x] **Decide and document the metric protocol**: DONE in the draft as it stands: the accept bar
      (0.95 query accuracy, support fallback) and "solved" (outcomes library_hit/refined/evolved)
      are stated in S4.1/S7.1, interrupted runs are reported with their status and never silently
      mixed with completed ones (S7.3, Appendix C). Revisit only if a reviewer pushes.

## 2. Ablation matrix (P0 for the ones supporting central claims)

Each row: flip, what claim it supports, what to measure. Rungs 1-5 ladder unless noted. 3 seeds each
once the harness from item 7.1 exists.

| Flip | Supports | Measure |
|---|---|---|
| `refine.budget_k` 24 vs 0 | refinement-on-hit (S4.2, S7.2) | hit metric drift over revisits, refine counters, library tombstones |
| `wall.ledger` on/off | stepping stones (S4.3, S7.3) | two_spirals trajectory, stones admitted/improved |
| `evolve` with/without `routed`; `distill` true/false | routed substrate + distill contract (S4.4) | routed counters, wall clock, admission integrity |
| `admission` archive vs default vs accept_all | QD library (S6.2) | library size/diversity (behavior niches), downstream reuse (absorption hits, comp refs), solve rates |
| `absorb_top_k` 2 vs 0 | mid-run absorption (S6.3) | composition-strategy win rate, time-to-solve on later tasks |
| mutation set without `add_macro_node`+`add_library_module` | macro reuse (S5.1) | level-2+ emergence, two_spirals warm trajectory |
| `tweak_refine_steps` off / `refine_steps` capped 1 | TRM depth lever (S5.1) | any rung where refine>1 champions appear (check attempts ledger first) |
| `[evolution.novelty]` present vs absent; `archive_cap` 256 vs 0 | novelty (S5.6, G1) | two_spirals plateau, stall rates (novelty runs full budget) |
| `nsga2` vs `tournament` | G1 claim | already the G0/G1 pair; needs the seed sweep |
| `objectives` with/without `connection_cost` | wiring-cost modularity (S5.6) | modularity of champions (motif census), two_spirals |
| `parsimony_epsilon` 0.01 vs 0 | anti-bloat (S5.6) | champion complexity distributions, task wall clock |
| `self_adaptive` on/off | lever F (S5.2) | rate trajectories, solve rates; direct path only |
| `init` cppn vs minimal (rungs 1-5) | generative init cost (S5.3) | low-rung solve rate + complexity penalty drag |
| `w_weight_robustness` 0.5 vs 0 | robustness currency (S5.5) | library entry robustness distribution, composition/macro success rate downstream |
| `evaluate` hybrid vs standard | does the diagnostic cost anything | wall clock delta (should be small; quantify) |
| `speciation` neat vs none | speciation (S5.6) | species counts, solve rates |
| `gradient_scheduled` vs `gradient` vs `gradient_refine` on direct | training economics (S7.3) | two_spirals + parity, per-attempt seconds (scheduled measured 10-30x slower per attempt; quantify the tradeoff) |

Priority within the table: rows 1-4 and the novelty/nsga2 rows are P0 (they back the paper's named
contributions); the rest are P1.

## 3. External baselines (P0; the paper currently has none)

All under matched budgets (same task pool, same accept bar, comparable evaluations-per-task), 3 seeds.

- [ ] **Plain MLP + Adam per task**: fixed 2-layer and 3-layer widths, trained with the same
      gradient budget the direct path spends. The "why evolve at all" column.
- [ ] **Pure NEAT**: our own loop with `train = none`, `perturb_weights` on, canonical `add_node`
      (all registered already). Known to fail even XOR here; run it anyway and report, since it
      grounds the gradient-owns-weights claim.
- [ ] **WANN-style**: weight_samples evaluate as the ONLY metric (no training), rungs 1-3.
- [ ] **Random structural search**: mutation-only, no selection pressure (or random tournament),
      same operator menu. Separates "evolution works" from "the operators are enough".
- [ ] **No-memory control**: full system, `library_dir` pointed at a throwaway per task. This is the
      compounding baseline and the cheapest high-value run in this file.
- [ ] Optional P2: regularized evolution over a fixed cell space on rungs 6-7, as the
      NAS-literature anchor. Cite-and-decline is defensible if scoped out.

## 4. Open experiments the draft explicitly promises (P0/P1)

- [ ] **Two-spirals flip-to-GO probe** (P0): finish the interrupted G2 probe (20 attempts) at the
      pre-registered bar (evolved champion >= 0.95, complexity <= ~65, no-refit up-transfer within
      0.05), per-candidate step budget raised to 2-4k as gate E prescribed. The draft ends the case
      study on "pending"; a resolution either way strengthens it.
- [ ] **Routed-on-warm-library study** (P1): grow a 50-100 entry library, then measure routed
      solve/zero-shot rates, and run the `expert_ablation = "zero"` diagnostic (already in
      tests/test_routing.py) at run scale to prove frozen experts contribute signal beyond adapters.
      The draft flags routed as "unresolved bet"; this is the resolving experiment.
- [x] **A production decomposition** (P1): DONE 2026-07-07: the mechanism fired and SOLVED in the
      full-ladder coverage run (cosmic 65,536 x 65,536 decomposed, parts solved and admitted, the
      reassembled parent verified at 0.986, outcome `decomposed`; deepsea decomposed through its
      output slices in the same run). S4.5/S10 updated; artifacts frozen at
      `ai/archive/20260707_smoke18/`.
- [ ] **Rungs 6-10 campaign** (P1): IN FLIGHT: the recon_ladder seed-0 arm (rungs 1-10, 16, 17
      under max_task_seconds) is running as of 2026-07-06 evening
      (`results/20260706_090102_orchestrated`, 11/24 tasks). Scorecard it on completion and add
      seeds; rung 6's 0.900-vs-0.95 gap remains the cheap, high-value tuning probe. Instructions
      item I.
- [ ] **Rungs 11-14 unblock or descope** (P1): one of (a) cppn init at scale (its designed payoff;
      measure genes-at-init vs minimal), (b) output-side factorization design pass, (c) explicit
      descope in the paper. Currently the draft says "blocked outright", which is honest but invites
      "why not the init you built for exactly this".
- [ ] **Motif census at scale** (P2): the 35-entry flagship census is DONE and in the paper (358
      motifs, gated structure recovered); the 100+ entry census waits on a bigger library.
- [ ] **Library scale stress** (P2): synthetic 1k-entry library; measure index rewrite, query, GC;
      the draft cites watchpoints, back them with numbers or fix before review.

## 5. Figures and tables (P0; the draft has zero figures)

- [x] Fig 1: system diagram of the solve ladder. DONE 2026-07-06 (`paper/figures/fig1_ladder.pdf`,
      generated by `paper/figures/make_figures.py`).
- [x] Fig 2: marginal-cost-per-task curves by rung (log seconds). DONE (fig2_cost, from the frozen
      flagship run; cold only, warm overlay after the cold/warm A/B lands).
- [x] Fig 3: two_spirals trajectory plot, all four arms + 0.95 bar. DONE (fig3_spirals; seed bands
      once the sweep lands).
- [x] Fig 4: structure-vs-weights scatter with wall thresholds. DONE (fig4_wall, 222 champions).
- [x] Fig 5: library renders: overmind portrait, seven-macro artifact, level-3 chain, probe
      champion. DONE (fig5a-d).
- [x] Fig 6 (P1): motif atlas. DONE (fig6_motifs, census regenerated against the frozen flagship
      library).
- [ ] Table: per-rung results with CIs (after item 1); ablation table (after item 2); baseline table
      (after item 3). Run-gated; see `ai/instructions_for_publish.md` items A-G.

## 6. Writing mechanics (P0 unless noted)

- [x] **Citation verification pass.** DONE 2026-07-06: all 62 references verified against the
      arXiv export API and Crossref DOIs, including the four author-knowledge refs; zero errors.
      Page numbers where venues want them: at camera-ready.
- [x] **Related-work gap sweep**: DONE 2026-07-06: 11 verified references added (Elsken 2019a
      Lamarckian NAS, Learn to Grow, DEN, Veniat, Mendez & Eaton, LILO, ReGAL, ADAS, PropNEAT,
      OMNI-EPIC). Negative finding stated in S2.4: no published persistent-library neuroevolution
      system found.
- [x] **Terminology audit**: DONE 2026-07-06: zero "overmind" in the paper, no ENAS collision, no
      em dashes, naming consistent (ArdEVO / Icarus).
- [x] **Numbers audit**: DONE 2026-07-06: 85 claims in S7/S8 traced to frozen artifacts; 6
      mismatches fixed (tombstone accounting, 112-task failure breakdown, sin-node claim,
      diagnostic denominator, test count 513, test lines 8.7k). Re-run after each new campaign
      (protocol in `ai/instructions_for_publish.md` section 3).
- [x] **LaTeX conversion**: DONE 2026-07-06 (`paper/latex/`, venue-neutral article + generated
      references.bib; swap in the venue class at decision time).
- [ ] **Venue decision**: options with different framings:
      (a) NeurIPS/ICML main: needs items 1-3 complete plus at least one resolved headline (probe GO,
      or routed-at-scale);
      (b) NeurIPS Datasets & Benchmarks: lead with Icarus + the diagnostic protocol; system becomes
      the reference implementation; lower results bar, needs dataset documentation (below);
      (c) GECCO/ALIFE: system + case-study framing fits as-is after items 1-2;
      (d) arXiv preprint now, venue later.
      Recommendation: (d) immediately after items 1-2 land, then (a) or (b) by results.
- [x] **Icarus dataset documentation** (P0 if venue (b)): MOSTLY DONE 2026-07-06: per-rung task
      counts and I/O widths verified live for all 18 rungs (paper S3.2 table complete; corrected
      rung 11 to 10,800 x 100), new S3.4 covers generation, hosting, licensing, the underscore
      default bug, and the [-1,1] encoding note. REMAINING (user): the HF dataset card/datasheet
      itself and the upstream fixes (instructions section 4.4).
- [x] **Reproducibility appendix**: DONE 2026-07-06 (Appendix C: per-experiment config/seed/library
      table, hardware/software, provenance stamp note, license status stated openly). REMAINING
      (user): the license decision itself and the artifact deposit home (ai/* is gitignored).
- [x] **Broader impact / compute statement**: DONE 2026-07-06 (Appendix D: recorded ~6.3 h task
      wall clock for the reported experiments, single machine, no accelerators; safety paragraph).
- [ ] **Acknowledgments/authorship**: confirm the AI-assistance disclosure wording and author list
      (user decision; current wording stands in the draft).

## 7. Infrastructure to make 1-4 cheap (P0, do first)

- [x] **Multi-seed multi-config runner**: DONE (`uv run run_matrix`: config x seeds x cold/warm
      arms as crash-isolated subprocesses, CSV aggregation, per-rung tier scorecard).
- [x] **Plotting script** for Figs 1-4: DONE (`paper/figures/make_figures.py`, deterministic,
      documented in `paper/figures/README.md`; extend for CI bands when sweeps land).
- [ ] **Remote execution hygiene**: push before enqueue (agent clones the repo; untracked files
      never arrive), configs committed per experiment tag. LatticeCUDA also resolves the batched
      trainer crossover claim (bench T4/T5 on cuda) which S8 currently leaves as "expected".
      Run-gated; instructions item J.
- [x] **Run-provenance stamp**: DONE (`config_path` + `config_sha256` in every new
      run_summary.json; the flagship run predates it, documented in Appendix C).
- [x] **Freeze artifacts per experiment.** DONE for everything the paper cites:
      `ai/archive/20260706_flagship/` (run summary, post-GC library, census regenerated against
      exactly that library, rung-doctor transcript, config copy, SHA256SUMS) plus the existing
      g0/g1/g2 archives. The protocol is institutionalized in `ai/instructions_for_publish.md`
      section 3; repeat it for every future campaign.

## 8. Claims audit before submission (P0, final pass)

First pass DONE 2026-07-06 against current evidence; re-run the walk against final numbers at
submission (protocol in `ai/instructions_for_publish.md` section 3).

- [x] "routed can carry load": SHARPENED. The 92-task run's 30 wins were pre-distillation
      semantics, admitted nothing, and 19 were xor tasks lookup would clear; S7.2 and S9 now say
      so. The run-scale resolution stays run-gated (instructions item H).
- [x] "266 of 267 solved" and every completed-run number: re-derived from the frozen artifact
      (`ai/archive/20260706_flagship/`); the S7.2 table now cites the freeze, not live results/.
- [ ] The cross-generation two_spirals table: replace with the frozen-config seed-sweep version
      when instructions item A lands; the 0.688 diagnostic reading still needs replication.
- [x] "decompose never fired in production": kept honest, scoped to the current system (the
      64-decomposition incident predates the solvability gate).
- [x] "no evolutionary-computation dependency" (holds: deps are clearml/datasets/matplotlib/
      psutil/rich/torch), "roughly 13k lines" (13,683), test count updated to 513 (collected
      2026-07-06). Re-count at final freeze.
- [x] Superlatives: swept; the two found ("highest single-attempt metric", "first attempts ever")
      are now qualified.

---

## Suggested order of attack (updated 2026-07-06)

Items 5-8 and all of item 7 except remote hygiene are done; what remains is run-gated. Work from
`ai/instructions_for_publish.md`:

1. Instructions items A-D (gate sweep, headline seeds, cold/warm A/B, independent-attempt control).
2. Item E (the G3 flip-to-GO probe), since its outcome may change the paper's headline.
3. Items F-G (ablation P0 rows, baselines), same batch mechanism.
4. Items H-K as budget allows.
5. Re-run the item-8 claims walk against final numbers, update figures/tables per the integration
   protocol, then venue + license decisions and submission.
