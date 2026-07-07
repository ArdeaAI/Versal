# paper_todo.md: from draft to publishable

Companion to `ai/paper.md` (draft of 2026-07-06). The draft is honest about its evidence, which means
the gap to publishable is enumerable. This file is that enumeration: a checklist with the plan baked
in. Items marked **P0** block any submission; **P1** block a strong submission; **P2** are
polish/stretch. No em dashes anywhere in the paper; keep it that way.

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
- [ ] **Decide and document the metric protocol**: accept bar, what "solved" means, how interrupted
      runs are treated (the draft reports snapshots; a reviewer will ask).

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
- [ ] **A production decomposition** (P1): the mechanism has never fired outside tests. Either
      surface a rung where it fires naturally (interrupt budgets on rungs 6-10 may do it) or add a
      synthetic decomposable task family to Icarus and show the full recurse-admit-reassemble path
      end to end in a real run. Without this, soften S4.5 or mark it explicitly as untested-in-anger
      (it currently is marked; a demo is better).
- [ ] **Rungs 6-10 campaign** (P1): rung 6 now has 66 cold attempts on record (0 solves, best
      0.900, in the completed run); extend to rungs 7-10 with geometry operators +
      max_task_seconds on, report solve/fail honestly. The paper's image-wall narrative needs at
      least one full documented attempt at scale beyond rung 6. Rung 6's 0.900-vs-0.95 gap is
      close enough that budget/steps tuning may flip it: cheap, high-value probe.
- [ ] **Rungs 11-14 unblock or descope** (P1): one of (a) cppn init at scale (its designed payoff;
      measure genes-at-init vs minimal), (b) output-side factorization design pass, (c) explicit
      descope in the paper. Currently the draft says "blocked outright", which is honest but invites
      "why not the init you built for exactly this".
- [ ] **Motif census at scale** (P2): run on a 100+ entry library; the draft admits the current
      census reads as plumbing statistics.
- [ ] **Library scale stress** (P2): synthetic 1k-entry library; measure index rewrite, query, GC;
      the draft cites watchpoints, back them with numbers or fix before review.

## 5. Figures and tables (P0; the draft has zero figures)

- [ ] Fig 1: system diagram of the solve ladder (lookup -> refine -> routed/direct/composition ->
      decompose -> admit, with the library at center). The walkthrough's mermaid chart is the seed.
- [ ] Fig 2: marginal-cost-per-task curves by rung over task index (log seconds), cold vs warm.
      Data already in run_summary.json rows.
- [ ] Fig 3: two_spirals trajectory plot: per-attempt metric for G0/G1/warm/probe arms (seed sweep
      bands once available), 0.95 bar drawn.
- [ ] Fig 4: structure-vs-weights scatter: trained metric vs max_sample_accuracy per champion across
      all runs; the representation-wall figure. Data in sample_metrics of every attempt row.
- [ ] Fig 5: library render: the overmind portrait plus 2-3 evolved nets (library/images/ has these)
      including the level-3 chain and the warm-G1 seven-macro artifact.
- [ ] Fig 6 (P1): motif atlas excerpt (uv run motif_census --render).
- [ ] Table: per-rung results with CIs (after item 1); ablation table (after item 2); baseline table
      (after item 3).

## 6. Writing mechanics (P0 unless noted)

- [ ] **Citation verification pass.** All references were web-verified against arXiv/DOI during
      drafting EXCEPT four added from author knowledge: Deb et al. 2002 (NSGA-II), Kingma & Ba 2015
      (Adam), Wernicke 2006 (ESU), Milo et al. 2002 (network motifs). Verify those four; spot-check
      arXiv ids of the rest; add page numbers where venues want them.
- [ ] **Related-work gap sweep**: Voyager and Meyerson & Miikkulainen soft ordering are now cited
      (added in review). Still to sweep: lifelong/continual NAS (growth-based continual NAS
      specifically), "Lamarckian NAS", learned DSL/library systems newer than DreamCoder, any
      2025-2026 persistent-library neuroevolution work. One focused literature session; a reviewer
      in ENAS will look for these by name.
- [ ] **Terminology audit**: "overmind" appears only in repo internals; the paper says "routed
      substrate" throughout. Keep. Decide the public system name (ArdEVO) and benchmark name
      (Icarus) and check hyphenation/capitalization consistency. Check "ENAS" is not overloaded with
      Pham et al.'s ENAS (rename ours to "evolutionary NAS" in prose where ambiguous).
- [ ] **Numbers audit**: every quantitative claim in S7/S8 traced back to a run file or bench
      docstring (the draft was written from the repo's artifacts; re-verify after the reruns replace
      snapshot numbers, especially the flagship run's 32-task snapshot which will be superseded by
      its completion).
- [ ] **LaTeX conversion** (venue style; NeurIPS or ICML template), figure pipeline, bib file
      generated from the references section.
- [ ] **Venue decision**: options with different framings:
      (a) NeurIPS/ICML main: needs items 1-3 complete plus at least one resolved headline (probe GO,
      or routed-at-scale);
      (b) NeurIPS Datasets & Benchmarks: lead with Icarus + the diagnostic protocol; system becomes
      the reference implementation; lower results bar, needs dataset documentation (below);
      (c) GECCO/ALIFE: system + case-study framing fits as-is after items 1-2;
      (d) arXiv preprint now, venue later.
      Recommendation: (d) immediately after items 1-2 land, then (a) or (b) by results.
- [ ] **Icarus dataset documentation** (P0 if venue (b)): per-rung task counts for rungs 6-18
      verified live (`uv run rung_doctor --rungs 1-18` prints them; the draft's rung table has
      blanks), generation process description from the Icarus-Dataset repo, licensing, datasheet,
      hosting statement. Also fix the known vendored default bug (underscore repo name) upstream and
      note the [-1,1] continuous-encoding suggestion from gate E.
- [ ] **Reproducibility appendix**: exact config files per experiment (frozen copies committed),
      seeds, hardware, library state provenance (cold/warm), code release plan + license decision
      (repo LICENSE.md is currently Ardea internal-use; publishing requires resolving this),
      dataset HF link.
- [ ] **Broader impact / compute statement** (venue-dependent, P1): open-ended search safety
      paragraph, total compute used (reconstructable from run summaries' seconds fields).
- [ ] **Acknowledgments/authorship**: confirm the AI-assistance disclosure wording and author list.

## 7. Infrastructure to make 1-4 cheap (P0, do first)

- [ ] **Multi-seed multi-config runner**: a small driver that takes (config, seed list, cold/warm),
      runs sequentially or via ClearML lattice queues, and aggregates run_summary.json rows into a
      results parquet/CSV. Most of results.py exists; this is glue plus an aggregation script.
- [ ] **Plotting notebook/script** for Figs 2-4 straight from aggregated rows.
- [ ] **Remote execution hygiene**: push before enqueue (agent clones the repo; untracked files
      never arrive), configs committed per experiment tag. LatticeCUDA also resolves the batched
      trainer crossover claim (bench T4/T5 on cuda) which S8 currently leaves as "expected".
- [ ] **Run-provenance stamp**: write the config (or its hash) INTO run_summary.json. The results
      dossier had to identify runs' configs by schedule shape; that is one line to fix and a
      reviewer-facing reproducibility hole.
- [ ] **Freeze artifacts per experiment.** Drafting incident worth institutionalizing: the library
      and results moved/regrew UNDER the draft (a run completed mid-write; `library/motifs.json`
      was quoted from a different library state and had to be corrected in review). Every number in
      the paper must point at an immutable archived copy (`ai/archive/<tag>/`), and the motif
      census must be regenerated against, and labeled with, each frozen library snapshot.

## 8. Claims audit before submission (P0, final pass)

Walk the paper and re-justify or soften each of these against the then-current evidence:

- [ ] "routed can carry load" (currently one archived run at an older config).
- [ ] "266 of 267 solved" and every completed-run number (re-derive from the frozen artifact at
      submission; the S7.2 table currently reads `results/20260706_024726_orchestrated` directly).
- [ ] The cross-generation two_spirals table (replace with the frozen-config seed-sweep version;
      keep the historical table only as an appendix narrative if at all). The single above-chance
      diagnostic reading (0.688 in the stone-seeded snapshot) needs replication before it is more
      than a hint.
- [ ] "decompose never fired in production" (either demo or keep the honest statement).
- [ ] "no evolutionary-computation dependency", "roughly 13k lines", test count 461 (re-count at
      submission; 461 verified via pytest --collect-only on 2026-07-06).
- [ ] Every superlative ("first", "only") if any survive editing; the draft avoids them, keep it so.

---

## Suggested order of attack

1. Item 7 (runner + provenance stamp), because everything else multiplies through it.
2. Item 1 (seed sweeps + finished flagship + cold/warm A/B) and item 3's no-memory control, in one
   batch on lattice.
3. Item 2's P0 rows, same batch mechanism.
4. Item 4's probe (flip-to-GO), since its outcome may change the paper's headline.
5. Items 5-6 (figures, LaTeX, docs) while runs execute.
6. Item 8 last, against final numbers.
