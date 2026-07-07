# Paper figures

All figures are produced by one deterministic script (fixed jitter seed, PDF `CreationDate` stripped;
re-running yields byte-identical output):

```bash
uv run python paper/figures/make_figures.py   # from the repo root
```

Figures 1-4 are generated as vector PDF plus 200 dpi PNG. Figures 5a-5d and 6 are verbatim copies of
frozen archive renders (raster PNG only).

Design constants: Okabe-Ito colorblind-safe palette with a fixed color per rung reused across every
figure (rung 1 xor `#0072B2`, rung 2 parity `#56B4E9`, rung 3 two_spirals `#D55E00`, rung 4 pole
`#009E73`, rung 5 double_pole `#CC79A7`, rung 6 mnist family `#E69F00`; arms G0 `#000000`,
G1 `#0072B2`, G1 warm `#56B4E9`, probe `#D55E00`). One y-axis per plot, recessive light-gray grid,
log scale on all seconds axes.

## Figure 1: `fig1_ladder.pdf` / `.png`

System diagram of the per-task solve ladder (paper Section 4). Drawn programmatically; no data input.
Labels follow the prose of Sections 4.1-4.6: lookup (top-5 quick eval by I/O signature), refine on hit,
the evolve ladder (routed, direct, composition), decompose + recurse, gated admission, the wall ledger,
and the persistent library tier with its five reuse channels.

## Figure 2: `fig2_cost.pdf` / `.png`

Marginal cost per task encounter, one small-multiple panel per rung (shared log-seconds axis).

- Data: `ai/archive/20260706_flagship/results/run_summary.json`, key `tasks` (400 rows; fields
  `rung`, `seconds`, `outcome`). x = per-rung encounter index in schedule order; y = `seconds`.
- Markers: filled dot = `library_hit` or `refined`, open triangle = `evolved`, x = `failed`.

## Figure 3: `fig3_spirals.pdf` / `.png`

Two-spirals per-attempt trained metric for the four gate-experiment arms, with the 0.95 accept bar.

- G0 (20 attempts): `ai/archive/20260705_g0/results/20260705_175445_orchestrated/run_summary.json`
- G1 cold (20 attempts): `ai/archive/20260705_g1/results/20260705_183030_orchestrated/run_summary.json`
  (first attempts near 0.55, climbs late)
- G1 warm (20 attempts): `ai/archive/20260705_g1/results/20260705_185704_orchestrated/run_summary.json`
  (opens near 0.80)
- probe (3 attempts, interrupted): `ai/archive/20260705_g2/results/20260705_234350_orchestrated/run_summary.json`
- y = each task row's `metric` field, x = attempt index in run order.

## Figure 4: `fig4_wall.pdf` / `.png`

Structure-versus-weights scatter: x = `sample_metrics.max_sample_accuracy`, y = `metric`, one point
per champion row that carries a `sample_metrics` dict (222 rows total). Vertical dashed lines at the
pre-registered wall thresholds 0.70 and 0.80. Deterministic jitter (seed 20260706, x +-0.010,
y +-0.006, clipped to [0, 1]). Two-spirals rows (task name prefix `two_spirals`, all arms and runs)
are vermillion diamonds; all other rows are circles in their fixed rung color.

Sources (the `tasks` list of each):

- `ai/archive/20260706_flagship/results/run_summary.json` (148 rows with sample_metrics)
- `ai/archive/20260705_g0/results/20260705_175445_orchestrated/run_summary.json` (16)
- `ai/archive/20260705_g1/results/20260705_183030_orchestrated/run_summary.json` (11)
- `ai/archive/20260705_g1/results/20260705_185704_orchestrated/run_summary.json` (20)
- `ai/archive/20260705_g2/results/20260705_234350_orchestrated/run_summary.json` (3)
- `ai/archive/20260705_g2/results/20260706_015530_orchestrated/run_summary.json` (23)
- `ai/archive/20260705_g2/results/20260706_024019_orchestrated/run_summary.json` (1)

Rows without `sample_metrics` (217 pure library hits, which skip the hybrid evaluator, plus 34 rung-6
and 1 rung-3 failures) are excluded.

## Figure 5: curated library renders (verbatim copies)

- `fig5a_overmind.png` <- `ai/archive/20260706_flagship/library/images/overmind.png`
  The routed-substrate portrait at flagship run end: one synced expert vertex (`m1_67d6f25995a9`, 0%
  lifetime gate traffic), no trained input adapter or output head. The honest artifact behind
  "the routed strategy contributed no solves" (Section 7.2).
- `fig5b_seven_macro_artifact.png` <- `ai/archive/20260705_g1/results/20260705_185704_orchestrated/task_0009/net.png`
  Admission render of composition `c4_b68d3c3722b5` and its level-3 module `m3_c7a6e59f2e0b`
  (two_spirals, metric 0.807): seven frozen macros, six of them copies of the level-1 sin+product
  gadget `m1_3dc7886b39d2` plus one `m2_7610dbf705ee`; 73 tanh + 12 sin hidden nodes, structural
  complexity 903. The Section 7.3 "repetition assembled by hand" artifact.
- `fig5c_level3_chain.png` <- `ai/archive/20260705_g2/results/20260706_015530_orchestrated/task_0027/net.png`
  Admission render of `m3_e9604dcdaaf6` (parity.n6.b0, metric 1.0): the three-deep macro reuse chain,
  level 3 referencing `m2_39b06da8b190` referencing `m1_ae583c7687d8` (Section 7.2 snapshot run).
- `fig5d_probe_champion.png` <- `ai/archive/20260705_g2/results/20260705_234350_orchestrated/task_0001/net.png`
  The probe run's best two-spirals champion `m1_2c916927d7a3` (metric 0.917, admitted as a wall-ledger
  stepping stone): the deep free-growth topology produced by scheduled training + generative init +
  adaptive rates (Section 7.3).

Entry identities were confirmed against each archive's `library/index.json` plus the per-entry
provenance in `library/entries/<key>.json`, matched to the admitting task via the run summary's
`new_library_keys`.

## Figure 6: `fig6_motifs.png` (verbatim copy)

`ai/archive/20260706_flagship/library/images/motifs.png`: the motif census atlas mined from the
flagship run's post-GC library (`ai/archive/20260706_flagship/library/motifs.json`), motifs grouped
by diversity class with support (`s`) and instance (`n`) counts.
