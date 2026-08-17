# Apart Research submission bundle

This directory is the local backup for the Versal Digital Minds Research Sprint submission.

- `versal_digital_minds_sprint.pdf` is the final eight-page paper.
- `source/` contains the manuscript, native Icarus rung table, claim audit, evidence manifest, build files, and proposed quick ablation.
- `figures/` contains every figure used in the paper in its available PDF and PNG forms.
- `ablation/` contains the small matched persistence-probe configurations for later execution.

The clean canary run and its final library are stored beside this directory as `../20260817_031349_orchestrated/` and `../library_canary_clean_seed0/`. Historical evidence selected for the paper is under `../evidence/historical/`.

From the repository root, rebuild the paper with:

```text
uv run python ai/for_apart/build_submission.py
```

Verify the pinned evidence with:

```text
uv run python paper/tools/verify_evidence.py --manifest ai/for_apart/evidence_manifest.json
```

Final PDF SHA-256: `f8fe39e84c302273dc8a72221b42747e0520531ba13c01eb1b9beabf798b7d81`
