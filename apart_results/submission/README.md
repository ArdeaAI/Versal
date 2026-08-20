# Apart Research submission bundle

This directory is the local backup for the Versal Digital Minds Research Sprint submission.

- `versal_digital_minds_sprint.pdf` is the final eight-page paper.
- `versal_quick_slideshow.pptx` is the eight-slide visual presentation.
- `source/` contains the manuscript, native Icarus rung table, claim audit, evidence manifest, build files, and proposed quick ablation.
- `figures/` contains every figure used in the paper in its available PDF and PNG forms.
- `ablation/` contains the small matched persistence-probe configurations for later execution.

The clean canary run and its final library are stored beside this directory as `../20260817_031349_orchestrated/` and `../library_canary_clean_seed0/`. Historical evidence selected for the paper is under `../evidence/historical/`.

From the repository root, rebuild the paper with:

```text
uv run python apart_results/submission/source/build_submission.py
```

Verify the pinned evidence with:

```text
uv run python paper/tools/verify_evidence.py --manifest apart_results/submission/source/evidence_manifest.json
```

Final PDF SHA-256: `69dcda5ac4025028bcfbfa647564fe7b5606d7f207ec6387f15405bdd41a6bc9`

Final slideshow SHA-256: `795225148445aa5c1710390060ca14d961c989e5f7ef2984b5d9453b7a73c465`
