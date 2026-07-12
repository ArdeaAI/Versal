# ArdEVO Paper Workflow

`paper/preprint.md` is the only manuscript source. Do not edit generated LaTeX. Scientific claims and figures are pinned to immutable artifacts in `evidence/manifest.json`; its machine-readable contract is `evidence/manifest.schema.json`.

## Evidence Check

Restore the ignored `ai/archive/` evidence bundle, then run:

```bash
python paper/tools/verify_evidence.py
```

The verifier rejects malformed records, repository-escaping paths, missing files, conflicting pins, and SHA256 mismatches. Update a digest only after re-auditing the affected claim or figure against the replacement artifact. A typesetting-only preprint may use `--skip-evidence`, but submission and final builds cannot.

## NeurIPS Build

The build pins `pypandoc-binary==1.15` (Pandoc 3.6.1), authenticates the vendored official NeurIPS 2026 style, converts Markdown through the checked template and Lua filter, and runs `latexmk` with a fixed `SOURCE_DATE_EPOCH`.

```bash
uv run --with pypandoc-binary==1.15 python paper/tools/build_paper.py
uv run --with pypandoc-binary==1.15 python paper/tools/build_paper.py --mode submission
uv run --with pypandoc-binary==1.15 python paper/tools/build_paper.py --mode final
```

The default preprint is `latex/neurips_2026/ardevo-preprint.pdf`. Submission rendering removes acknowledgments and relies on the official style's anonymization. Submission and final modes refuse to build until every macro in `latex/neurips_2026/checklist.tex` is answered and the template instruction block is removed. The current manuscript remains longer than NeurIPS's nine-page main-content limit; a successful working-draft build is not a submission-readiness signal.

The official assets come from the [NeurIPS 2026 template](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip). Verify or restore the pinned copy with:

```bash
python paper/tools/sync_neurips_template.py
python paper/tools/sync_neurips_template.py --fetch
```

An upstream archive change fails closed and requires a deliberate lock review. `latex/main.tex`, `main.bbl`, and `main.pdf` are the frozen legacy conversion retained for comparison; they are not build inputs and must not be edited.
