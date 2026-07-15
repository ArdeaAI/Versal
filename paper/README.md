# ArdEVO Paper Workflow

The paper has two hand-edited Markdown sources:

- `preprint.md` is the concise conference core.
- `technical_appendix.md` is body-only supplementary material used by the technical-report edition.

Do not edit generated LaTeX or PDF files. Scientific claims, figures, and both manuscript sources are pinned in `evidence/manifest.json`; `evidence/manifest.schema.json` defines the current contract. The verifier still accepts the legacy single-manuscript schema for old evidence bundles.

## Evidence Check

Restore the ignored `ai/archive/` evidence bundle, then run:

```bash
python paper/tools/verify_evidence.py
```

The verifier rejects malformed records, repository-escaping paths, missing files, conflicting pins, and SHA256 mismatches. Update a digest only after re-auditing the corresponding prose or figure against the replacement artifact. A typesetting-only preprint may use `--skip-evidence`; submission and final builds cannot.

The manuscript distinguishes three evidence eras:

- July 5-6: historical lower-ladder compounding and two-spirals diagnostics.
- July 12: broad, ten-task-per-rung pre-change characterization.
- July 15: current-method, one-task-per-rung functionality canary.

Do not compare the two canaries causally. They differ in code, task sample, task count, budgets, and library trajectory. ARC values are held-out cell/sample accuracy unless an artifact explicitly contains exact-grid fields.

## Conference Edition

The default build is the concise conference preprint:

```bash
uv run --with pypandoc-binary==1.15 python paper/tools/build_paper.py
uv run --with pypandoc-binary==1.15 python paper/tools/build_paper.py --edition conference --mode preprint
uv run --with pypandoc-binary==1.15 python paper/tools/build_paper.py --edition conference --mode submission
uv run --with pypandoc-binary==1.15 python paper/tools/build_paper.py --edition conference --mode final
```

Outputs are `latex/neurips_2026/ardevo-{mode}.tex` and `.pdf`. Submission rendering removes acknowledgments and uses the official style's anonymization. Submission and final modes fail closed until every checklist macro is answered, the instruction block is removed, evidence verifies, and layout checks pass.

## Technical Report Edition

The technical report combines the conference core with the supplement before the references:

```bash
uv run --with pypandoc-binary==1.15 python paper/tools/build_paper.py --edition technical-report
```

It produces `latex/neurips_2026/ardevo-technical-report.tex` and `.pdf`. The technical report supports preprint mode only; submission and final modes are intentionally rejected. Detailed rung tables, extended methods, historical diagnostics, engineering incidents, and provenance belong in the supplement rather than the conference core.

## Template and Generated Files

The build pins `pypandoc-binary==1.15` (Pandoc 3.6.1), authenticates the vendored official NeurIPS 2026 style, converts Markdown through the checked template and Lua filter, and runs `latexmk` under a fixed `SOURCE_DATE_EPOCH`.

Verify or restore the official assets with:

```bash
python paper/tools/sync_neurips_template.py
python paper/tools/sync_neurips_template.py --fetch
```

An upstream archive change requires a deliberate lock review. `latex/main.tex`, `main.bbl`, and `main.pdf` are frozen legacy outputs retained only for comparison; they are not build inputs and must not be edited.
