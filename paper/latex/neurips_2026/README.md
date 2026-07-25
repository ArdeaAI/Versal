# NeurIPS 2026 Generated Paper

This directory contains the authenticated official `neurips_2026.sty`, the editable official checklist, the checked Pandoc template, and generated mode-specific outputs.

- `ardevo-preprint.tex` and `.pdf` are generated from `paper/preprint.md`.
- `ardevo-technical-report.tex` and `.pdf` combine that core with `paper/technical_appendix.md` before the shared references.
- `neurips_2026.sty` must match `template.lock.json` byte for byte.
- `checklist.tex` may be edited only to remove its instruction block and answer the supplied questions; do not alter question text.
- `ardevo-submission.*` and `ardevo-final.*` are generated on demand and should not be hand-edited.

Run both editions from the repository root through `paper/tools/build_paper.py`; invoking Pandoc or LaTeX directly bypasses evidence and template checks. The technical report is preprint-only, while conference submission and final modes remain gated by the official checklist.
