#!/usr/bin/env python3
"""Render the conference paper or technical report with pinned Pandoc and NeurIPS style."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "paper"
SOURCE_PATH = PAPER_DIR / "preprint.md"
TECHNICAL_APPENDIX_PATH = PAPER_DIR / "technical_appendix.md"
TOOLS_DIR = PAPER_DIR / "tools"
OUTPUT_DIR = PAPER_DIR / "latex" / "neurips_2026"
TEMPLATE_PATH = OUTPUT_DIR / "pandoc-template.tex"
FILTER_PATH = TOOLS_DIR / "manuscript.lua"
CHECKLIST_PATH = OUTPUT_DIR / "checklist.tex"
PYPANDOC_VERSION = "1.15"
PANDOC_VERSION = "3.6.1"
SOURCE_DATE_EPOCH = "946684800"
REFERENCE_BOUNDARY = "\n---\n\n## References"


@dataclass(frozen=True)
class Manuscript:
    title: str
    author_name: str
    author_affiliation: str
    author_email: str
    abstract: str
    body: str
    source_sha256: str


class BuildError(RuntimeError):
    """Raised when the manuscript cannot be rendered reproducibly."""


class PandocModule(Protocol):
    def get_pandoc_version(self) -> object: ...

    def convert_text(self, source: str, *, to: str, format: str, outputfile: str, extra_args: list[str]) -> str: ...


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def parse_manuscript(source: str) -> Manuscript:
    lines = source.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise BuildError("paper/preprint.md must start with one H1 title")
    title = lines[0][2:].strip()

    try:
        metadata_end = lines.index("---", 1)
        abstract_header = lines.index("## Abstract", metadata_end + 1)
        abstract_end = lines.index("---", abstract_header + 1)
    except ValueError as exc:
        raise BuildError("paper/preprint.md must retain its title/author, Abstract, and horizontal-rule front matter") from exc

    metadata = [line.strip() for line in lines[1:metadata_end] if line.strip()]
    if len(metadata) < 3 or not (metadata[0].startswith("**") and metadata[0].endswith("**")):
        raise BuildError("paper/preprint.md author block must start with a bold author name, affiliation, and email")
    author_name = metadata[0][2:-2].strip()
    author_affiliation = metadata[1]
    author_email = metadata[2]
    if "@" not in author_email:
        raise BuildError("paper/preprint.md author block must contain an email address")

    abstract = "\n".join(lines[abstract_header + 1 : abstract_end]).strip()
    body = "\n".join(lines[abstract_end + 1 :]).lstrip()
    if not abstract or re.match(r"## 1(?:[ .:]|$)", body) is None:
        raise BuildError("paper/preprint.md must contain a non-empty abstract followed by Section 1")
    return Manuscript(
        title=title,
        author_name=author_name,
        author_affiliation=author_affiliation,
        author_email=author_email,
        abstract=abstract,
        body=body,
        source_sha256=_sha256(source.encode("utf-8")),
    )


def assemble_technical_report(core_source: str, appendix_source: str) -> str:
    """Insert the body-only technical supplement before the shared references."""
    appendix = appendix_source.strip()
    if not appendix:
        raise BuildError("paper/technical_appendix.md must not be empty")
    if not appendix.startswith("## ") or any(line.startswith("# ") for line in appendix.splitlines()):
        raise BuildError("paper/technical_appendix.md must be body-only Markdown beginning with an H2 heading")
    if core_source.count(REFERENCE_BOUNDARY) != 1:
        raise BuildError("paper/preprint.md must contain one horizontal-rule References boundary")
    before_references, references = core_source.split(REFERENCE_BOUNDARY, maxsplit=1)
    return f"{before_references.rstrip()}\n\n---\n\n{appendix}\n\n---\n\n## References{references}"


def source_for_edition(edition: str) -> str:
    core_source = SOURCE_PATH.read_text(encoding="utf-8")
    if edition == "conference":
        return core_source
    if edition == "technical-report":
        appendix_source = TECHNICAL_APPENDIX_PATH.read_text(encoding="utf-8")
        return assemble_technical_report(core_source, appendix_source)
    raise BuildError(f"unknown paper edition: {edition}")


def tex_path_for(edition: str, mode: str) -> Path:
    if edition == "technical-report":
        return OUTPUT_DIR / "versal-technical-report.tex"
    if edition == "conference":
        return OUTPUT_DIR / f"versal-{mode}.tex"
    raise BuildError(f"unknown paper edition: {edition}")


def _without_acknowledgments(body: str) -> str:
    pattern = re.compile(r"\n## Acknowledgments\n.*?(?=\n---\n\n## References)", re.DOTALL)
    stripped, replacements = pattern.subn("", body, count=1)
    if replacements != 1:
        raise BuildError("submission rendering could not isolate the Acknowledgments section")
    return stripped


def _check_template() -> None:
    result = subprocess.run([sys.executable, str(TOOLS_DIR / "sync_neurips_template.py")], cwd=REPO_ROOT, text=True)
    if result.returncode != 0:
        raise BuildError("official NeurIPS template verification failed")


def _check_evidence() -> None:
    sys.path.insert(0, str(TOOLS_DIR))
    try:
        from verify_evidence import ManifestError, verify_manifest

        verify_manifest()
    except (ImportError, ManifestError) as exc:
        raise BuildError(str(exc)) from exc
    finally:
        sys.path.pop(0)


def _check_checklist(mode: str) -> None:
    if mode == "preprint":
        return
    checklist = CHECKLIST_PATH.read_text(encoding="utf-8")
    unresolved = checklist.count(r"\answerTODO") + checklist.count(r"\justificationTODO")
    if unresolved:
        raise BuildError(f"{mode} build refused: paper/latex/neurips_2026/checklist.tex has {unresolved} unresolved TODO macros")
    if "%%% BEGIN INSTRUCTIONS %%%" in checklist:
        raise BuildError(f"{mode} build refused: remove the official checklist instruction block after answering it")


def _load_pandoc() -> PandocModule:
    try:
        pypandoc = cast(PandocModule, importlib.import_module("pypandoc"))
    except ImportError as exc:
        raise BuildError(f"pypandoc-binary=={PYPANDOC_VERSION} is required; use the documented 'uv run --with' build command") from exc
    if str(pypandoc.get_pandoc_version()) != PANDOC_VERSION:
        raise BuildError(f"Pandoc {PANDOC_VERSION} is required, found {pypandoc.get_pandoc_version()}")
    return pypandoc


def _render_tex(manuscript: Manuscript, mode: str, destination: Path) -> bool:
    pypandoc = _load_pandoc()
    style_options = {"preprint": "preprint", "submission": "main", "final": "main,final"}[mode]
    body = _without_acknowledgments(manuscript.body) if mode == "submission" else manuscript.body
    metadata = {
        "title": manuscript.title,
        "author-name": manuscript.author_name,
        "author-affiliation": manuscript.author_affiliation,
        "author-email": manuscript.author_email,
        "abstract": manuscript.abstract,
    }

    with tempfile.TemporaryDirectory(prefix="versal-paper-") as temporary_directory:
        temporary = Path(temporary_directory)
        metadata_path = temporary / "metadata.json"
        rendered_path = temporary / destination.name
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=True), encoding="utf-8")
        extra_args = [
            "--standalone",
            f"--template={TEMPLATE_PATH}",
            f"--lua-filter={FILTER_PATH}",
            f"--metadata-file={metadata_path}",
            "--shift-heading-level-by=-1",
            "--top-level-division=section",
            "--wrap=none",
            f"--resource-path={PAPER_DIR}",
            "--variable=indent:true",
            f"--variable=style-options:{style_options}",
            f"--variable=source-sha256:{manuscript.source_sha256}",
        ]
        if mode != "preprint":
            extra_args.append("--variable=include-checklist:true")
        try:
            pypandoc.convert_text(
                body,
                to="latex",
                format="gfm+tex_math_dollars",
                outputfile=str(rendered_path),
                extra_args=extra_args,
            )
        except RuntimeError as exc:
            raise BuildError(f"Pandoc conversion failed: {exc}") from exc

        rendered = rendered_path.read_text(encoding="utf-8")
        expected_sections = sum(line.startswith("## ") for line in body.splitlines())
        expected_subsections = sum(line.startswith("### ") for line in body.splitlines())
        if rendered.count(r"\section{") != expected_sections or rendered.count(r"\subsection{") != expected_subsections:
            raise BuildError("generated LaTeX heading counts do not match paper/preprint.md")
        expected_images = len(re.findall(r"^!\[[^]]*\]\([^)]*\)$", body, flags=re.MULTILINE))
        if rendered.count(r"\includegraphics") != expected_images:
            raise BuildError("generated LaTeX image count does not match paper/preprint.md")

        previous = destination.read_text(encoding="utf-8") if destination.exists() else None
        if previous == rendered:
            return False
        destination.write_text(rendered, encoding="utf-8", newline="\n")
        return True


def _clean_auxiliary_files(stem: str) -> None:
    for suffix in (".aux", ".fdb_latexmk", ".fls", ".log", ".out"):
        (OUTPUT_DIR / f"{stem}{suffix}").unlink(missing_ok=True)


def _compile_pdf(tex_path: Path) -> Path:
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise BuildError("latexmk is required to compile the generated NeurIPS PDF")
    _clean_auxiliary_files(tex_path.stem)
    environment = os.environ.copy()
    environment.update({"FORCE_SOURCE_DATE": "1", "SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH, "TZ": "UTC"})
    command = [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", tex_path.name]
    result = subprocess.run(command, cwd=OUTPUT_DIR, env=environment, text=True, capture_output=True)
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise BuildError(f"LaTeX compilation failed; inspect {OUTPUT_DIR / (tex_path.stem + '.log')}")
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise BuildError("LaTeX reported success but did not produce a non-empty PDF")
    log_path = OUTPUT_DIR / f"{tex_path.stem}.log"
    log = log_path.read_text(encoding="utf-8", errors="replace")
    overfull = re.findall(r"Overfull \\[hv]box \(([^)]+)\)", log)
    if overfull:
        raise BuildError(f"LaTeX produced {len(overfull)} overfull boxes; inspect {log_path}")
    _clean_auxiliary_files(tex_path.stem)
    return pdf_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edition", choices=("conference", "technical-report"), default="conference")
    parser.add_argument("--mode", choices=("preprint", "submission", "final"), default="preprint")
    parser.add_argument("--tex-only", action="store_true", help="render deterministic LaTeX without invoking latexmk")
    parser.add_argument("--skip-evidence", action="store_true", help="allow a preprint typesetting build without the ignored evidence archive")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.edition == "technical-report" and args.mode != "preprint":
        print("the technical-report edition is available only in preprint mode", file=sys.stderr)
        return 2
    if args.skip_evidence and args.mode != "preprint":
        print("--skip-evidence is allowed only for non-submission preprint typesetting", file=sys.stderr)
        return 2
    try:
        source = source_for_edition(args.edition)
        manuscript = parse_manuscript(source)
        _check_template()
        if not args.skip_evidence:
            _check_evidence()
        _check_checklist(args.mode)
        tex_path = tex_path_for(args.edition, args.mode)
        changed = _render_tex(manuscript, args.mode, tex_path)
        if args.tex_only:
            print(f"{'rendered' if changed else 'verified'} {tex_path.relative_to(REPO_ROOT)}")
            return 0
        pdf_path = _compile_pdf(tex_path)
    except (BuildError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"{'rendered' if changed else 'verified'} {tex_path.relative_to(REPO_ROOT)}; built {pdf_path.relative_to(REPO_ROOT)} ({pdf_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
