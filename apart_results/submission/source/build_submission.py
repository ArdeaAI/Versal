"""Build and validate the Apart sprint submission PDF."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "ai" / "for_apart"
DRAFT = SOURCE_DIR / "draft.md"
FILTER = SOURCE_DIR / "submission_filter.lua"
HEADER = SOURCE_DIR / "submission_header.tex"
OUTPUT = REPO_ROOT / "output" / "pdf" / "versal_digital_minds_sprint.pdf"
BACKUP_DIR = REPO_ROOT / "apart_results" / "submission"
BACKUP = BACKUP_DIR / OUTPUT.name

FORBIDDEN_CHARACTERS = {
    "\u2010",
    "\u2011",
    "\u2012",
    "\u2013",
    "\u2014",
    "\u2015",
    "\u2212",
}


def run(*command: str) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def validate_source() -> None:
    text = DRAFT.read_text(encoding="utf-8")
    forbidden = sorted({character for character in text if character in FORBIDDEN_CHARACTERS})
    if forbidden:
        names = ", ".join(unicodedata.name(character) for character in forbidden)
        raise ValueError(f"draft contains forbidden dash characters: {names}")

    plural_tokens = (" We ", " we ", " Our ", " our ", " Us ", " us ")
    padded = f" {text} "
    found = [token.strip() for token in plural_tokens if token in padded]
    if found:
        raise ValueError(f"draft contains first-person plural tokens: {sorted(set(found))}")

    if "Screenshot%202026-07-18" in text:
        raise ValueError("draft still references the Icarus screenshot")
    if "| 18 | ARC-AGI v1 |" not in text:
        raise ValueError("draft does not contain the native 18-rung Icarus table")


def validate_pdf() -> None:
    page_output = subprocess.check_output(["pdfinfo", str(OUTPUT)], text=True)
    page_line = next(line for line in page_output.splitlines() if line.startswith("Pages:"))
    pages = int(page_line.split(":", maxsplit=1)[1].strip())
    if pages > 8:
        raise ValueError(f"submission is {pages} pages; limit is 8")

    extracted = subprocess.check_output(["pdftotext", str(OUTPUT), "-"], text=True)
    forbidden = sorted({character for character in extracted if character in FORBIDDEN_CHARACTERS})
    if forbidden:
        names = ", ".join(unicodedata.name(character) for character in forbidden)
        raise ValueError(f"PDF contains forbidden dash characters: {names}")
    if "We investigate" in extracted or "We evaluated" in extracted or "We also report" in extracted:
        raise ValueError("PDF contains first-person plural manuscript voice")
    if "The Icarus curriculum" not in extracted or "ARC-AGI v1" not in extracted:
        raise ValueError("PDF is missing the native Icarus table")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    validate_source()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    run(
        "pandoc",
        str(DRAFT),
        "--from=gfm",
        "--to=latex",
        "--standalone",
        "--pdf-engine=xelatex",
        f"--lua-filter={FILTER}",
        f"--include-in-header={HEADER}",
        "--shift-heading-level-by=-1",
        f"--resource-path={SOURCE_DIR}",
        "-V",
        "papersize=letter",
        "-V",
        "geometry:top=0.58in,bottom=0.58in,left=0.68in,right=0.68in",
        "-V",
        "mainfont=Times New Roman",
        "-V",
        "fontsize=10pt",
        "-o",
        str(OUTPUT),
    )
    validate_pdf()
    shutil.copy2(OUTPUT, BACKUP)
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")
    print(f"backed up {BACKUP.relative_to(REPO_ROOT)}")
    print(f"sha256 {sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
