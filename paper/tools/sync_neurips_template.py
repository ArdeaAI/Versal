#!/usr/bin/env python3
"""Fetch or verify the hash-pinned official NeurIPS 2026 LaTeX assets."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = REPO_ROOT / "paper" / "latex" / "neurips_2026"
LOCK_PATH = TEMPLATE_DIR / "template.lock.json"


class TemplateError(ValueError):
    """Raised when official template assets cannot be authenticated."""


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _load_lock() -> dict[str, Any]:
    try:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TemplateError(f"cannot read template lock {LOCK_PATH}: {exc}") from exc
    if lock.get("schema_version") != 1 or not isinstance(lock.get("source_url"), str) or not isinstance(lock.get("archive_sha256"), str):
        raise TemplateError("template lock has an unsupported shape")
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise TemplateError("template lock must contain artifacts")
    return lock


def _artifact_fields(value: Any, index: int) -> tuple[str, str, str, bool]:
    if not isinstance(value, dict) or set(value) != {"member", "path", "sha256", "editable"}:
        raise TemplateError(f"artifacts[{index}] has an unsupported shape")
    member = value["member"]
    target = value["path"]
    digest = value["sha256"]
    editable = value["editable"]
    if not all(isinstance(item, str) and item for item in (member, target, digest)) or not isinstance(editable, bool):
        raise TemplateError(f"artifacts[{index}] contains invalid values")
    if Path(member).name != member or Path(target).name != target or len(digest) != 64:
        raise TemplateError(f"artifacts[{index}] must name flat files and a SHA256 digest")
    return member, target, digest, editable


def check_assets() -> tuple[int, int]:
    lock = _load_lock()
    exact = 0
    editable = 0
    for index, value in enumerate(lock["artifacts"]):
        _, target, digest, is_editable = _artifact_fields(value, index)
        path = TEMPLATE_DIR / target
        if not path.is_file():
            raise TemplateError(f"missing NeurIPS template asset: {path.relative_to(REPO_ROOT)}")
        actual = sha256_bytes(path.read_bytes())
        if is_editable:
            editable += 1
            continue
        if actual != digest:
            raise TemplateError(f"official template asset was modified: {path.relative_to(REPO_ROOT)}\n  expected {digest}\n  actual   {actual}")
        exact += 1
    return exact, editable


def fetch_assets(*, reset_editable: bool) -> tuple[int, int]:
    lock = _load_lock()
    request = urllib.request.Request(lock["source_url"], headers={"User-Agent": "ArdEVO-paper-build/1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            archive = response.read()
    except OSError as exc:
        raise TemplateError(f"failed to download official NeurIPS template: {exc}") from exc

    actual_archive_digest = sha256_bytes(archive)
    if actual_archive_digest != lock["archive_sha256"]:
        raise TemplateError(
            f"official template archive changed; review and update template.lock.json deliberately\n  expected {lock['archive_sha256']}\n  actual   {actual_archive_digest}"
        )

    written = 0
    preserved = 0
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            for index, value in enumerate(lock["artifacts"]):
                member, target, digest, is_editable = _artifact_fields(value, index)
                content = bundle.read(member)
                if sha256_bytes(content) != digest:
                    raise TemplateError(f"archive member hash mismatch: {member}")
                destination = TEMPLATE_DIR / target
                if is_editable and destination.exists() and not reset_editable:
                    preserved += 1
                    continue
                destination.write_bytes(content)
                written += 1
    except (KeyError, zipfile.BadZipFile) as exc:
        raise TemplateError(f"official template archive has an unexpected shape: {exc}") from exc
    return written, preserved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="download authenticated assets; otherwise only verify local files")
    parser.add_argument("--reset-editable", action="store_true", help="with --fetch, replace the editable checklist with the official TODO template")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.reset_editable and not args.fetch:
        print("--reset-editable requires --fetch", file=sys.stderr)
        return 2
    try:
        if args.fetch:
            written, preserved = fetch_assets(reset_editable=args.reset_editable)
            print(f"fetched {written} NeurIPS assets; preserved {preserved} editable assets")
        exact, editable = check_assets()
    except TemplateError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"verified {exact} immutable NeurIPS assets; found {editable} editable assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
