#!/usr/bin/env python3
"""Verify that every manuscript evidence path exists and matches its pinned digest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "paper" / "evidence" / "manifest.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ID_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
ROLES = {"config", "library", "manuscript", "output", "render", "script", "summary", "transcript"}


class ManifestError(ValueError):
    """Raised when the evidence manifest is malformed or cannot be verified."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} must be a non-empty string")
    return value


def _validate_id(value: Any, field: str) -> str:
    identifier = _require_string(value, field)
    if ID_RE.fullmatch(identifier) is None:
        raise ManifestError(f"{field} must use lowercase kebab-case: {identifier!r}")
    return identifier


def _validate_artifact(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ManifestError(f"{field} must be an object")
    expected = {"path", "sha256", "role"}
    if set(value) != expected:
        raise ManifestError(f"{field} must contain exactly {sorted(expected)}")

    raw_path = _require_string(value["path"], f"{field}.path")
    pure_path = PurePosixPath(raw_path)
    if pure_path.is_absolute() or ".." in pure_path.parts or "\\" in raw_path or pure_path.as_posix() != raw_path:
        raise ManifestError(f"{field}.path must be a normalized repository-relative POSIX path: {raw_path!r}")

    digest = _require_string(value["sha256"], f"{field}.sha256")
    if SHA256_RE.fullmatch(digest) is None:
        raise ManifestError(f"{field}.sha256 must be a lowercase SHA256 digest")

    role = _require_string(value["role"], f"{field}.role")
    if role not in ROLES:
        raise ManifestError(f"{field}.role must be one of {sorted(ROLES)}")
    return {"path": raw_path, "sha256": digest, "role": role}


def _validate_entry(value: Any, kind: str, index: int) -> tuple[str, list[dict[str, str]]]:
    field = f"{kind}[{index}]"
    if not isinstance(value, dict):
        raise ManifestError(f"{field} must be an object")
    expected = {"id", "section", "summary", "evidence"}
    if kind == "figures":
        expected.add("outputs")
    if set(value) != expected:
        raise ManifestError(f"{field} must contain exactly {sorted(expected)}")

    identifier = _validate_id(value["id"], f"{field}.id")
    _require_string(value["section"], f"{field}.section")
    _require_string(value["summary"], f"{field}.summary")

    evidence = value["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ManifestError(f"{field}.evidence must be a non-empty array")
    artifacts = [_validate_artifact(item, f"{field}.evidence[{item_index}]") for item_index, item in enumerate(evidence)]

    if kind == "figures":
        outputs = value["outputs"]
        if not isinstance(outputs, list) or not outputs:
            raise ManifestError(f"{field}.outputs must be a non-empty array")
        artifacts.extend(_validate_artifact(item, f"{field}.outputs[{item_index}]") for item_index, item in enumerate(outputs))
    return identifier, artifacts


def validate_manifest(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, dict):
        raise ManifestError("manifest must be an object")
    expected = {"schema_version", "manuscript", "claims", "figures"}
    if set(value) != expected:
        raise ManifestError(f"manifest must contain exactly {sorted(expected)}")
    if value["schema_version"] != 1:
        raise ManifestError("schema_version must be 1")

    manuscript = _validate_artifact(value["manuscript"], "manuscript")
    if manuscript["role"] != "manuscript":
        raise ManifestError("manuscript.role must be 'manuscript'")

    artifacts = [manuscript]
    identifiers: set[str] = set()
    for kind in ("claims", "figures"):
        entries = value[kind]
        if not isinstance(entries, list) or not entries:
            raise ManifestError(f"{kind} must be a non-empty array")
        for index, entry in enumerate(entries):
            identifier, entry_artifacts = _validate_entry(entry, kind, index)
            if identifier in identifiers:
                raise ManifestError(f"duplicate claim/figure id: {identifier}")
            identifiers.add(identifier)
            artifacts.extend(entry_artifacts)
    return artifacts


def verify_manifest(manifest_path: Path = DEFAULT_MANIFEST) -> tuple[int, int]:
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest does not exist: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON in {manifest_path}: {exc}") from exc

    artifacts = validate_manifest(value)
    pinned: dict[str, str] = {}
    for artifact in artifacts:
        raw_path = artifact["path"]
        expected_digest = artifact["sha256"]
        previous = pinned.setdefault(raw_path, expected_digest)
        if previous != expected_digest:
            raise ManifestError(f"conflicting digests are pinned for {raw_path}")

    failures: list[str] = []
    for raw_path, expected_digest in sorted(pinned.items()):
        path = (REPO_ROOT / raw_path).resolve()
        try:
            path.relative_to(REPO_ROOT)
        except ValueError:
            failures.append(f"path escapes repository root: {raw_path}")
            continue
        if not path.is_file():
            failures.append(f"missing evidence: {raw_path}")
            continue
        actual_digest = sha256(path)
        if actual_digest != expected_digest:
            failures.append(f"hash mismatch: {raw_path}\n  expected {expected_digest}\n  actual   {actual_digest}")

    if failures:
        raise ManifestError("evidence verification failed:\n" + "\n".join(f"- {failure}" for failure in failures))
    return len(pinned), len(artifacts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="manifest path (default: paper/evidence/manifest.json)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    try:
        unique_files, references = verify_manifest(manifest_path.resolve())
    except ManifestError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"verified {unique_files} evidence files across {references} manifest references")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
