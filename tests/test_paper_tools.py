from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from paper.tools import build_paper, verify_evidence


def _artifact(path: str, role: str = "summary", digest: str | None = None) -> dict[str, str]:
    return {"path": path, "sha256": digest or "0" * 64, "role": role}


def _claim(artifact: dict[str, str]) -> dict[str, Any]:
    return {"id": "measured-result", "section": "Results", "summary": "A measured result.", "evidence": [artifact]}


def _figure(evidence: dict[str, str], output: dict[str, str]) -> dict[str, Any]:
    return {
        "id": "result-figure",
        "section": "Results",
        "summary": "A result figure.",
        "evidence": [evidence],
        "outputs": [output],
    }


def _v1_manifest() -> dict[str, Any]:
    evidence = _artifact("evidence/result.json")
    return {
        "schema_version": 1,
        "manuscript": _artifact("paper/preprint.md", "manuscript"),
        "claims": [_claim(evidence)],
        "figures": [_figure(evidence, _artifact("paper/figures/result.png", "output"))],
    }


def _v2_manifest() -> dict[str, Any]:
    evidence = _artifact("evidence/result.json")
    return {
        "schema_version": 2,
        "manuscripts": [
            {"id": "conference-core", "artifact": _artifact("paper/preprint.md", "manuscript")},
            {"id": "technical-appendix", "artifact": _artifact("paper/technical_appendix.md", "manuscript")},
        ],
        "claims": [_claim(evidence)],
        "figures": [_figure(evidence, _artifact("paper/figures/result.png", "output"))],
    }


def _core_source() -> str:
    return """# ArdEVO

**Researcher**
Institute
researcher@example.com

---

## Abstract

An abstract.

---

## 1 Why Evolve Intelligence?

Core argument.

## Acknowledgments

Thanks.

---

## References

References remain last.
"""


def test_build_cli_defaults_to_compatible_conference_preprint() -> None:
    args = build_paper.parse_args([])

    assert args.edition == "conference"
    assert args.mode == "preprint"
    assert build_paper.tex_path_for(args.edition, args.mode).name == "ardevo-preprint.tex"


@pytest.mark.parametrize(
    ("edition", "mode", "expected"),
    [
        ("conference", "preprint", "ardevo-preprint.tex"),
        ("conference", "submission", "ardevo-submission.tex"),
        ("conference", "final", "ardevo-final.tex"),
        ("technical-report", "preprint", "ardevo-technical-report.tex"),
    ],
)
def test_output_names_are_deterministic(edition: str, mode: str, expected: str) -> None:
    assert build_paper.tex_path_for(edition, mode).name == expected


def test_technical_report_inserts_body_only_supplement_before_references() -> None:
    assembled = build_paper.assemble_technical_report(_core_source(), "\n## Technical Supplement\n\nExtended evidence.\n")
    manuscript = build_paper.parse_manuscript(assembled)

    assert assembled.index("## Technical Supplement") < assembled.index("## References")
    assert manuscript.body.count("## Technical Supplement") == 1
    assert manuscript.body.rstrip().endswith("References remain last.")
    assert manuscript.source_sha256 == hashlib.sha256(assembled.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("appendix", ["", "# Duplicate title\n", "Plain body without a heading\n"])
def test_technical_report_rejects_invalid_appendix_front_matter(appendix: str) -> None:
    with pytest.raises(build_paper.BuildError, match="technical_appendix"):
        build_paper.assemble_technical_report(_core_source(), appendix)


@pytest.mark.parametrize("mode", ["submission", "final"])
def test_technical_report_rejects_submission_modes_before_building(mode: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert build_paper.main(["--edition", "technical-report", "--mode", mode, "--tex-only"]) == 2
    assert "only in preprint mode" in capsys.readouterr().err


def test_schema_v1_manifest_remains_valid() -> None:
    artifacts = verify_evidence.validate_manifest(_v1_manifest())

    assert len(artifacts) == 4
    assert artifacts[0]["path"] == "paper/preprint.md"


def test_schema_v2_validates_multiple_identified_manuscripts() -> None:
    artifacts = verify_evidence.validate_manifest(_v2_manifest())

    assert len(artifacts) == 5
    assert [artifact["path"] for artifact in artifacts[:2]] == ["paper/preprint.md", "paper/technical_appendix.md"]


def test_schema_v2_rejects_duplicate_manuscript_ids() -> None:
    manifest = _v2_manifest()
    manifest["manuscripts"][1]["id"] = "conference-core"

    with pytest.raises(verify_evidence.ManifestError, match="duplicate manuscript id"):
        verify_evidence.validate_manifest(manifest)


def test_schema_v2_requires_manuscript_roles() -> None:
    manifest = _v2_manifest()
    manifest["manuscripts"][1]["artifact"]["role"] = "summary"

    with pytest.raises(verify_evidence.ManifestError, match="role must be 'manuscript'"):
        verify_evidence.validate_manifest(manifest)


def test_verify_v2_manifest_hashes_dual_sources_and_deduplicates_references(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = {
        "paper/preprint.md": b"core",
        "paper/technical_appendix.md": b"appendix",
        "evidence/result.json": b"result",
        "paper/figures/result.png": b"figure",
    }
    for raw_path, content in files.items():
        path = tmp_path / raw_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    manifest = _v2_manifest()
    for manuscript in manifest["manuscripts"]:
        path = manuscript["artifact"]["path"]
        manuscript["artifact"]["sha256"] = hashlib.sha256(files[path]).hexdigest()
    digest = hashlib.sha256(files["evidence/result.json"]).hexdigest()
    manifest["claims"][0]["evidence"][0]["sha256"] = digest
    manifest["figures"][0]["evidence"][0]["sha256"] = digest
    manifest["figures"][0]["outputs"][0]["sha256"] = hashlib.sha256(files["paper/figures/result.png"]).hexdigest()
    manifest_path = tmp_path / "paper/evidence/manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(verify_evidence, "REPO_ROOT", tmp_path)

    assert verify_evidence.verify_manifest(manifest_path) == (4, 5)


def test_manifest_schema_declares_v1_and_v2_contracts() -> None:
    schema = json.loads((build_paper.PAPER_DIR / "evidence/manifest.schema.json").read_text(encoding="utf-8"))

    assert schema["$defs"]["manifestV1"]["properties"]["schema_version"] == {"const": 1}
    assert schema["$defs"]["manifestV2"]["properties"]["schema_version"] == {"const": 2}
    assert schema["$defs"]["manuscript"]["required"] == ["id", "artifact"]
