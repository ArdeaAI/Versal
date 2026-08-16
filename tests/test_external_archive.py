"""External experiment snapshots are coherent, namespaced, restorable, and offline-testable."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import versal.external_archive as archive_module
from versal import rendering
from versal.external_archive import ArchiveManager, ExperimentLock, list_runs, restore_snapshot
from versal.tools.experiment_archive import main as archive_main
from versal.trials.orchestrated_trial import OrchestratedTrial


def _experiment_tree(root: Path, name: str) -> tuple[Path, Path]:
    run = root / f"{name}_run"
    library = root / f"{name}_library"
    (run / "task_0001").mkdir(parents=True)
    (run / "checkpoint.json").write_text('{"task_cursor": 1}')
    (run / "run_summary.json").write_text('{"status": "running"}')
    (run / "task_0001" / "net.png").write_bytes(b"rendered-network")
    (library / "entries").mkdir(parents=True)
    (library / "index.json").write_text('[{"key": "m1_fixture"}]')
    (library / "entries" / "m1_fixture.json").write_text('{"key": "m1_fixture"}')
    return run, library


def _manager(uri: str, run: Path, library: Path, run_key: str) -> ArchiveManager:
    manager = ArchiveManager.from_config(
        {
            "archive": {"enabled": True, "backend": "file", "uri": uri, "run_key": run_key},
            "seed": 0,
            "config_effective_sha256": "a" * 64,
        },
        run,
        library,
    )
    assert manager is not None
    return manager


def test_file_archive_snapshot_verify_restore_and_catalog(tmp_path: Path) -> None:
    run, library = _experiment_tree(tmp_path, "alpha")
    remote = tmp_path / "remote"
    uri = remote.resolve().as_uri()
    manager = _manager(uri, run, library, "alpha-seed0")

    manifest = manager.snapshot(1)
    local_state = json.loads((run / "archive_state.json").read_text())
    assert local_state["status"] == "complete" and local_state["snapshot_status"] == "running"

    snapshot_id = manifest["snapshot_id"]
    prefix = remote / "runs" / "alpha-seed0"
    assert not (remote / "latest.json").exists()
    assert json.loads((prefix / "latest.json").read_text())["snapshot_id"] == snapshot_id
    assert not (prefix / "snapshots" / snapshot_id / "INCOMPLETE.json").exists()
    assert list_runs(uri) == [
        {
            "schema_version": 1,
            "run_key": "alpha-seed0",
            "run_name": run.name,
            "library_name": library.name,
            "latest_snapshot_id": snapshot_id,
            "latest_status": "running",
            "latest_task_cursor": 1,
            "updated_unix": pytest.approx(json.loads((prefix / "run.json").read_text())["updated_unix"]),
        }
    ]

    verify_destination = tmp_path / "verify-must-not-exist"
    verified = restore_snapshot(uri, verify_destination, run_key="alpha-seed0", verify_only=True)
    assert verified["payload_sha256"] == manifest["payload_sha256"]
    assert not verify_destination.exists()

    restored = tmp_path / "restored"
    restore_snapshot(uri, restored, run_key="alpha-seed0")
    assert (restored / "run" / "checkpoint.json").read_text() == '{"task_cursor": 1}'
    assert (restored / "run" / "task_0001" / "net.png").read_bytes() == b"rendered-network"
    assert (restored / "library" / "entries" / "m1_fixture.json").exists()
    for entry in manifest["files"]:
        path = restored / entry["path"]
        assert path.stat().st_size == entry["size"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def test_shared_uri_keeps_run_pointers_independent(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    uri = remote.resolve().as_uri()
    alpha_run, alpha_library = _experiment_tree(tmp_path, "alpha")
    beta_run, beta_library = _experiment_tree(tmp_path, "beta")

    alpha = _manager(uri, alpha_run, alpha_library, "alpha")
    beta = _manager(uri, beta_run, beta_library, "beta")
    alpha_manifest = alpha.snapshot(1)
    beta_manifest = beta.snapshot(2)
    alpha_retry = alpha.snapshot(1)

    assert alpha_retry["snapshot_id"] != alpha_manifest["snapshot_id"]
    assert json.loads((remote / "runs" / "alpha" / "latest.json").read_text())["snapshot_id"] == alpha_retry["snapshot_id"]
    assert json.loads((remote / "runs" / "beta" / "latest.json").read_text())["snapshot_id"] == beta_manifest["snapshot_id"]
    assert [record["run_key"] for record in list_runs(uri)] == ["alpha", "beta"]


def test_verify_rejects_tampered_payload_and_restore_rejects_nonempty_destination(tmp_path: Path) -> None:
    run, library = _experiment_tree(tmp_path, "alpha")
    remote = tmp_path / "remote"
    uri = remote.resolve().as_uri()
    manager = _manager(uri, run, library, "alpha")
    manifest = manager.snapshot(1)
    payload = remote / "runs" / "alpha" / "snapshots" / manifest["snapshot_id"] / "snapshot.tar.gz"
    payload.write_bytes(payload.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="payload hash mismatch"):
        restore_snapshot(uri, tmp_path / "unused", run_key="alpha", verify_only=True)

    # A fresh snapshot restores only into an empty directory, so existing work is never merged.
    manager.snapshot(2)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="not empty"):
        restore_snapshot(uri, occupied, run_key="alpha")
    assert (occupied / "keep.txt").read_text() == "keep"


def test_verify_rejects_manifest_snapshot_id_redirect(tmp_path: Path) -> None:
    run, library = _experiment_tree(tmp_path, "alpha")
    remote = tmp_path / "remote"
    uri = remote.resolve().as_uri()
    manifest = _manager(uri, run, library, "alpha").snapshot(1)
    path = remote / "runs" / "alpha" / "snapshots" / manifest["snapshot_id"] / "manifest.json"
    redirected = json.loads(path.read_text())
    redirected["snapshot_id"] = ".."
    path.write_text(json.dumps(redirected))

    with pytest.raises(ValueError, match="manifest ID does not match"):
        restore_snapshot(uri, tmp_path / "unused", run_key="alpha", verify_only=True)


def test_archive_config_missing_uri_and_backend_mismatch_are_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, library = _experiment_tree(tmp_path, "alpha")
    monkeypatch.delenv("VERSAL_MISSING_ARCHIVE", raising=False)

    assert ArchiveManager.from_config({"archive": {"enabled": False}}, run, library) is None
    with pytest.raises(RuntimeError, match="VERSAL_MISSING_ARCHIVE.*empty"):
        ArchiveManager.from_config({"archive": {"enabled": True, "uri_env": "VERSAL_MISSING_ARCHIVE"}}, run, library)
    with pytest.raises(ValueError, match="backend='s3'"):
        ArchiveManager.from_config({"archive": {"enabled": True, "backend": "s3", "uri": (tmp_path / "remote").resolve().as_uri()}}, run, library)

    config = {"archive": {"enabled": True, "backend": "file", "uri": (tmp_path / "remote").resolve().as_uri()}, "seed": 3, "config_sha256": "b" * 64}
    first = ArchiveManager.from_config(config, run, library)
    second = ArchiveManager.from_config(config, run, library)
    assert first is not None and second is not None
    assert first.run_key == second.run_key == (run / "archive_run_key.txt").read_text().strip()
    assert first.run_key.startswith(f"{run.name}-s3-{'b' * 12}-")
    assert len(first.run_key.rsplit("-", 1)[-1]) == 10


def test_archive_cli_lists_verifies_restores_and_reports_missing_uri(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    run, library = _experiment_tree(tmp_path, "alpha")
    remote = tmp_path / "remote"
    uri = remote.resolve().as_uri()
    _manager(uri, run, library, "alpha").snapshot(1)

    archive_main(["--uri", uri, "list"])
    assert json.loads(capsys.readouterr().out)[0]["run_key"] == "alpha"
    archive_main(["--uri", uri, "verify", "--run-key", "alpha"])
    assert json.loads(capsys.readouterr().out)["run_key"] == "alpha"
    destination = tmp_path / "cli-restore"
    archive_main(["--uri", uri, "restore", "--run-key", "alpha", "--destination", str(destination)])
    assert json.loads(capsys.readouterr().out)["run_key"] == "alpha"
    assert (destination / "run" / "checkpoint.json").exists()
    archive_main(
        [
            "--uri",
            uri,
            "snapshot",
            "--run-dir",
            str(run),
            "--library-dir",
            str(library),
            "--run-key",
            "manual",
            "--task-cursor",
            "7",
        ]
    )
    assert json.loads(capsys.readouterr().out)["run_key"] == "manual"
    assert (remote / "runs" / "manual" / "latest.json").exists()

    monkeypatch.delenv("MISSING_ARCHIVE_URI", raising=False)
    with pytest.raises(SystemExit, match="2"):
        archive_main(["--uri-env", "MISSING_ARCHIVE_URI", "list"])
    assert "archive URI is missing" in capsys.readouterr().err


def test_manual_snapshot_refuses_live_run_or_library(tmp_path: Path) -> None:
    run, library = _experiment_tree(tmp_path, "alpha")
    remote = tmp_path / "remote"
    lock = ExperimentLock(run, library)
    lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="snapshot requires quiescent state"):
            archive_main(
                [
                    "--uri",
                    remote.resolve().as_uri(),
                    "snapshot",
                    "--run-dir",
                    str(run),
                    "--library-dir",
                    str(library),
                    "--run-key",
                    "alpha",
                    "--task-cursor",
                    "1",
                ]
            )
    finally:
        lock.release()


def test_restore_copy_failure_leaves_no_partial_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, library = _experiment_tree(tmp_path, "alpha")
    remote = tmp_path / "remote"
    uri = remote.resolve().as_uri()
    _manager(uri, run, library, "alpha").snapshot(1)
    destination = tmp_path / "restored"

    monkeypatch.setattr(archive_module.shutil, "copytree", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        restore_snapshot(uri, destination, run_key="alpha")

    assert not destination.exists()
    assert list(tmp_path.glob(".restored.restore-*")) == []


def test_local_retention_keeps_newest_same_cursor_snapshot(tmp_path: Path) -> None:
    run, library = _experiment_tree(tmp_path, "alpha")
    remote = tmp_path / "remote"
    manager = _manager(remote.resolve().as_uri(), run, library, "alpha")
    manager.retain_local_snapshots = 1
    first = manager.snapshot(1, status="running")
    second = manager.snapshot(1, status="done")
    retained = list((run.parent / ".archive_snapshots" / "alpha").glob("*.tar.gz"))

    assert len(retained) == 1
    assert retained[0].name == f"{second['snapshot_id']}.tar.gz"
    assert first["snapshot_id"] != second["snapshot_id"]


def test_retention_order_uses_snapshot_timestamp_when_mtimes_tie(tmp_path: Path) -> None:
    older = tmp_path / "task-000001-running-100.tar.gz"
    newer = tmp_path / "task-000001-done-200.tar.gz"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    tied = 1_700_000_000_000_000_000
    os.utime(older, ns=(tied, tied))
    os.utime(newer, ns=(tied, tied))

    assert sorted((newer, older), key=archive_module._snapshot_sort_key) == [older, newer]


def test_trial_archive_boundary_orders_flush_checkpoint_snapshot_and_suppresses_crash_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class ManagerSpy:
        def due(self, _task_cursor: int) -> bool:
            return True

        def snapshot(self, task_cursor: int, *, status: str = "running") -> dict[str, Any]:
            events.append(f"snapshot:{task_cursor}:{status}")
            return {"snapshot_id": "ok"}

    trial = object.__new__(OrchestratedTrial)
    trial.archive_manager = ManagerSpy()
    trial._persist_resume_state = lambda _orchestrator, _state, _cursor: events.append("checkpoint")  # type: ignore[method-assign]
    monkeypatch.setattr(rendering, "flush_renders", lambda: events.append("flush"))

    result = trial._archive_boundary(SimpleNamespace(), SimpleNamespace(), 3)

    assert result == {"snapshot_id": "ok"}
    assert events == ["flush", "checkpoint", "snapshot:3:running"]

    class FailingManager(ManagerSpy):
        def snapshot(self, task_cursor: int, *, status: str = "running") -> dict[str, Any]:
            raise RuntimeError("offline")

    trial.archive_manager = FailingManager()
    assert trial._archive_boundary(SimpleNamespace(), SimpleNamespace(), 3, status="crashed-RuntimeError", force=True, best_effort=True) is None
    with pytest.raises(RuntimeError, match="offline"):
        trial._archive_boundary(SimpleNamespace(), SimpleNamespace(), 3, force=True)


def test_failed_final_archive_rewrites_done_summary_as_crashed() -> None:
    trial = object.__new__(OrchestratedTrial)
    statuses: list[str] = []
    trial._archive_boundary = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("S3 unavailable"))  # type: ignore[method-assign]
    trial._write_run_summary = lambda *_args, **kwargs: statuses.append(str(kwargs["status"]))  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="S3 unavailable"):
        trial._publish_final_archive(SimpleNamespace(), SimpleNamespace(), 180)

    assert statuses == ["crashed: external archive: RuntimeError: S3 unavailable"]


def test_task_record_and_clearml_logging_include_resource_metrics() -> None:
    trial = object.__new__(OrchestratedTrial)
    trial.task_records = []
    attempt = SimpleNamespace(
        outcome="failed",
        metric=0.1,
        strategy="composition",
        generations=0,
        depth=0,
        decompose_op=None,
        failure_stage="resource_guard",
        resource_metrics={"glue_host_required_bytes": 1024.0, "glue_declined": 1.0},
    )
    entry = SimpleNamespace(rung=14, name="psicov.fixture")

    trial._record_task(entry, attempt, [], 0)

    assert trial.task_records[0]["resource_metrics"] == attempt.resource_metrics

    scalars: list[tuple[str, str, float, int]] = []
    trial.library = type("LibrarySpy", (), {"__len__": lambda self: 0})()
    trial.skipped_rungs = []
    trial.log_scalar = lambda category, series, value, iteration: scalars.append((category, series, value, iteration))  # type: ignore[method-assign]
    trial.log_hardware_stats = lambda _iteration: None  # type: ignore[method-assign]
    orchestrator = SimpleNamespace(counters={}, attempts=[attempt])
    state = SimpleNamespace(repaired_refs=0, modules=[], species_champions={})

    trial._log_task(orchestrator, state, 1)

    assert ("Resources", "glue_host_required_bytes", 1024.0, 1) in scalars
    assert ("Resources", "glue_declined", 1.0, 1) in scalars
