"""Coherent experiment snapshots for storage outside an ephemeral compute cluster."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import tarfile
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

SCHEMA_VERSION = 1
CONTENT_SCHEMA_VERSION = 2
ACTIVE_LOCK_NAME = ".versal-active.lock"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class ObjectStore(Protocol):
    def put_file(self, source: Path, key: str) -> None: ...

    def put_bytes(self, payload: bytes, key: str) -> None: ...

    def get_file(self, key: str, destination: Path) -> None: ...

    def get_bytes(self, key: str) -> bytes: ...

    def delete(self, key: str) -> None: ...

    def list_keys(self, prefix: str) -> list[str]: ...


@dataclass
class ExperimentLock:
    """Exclusive run + library lease shared by live trials and manual snapshot commands."""

    run_dir: Path
    library_dir: Path
    _handles: list[Any] = field(default_factory=list, init=False, repr=False)

    def acquire(self) -> None:
        import fcntl

        if self._handles:
            raise RuntimeError("experiment lock is already held")
        roots = sorted({self.run_dir.resolve(), self.library_dir.resolve()}, key=str)
        try:
            for root in roots:
                root.mkdir(parents=True, exist_ok=True)
                handle = open(root / ACTIVE_LOCK_NAME, "a+")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    handle.close()
                    raise RuntimeError(f"run or library is active; snapshot requires quiescent state: {root}") from error
                self._handles.append(handle)
        except BaseException:
            self.release()
            raise

    def release(self) -> None:
        import fcntl

        for handle in reversed(self._handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        self._handles.clear()


@dataclass
class LocalObjectStore:
    root: Path

    def _path(self, key: str) -> Path:
        return self.root / PurePosixPath(key)

    def put_file(self, source: Path, key: str) -> None:
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)

    def put_bytes(self, payload: bytes, key: str) -> None:
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, destination)

    def get_file(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(key), destination)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def list_keys(self, prefix: str) -> list[str]:
        root = self._path(prefix)
        if root.is_file():
            return [str(PurePosixPath(prefix))]
        if not root.exists():
            return []
        return sorted(str(path.relative_to(self.root).as_posix()) for path in root.rglob("*") if path.is_file())


class S3ObjectStore:
    def __init__(self, uri: str) -> None:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"invalid S3 archive URI {uri!r}; expected s3://bucket/prefix")
        try:
            boto3 = importlib.import_module("boto3")
        except ImportError as error:
            raise RuntimeError("S3 archival requires the optional 'boto3' dependency") from error
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        self.client = boto3.client("s3")

    def _key(self, key: str) -> str:
        return "/".join(part for part in (self.prefix, key.lstrip("/")) if part)

    def _retry(self, operation: Any) -> Any:
        last: Exception | None = None
        for attempt in range(4):
            try:
                return operation()
            except Exception as error:
                last = error
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("unreachable S3 retry state") from last

    def put_file(self, source: Path, key: str) -> None:
        self._retry(lambda: self.client.upload_file(str(source), self.bucket, self._key(key)))

    def put_bytes(self, payload: bytes, key: str) -> None:
        self._retry(lambda: self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=payload))

    def get_file(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._retry(lambda: self.client.download_file(self.bucket, self._key(key), str(destination)))

    def get_bytes(self, key: str) -> bytes:
        response = self._retry(lambda: self.client.get_object(Bucket=self.bucket, Key=self._key(key)))
        return bytes(response["Body"].read())

    def delete(self, key: str) -> None:
        self._retry(lambda: self.client.delete_object(Bucket=self.bucket, Key=self._key(key)))

    def list_keys(self, prefix: str) -> list[str]:
        remote_prefix = self._key(prefix)
        paginator = self.client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in self._retry(lambda: list(paginator.paginate(Bucket=self.bucket, Prefix=remote_prefix))):
            for item in page.get("Contents", []):
                key = str(item["Key"])
                if self.prefix and key.startswith(f"{self.prefix}/"):
                    key = key[len(self.prefix) + 1 :]
                keys.append(key)
        return sorted(keys)


def object_store(uri: str) -> ObjectStore:
    if uri.startswith("s3://"):
        return S3ObjectStore(uri)
    if uri.startswith("file://"):
        parsed = urlparse(uri)
        if parsed.netloc not in ("", "localhost"):
            raise ValueError(f"non-local file archive URI {uri!r} is unsupported")
        return LocalObjectStore(Path(unquote(parsed.path)))
    raise ValueError(f"unsupported archive URI {uri!r}; expected s3:// or file://")


def _snapshot_files(run_dir: Path, library_dir: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    roots = (("run", run_dir), ("library", library_dir))
    resolved_run = run_dir.resolve()
    resolved_library = library_dir.resolve()
    for label, root in roots:
        if label == "library" and (resolved_library == resolved_run or resolved_run in resolved_library.parents):
            continue
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink() or path.name == ACTIVE_LOCK_NAME or path.name.endswith((".tmp", ".part")):
                continue
            if label == "run" and path.relative_to(root).as_posix() == "archive_state.json":
                continue
            files.append((str(PurePosixPath(label) / path.relative_to(root)), path))
    return files


def build_snapshot(run_dir: Path, library_dir: Path, destination: Path, *, snapshot_id: str, task_cursor: int, status: str) -> dict[str, Any]:
    """Build one immutable payload and its per-file manifest while the caller holds a task boundary."""

    files = _snapshot_files(run_dir, library_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz", compresslevel=6) as archive:
        for name, path in files:
            archive.add(path, arcname=name, recursive=False)
    # Derive member metadata from the immutable payload, not the mutable sources. A file changing
    # during archive.add can never publish a manifest that describes different bytes than the tar.
    entries: list[dict[str, Any]] = []
    with tarfile.open(destination, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"snapshot member is unreadable: {member.name}")
            digest = hashlib.sha256()
            size = 0
            for block in iter(lambda: extracted.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
            if size != member.size:
                raise ValueError(f"snapshot member changed while archived: {member.name}")
            entries.append({"path": member.name, "size": size, "sha256": digest.hexdigest()})
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "created_unix": time.time(),
        "task_cursor": int(task_cursor),
        "status": status,
        "run_name": run_dir.name,
        "library_name": library_dir.name,
        "payload": "snapshot.tar.gz",
        "payload_size": destination.stat().st_size,
        "payload_sha256": _sha256(destination),
        "files": entries,
    }


def build_content_snapshot(
    run_dir: Path,
    library_dir: Path,
    destination: Path,
    store: ObjectStore,
    *,
    snapshot_id: str,
    task_cursor: int,
    status: str,
) -> dict[str, Any]:
    """Publish immutable content objects and build a small snapshot envelope."""

    entries: list[dict[str, Any]] = []
    uploaded = reused = 0
    staging = destination.parent / "objects"
    staging.mkdir(parents=True, exist_ok=True)
    for ordinal, (name, source) in enumerate(_snapshot_files(run_dir, library_dir)):
        stable = staging / f"{ordinal:08d}"
        shutil.copyfile(source, stable)
        digest = _sha256(stable)
        size = stable.stat().st_size
        object_key = f"objects/sha256/{digest[:2]}/{digest}"
        if store.list_keys(object_key):
            reused += 1
        else:
            store.put_file(stable, object_key)
            uploaded += 1
        entries.append({"path": name, "size": size, "sha256": digest, "object": object_key})
    envelope = {
        "schema_version": CONTENT_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "objects": [{"path": entry["path"], "size": entry["size"], "sha256": entry["sha256"], "object": entry["object"]} for entry in entries],
    }
    destination.write_bytes((json.dumps(envelope, sort_keys=True) + "\n").encode())
    return {
        "schema_version": CONTENT_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "created_unix": time.time(),
        "task_cursor": int(task_cursor),
        "status": status,
        "run_name": run_dir.name,
        "library_name": library_dir.name,
        "payload": "snapshot.tar.gz",
        "payload_size": destination.stat().st_size,
        "payload_sha256": _sha256(destination),
        "files": entries,
        "objects_uploaded": uploaded,
        "objects_reused": reused,
    }


@dataclass
class ArchiveManager:
    store: ObjectStore
    run_dir: Path
    library_dir: Path
    snapshot_every_tasks: int = 1
    retain_local_snapshots: int = 0
    run_key: str = ""

    def __post_init__(self) -> None:
        if not self.run_key:
            self.run_key = self.run_dir.name
        if not _valid_run_key(self.run_key):
            raise ValueError(f"invalid archive run key {self.run_key!r}; use letters, digits, '.', '_', or '-'")

    @classmethod
    def from_config(cls, config: dict[str, Any], run_dir: Path, library_dir: Path) -> "ArchiveManager | None":
        table = dict(config.get("archive", {}) or {})
        if not bool(table.get("enabled", False)):
            return None
        uri = str(table.get("uri", ""))
        uri_env = str(table.get("uri_env", ""))
        if not uri and uri_env:
            uri = os.environ.get(uri_env, "")
        if not uri:
            location = f" environment variable {uri_env}" if uri_env else " [archive] uri"
            raise RuntimeError(f"external archival is enabled but{location} is empty")
        backend = str(table.get("backend", urlparse(uri).scheme))
        if backend not in ("file", "s3"):
            raise ValueError(f"unsupported [archive] backend {backend!r}; expected 'file' or 's3'")
        if backend == "s3" and not uri.startswith("s3://"):
            raise ValueError("[archive] backend='s3' requires an s3:// URI")
        if backend == "file" and not uri.startswith("file://"):
            raise ValueError("[archive] backend='file' requires a file:// URI")
        configured_key = str(table.get("run_key", "")).strip()
        identity = str(config.get("config_effective_sha256") or config.get("config_sha256") or "unhashed")[:12]
        run_key = configured_key or _persistent_run_key(Path(run_dir), seed=int(config.get("seed", 0)), identity=identity)
        if not _valid_run_key(run_key):
            raise ValueError(f"invalid archive run key {run_key!r}; use letters, digits, '.', '_', or '-'")
        return cls(
            store=object_store(uri),
            run_dir=Path(run_dir),
            library_dir=Path(library_dir),
            snapshot_every_tasks=max(1, int(table.get("snapshot_every_tasks", 1))),
            retain_local_snapshots=max(0, int(table.get("retain_local_snapshots", 0))),
            run_key=run_key,
        )

    @property
    def state_path(self) -> Path:
        return self.run_dir / "archive_state.json"

    @property
    def remote_prefix(self) -> str:
        return f"runs/{self.run_key}"

    def due(self, task_cursor: int) -> bool:
        return task_cursor > 0 and task_cursor % self.snapshot_every_tasks == 0

    def snapshot(self, task_cursor: int, *, status: str = "running") -> dict[str, Any]:
        base_id = f"task-{int(task_cursor):06d}"
        snapshot_id = f"{base_id}-{_status_slug(status)}-{time.time_ns()}"
        state = {
            "schema_version": SCHEMA_VERSION,
            "run_key": self.run_key,
            "snapshot_id": snapshot_id,
            "status": "uploading",
            "snapshot_status": status,
            "task_cursor": int(task_cursor),
        }
        _atomic_json(self.state_path, state)
        with tempfile.TemporaryDirectory(prefix="versal_snapshot_") as temporary:
            payload = Path(temporary) / "snapshot.tar.gz"
            prefix = f"{self.remote_prefix}/snapshots/{snapshot_id}"
            incomplete = json.dumps({"snapshot_id": snapshot_id, "created_unix": time.time()}, sort_keys=True).encode("utf-8")
            self.store.put_bytes(incomplete, f"{prefix}/INCOMPLETE.json")
            manifest = build_content_snapshot(
                self.run_dir,
                self.library_dir,
                payload,
                self.store,
                snapshot_id=snapshot_id,
                task_cursor=task_cursor,
                status=status,
            )
            manifest["run_key"] = self.run_key
            self.store.put_file(payload, f"{prefix}/snapshot.tar.gz")
            self.store.put_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"), f"{prefix}/manifest.json")
            self.store.delete(f"{prefix}/INCOMPLETE.json")
            latest = {"schema_version": SCHEMA_VERSION, "run_key": self.run_key, "snapshot_id": snapshot_id, "manifest": f"{prefix}/manifest.json"}
            self.store.put_bytes((json.dumps(latest, indent=2, sort_keys=True) + "\n").encode("utf-8"), f"{self.remote_prefix}/latest.json")
            catalog = {
                "schema_version": SCHEMA_VERSION,
                "run_key": self.run_key,
                "run_name": self.run_dir.name,
                "library_name": self.library_dir.name,
                "latest_snapshot_id": snapshot_id,
                "latest_status": status,
                "latest_task_cursor": int(task_cursor),
                "updated_unix": time.time(),
            }
            self.store.put_bytes((json.dumps(catalog, indent=2, sort_keys=True) + "\n").encode("utf-8"), f"{self.remote_prefix}/run.json")
            self._retain_local(payload, manifest)
        _atomic_json(self.state_path, {**state, "status": "complete", "payload_sha256": manifest["payload_sha256"]})
        return manifest

    def _retain_local(self, payload: Path, manifest: dict[str, Any]) -> None:
        if self.retain_local_snapshots <= 0:
            return
        root = self.run_dir.parent / ".archive_snapshots" / self.run_key
        root.mkdir(parents=True, exist_ok=True)
        snapshot_id = str(manifest["snapshot_id"])
        shutil.copyfile(payload, root / f"{snapshot_id}.tar.gz")
        _atomic_json(root / f"{snapshot_id}.json", manifest)
        payloads = sorted(root.glob("task-*.tar.gz"), key=_snapshot_sort_key)
        for stale in payloads[: -self.retain_local_snapshots]:
            stale.unlink(missing_ok=True)
            stale.with_suffix("").with_suffix(".json").unlink(missing_ok=True)


def _valid_run_key(value: str) -> bool:
    return bool(value) and len(value) <= 160 and all(character.isalnum() or character in "._-" for character in value)


def _persistent_run_key(run_dir: Path, *, seed: int, identity: str) -> str:
    """Mint one collision-resistant namespace once; a restored/resumed run reuses the file."""
    path = run_dir / "archive_run_key.txt"
    if path.exists():
        return path.read_text().strip()
    run_dir.mkdir(parents=True, exist_ok=True)
    value = f"{run_dir.name}-s{seed}-{identity}-{uuid.uuid4().hex[:10]}"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return path.read_text().strip()
    with os.fdopen(descriptor, "w") as handle:
        handle.write(f"{value}\n")
    return value


def _valid_snapshot_id(value: str) -> bool:
    return bool(value) and len(value) <= 200 and all(character.isalnum() or character in "._-" for character in value)


def _status_slug(status: str) -> str:
    slug = "-".join(part for part in "".join(character.lower() if character.isalnum() else " " for character in status).split() if part)
    return slug[:24] or "snapshot"


def _snapshot_sort_key(path: Path) -> tuple[int, str]:
    """Order retained payloads by the nanosecond suffix, independent of status or filesystem mtime."""

    try:
        timestamp = int(path.name.removesuffix(".tar.gz").rsplit("-", 1)[-1])
    except ValueError:
        timestamp = path.stat().st_mtime_ns
    return timestamp, path.name


def list_runs(uri: str) -> list[dict[str, Any]]:
    """Return one catalog record per archived run under a shared object-store prefix."""
    store = object_store(uri)
    records: list[dict[str, Any]] = []
    for key in store.list_keys("runs"):
        parts = PurePosixPath(key).parts
        if len(parts) == 3 and parts[0] == "runs" and parts[2] == "run.json":
            records.append(json.loads(store.get_bytes(key)))
    return sorted(records, key=lambda item: str(item.get("run_key", "")))


def _manifest(store: ObjectStore, run_key: str, snapshot_id: str | None) -> dict[str, Any]:
    if not _valid_run_key(run_key):
        raise ValueError(f"invalid archive run key {run_key!r}")
    prefix = f"runs/{run_key}"
    if snapshot_id is None or snapshot_id == "latest":
        latest = json.loads(store.get_bytes(f"{prefix}/latest.json"))
        snapshot_id = str(latest["snapshot_id"])
    if not _valid_snapshot_id(snapshot_id):
        raise ValueError(f"invalid archive snapshot id {snapshot_id!r}")
    manifest = json.loads(store.get_bytes(f"{prefix}/snapshots/{snapshot_id}/manifest.json"))
    if str(manifest.get("snapshot_id", "")) != snapshot_id:
        raise ValueError(f"snapshot manifest ID does not match requested snapshot {snapshot_id!r}")
    if manifest.get("run_key") not in (None, run_key):
        raise ValueError(f"snapshot manifest belongs to run {manifest.get('run_key')!r}, not {run_key!r}")
    return manifest


def _install_restore(extracted: Path, destination: Path) -> None:
    """Copy beside the destination, then atomically rename a complete verified tree into place."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.restore-", dir=destination.parent))
    try:
        for child in extracted.iterdir():
            target = staging / child.name
            if child.is_dir():
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)
        if destination.exists():
            if not destination.is_dir() or any(destination.iterdir()):
                raise FileExistsError(f"restore destination is not empty: {destination}")
            destination.rmdir()
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def restore_snapshot(uri: str, destination: Path, *, run_key: str, snapshot_id: str | None = None, verify_only: bool = False) -> dict[str, Any]:
    destination = Path(destination)
    if not verify_only and destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"restore destination is not empty: {destination}")
    store = object_store(uri)
    manifest = _manifest(store, run_key, snapshot_id)
    schema_version = int(manifest.get("schema_version", 0))
    if schema_version not in (SCHEMA_VERSION, CONTENT_SCHEMA_VERSION) or manifest.get("snapshot_id") is None:
        raise ValueError("invalid or unsupported snapshot manifest")
    if manifest.get("run_key") not in (None, run_key):
        raise ValueError(f"snapshot manifest belongs to run {manifest.get('run_key')!r}, not {run_key!r}")
    snapshot_id = str(manifest["snapshot_id"])
    with tempfile.TemporaryDirectory(prefix="versal_restore_") as temporary:
        payload = Path(temporary) / "snapshot.tar.gz"
        store.get_file(f"runs/{run_key}/snapshots/{snapshot_id}/snapshot.tar.gz", payload)
        actual_payload_hash = _sha256(payload)
        if actual_payload_hash != manifest["payload_sha256"]:
            raise ValueError(f"snapshot payload hash mismatch: expected {manifest['payload_sha256']}, got {actual_payload_hash}")
        if payload.stat().st_size != int(manifest["payload_size"]):
            raise ValueError("snapshot payload size mismatch")
        extracted = Path(temporary) / "extracted"
        extracted.mkdir(parents=True, exist_ok=True)
        if schema_version == SCHEMA_VERSION:
            with tarfile.open(payload, "r:gz") as archive:
                archive.extractall(extracted, filter="data")
        else:
            envelope = json.loads(payload.read_text())
            envelope_files = {(str(item["path"]), str(item["sha256"]), int(item["size"]), str(item["object"])) for item in envelope.get("objects", [])}
            manifest_files = {(str(item["path"]), str(item["sha256"]), int(item["size"]), str(item["object"])) for item in manifest["files"]}
            if envelope.get("snapshot_id") != snapshot_id or envelope_files != manifest_files:
                raise ValueError("content-addressed snapshot envelope does not match manifest")
            for entry in manifest["files"]:
                store.get_file(str(entry["object"]), extracted / PurePosixPath(entry["path"]))
        expected_paths = {str(entry["path"]) for entry in manifest["files"]}
        actual_paths = {path.relative_to(extracted).as_posix() for path in extracted.rglob("*") if path.is_file()}
        if actual_paths != expected_paths:
            raise ValueError("snapshot member list does not match manifest")
        for entry in manifest["files"]:
            path = extracted / PurePosixPath(entry["path"])
            if not path.is_file() or path.stat().st_size != int(entry["size"]) or _sha256(path) != entry["sha256"]:
                raise ValueError(f"snapshot member verification failed: {entry['path']}")
        if not verify_only:
            if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
                raise FileExistsError(f"restore destination is not empty: {destination}")
            _install_restore(extracted, destination)
    return manifest
