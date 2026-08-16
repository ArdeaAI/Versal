"""Create, list, verify, and restore coherent external experiment snapshots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from versal.external_archive import ArchiveManager, ExperimentLock, list_runs, restore_snapshot


def _resolve_uri(uri: str | None, uri_env: str, parser: argparse.ArgumentParser) -> str:
    resolved = uri or os.environ.get(uri_env, "")
    if not resolved:
        parser.error(f"archive URI is missing; pass --uri or set {uri_env}")
    return resolved


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="External Versal experiment archive maintenance.")
    parser.add_argument("--uri", default=None, help="file:///... or s3://bucket/prefix (otherwise read --uri-env).")
    parser.add_argument("--uri-env", default="VERSAL_ARCHIVE_URI", help="Environment variable used when --uri is absent.")
    commands = parser.add_subparsers(dest="command", required=True)

    snapshot_parser = commands.add_parser("snapshot", help="Upload a coherent run + library snapshot.")
    snapshot_parser.add_argument("--run-dir", type=Path, required=True)
    snapshot_parser.add_argument("--library-dir", type=Path, required=True)
    snapshot_parser.add_argument("--run-key", required=True)
    snapshot_parser.add_argument("--task-cursor", type=int, required=True)
    snapshot_parser.add_argument("--status", default="manual")
    snapshot_parser.add_argument("--retain-local", type=int, default=0)

    commands.add_parser("list", help="List run namespaces stored under this URI.")

    verify_parser = commands.add_parser("verify", help="Download and verify a snapshot without restoring it.")
    verify_parser.add_argument("--run-key", required=True)
    verify_parser.add_argument("--snapshot", default="latest")

    restore_parser = commands.add_parser("restore", help="Verify and restore a snapshot into an empty directory.")
    restore_parser.add_argument("--run-key", required=True)
    restore_parser.add_argument("--snapshot", default="latest")
    restore_parser.add_argument("--destination", type=Path, required=True)

    args = parser.parse_args(argv)
    uri = _resolve_uri(args.uri, args.uri_env, parser)
    if args.command == "list":
        _print(list_runs(uri))
        return
    if args.command == "snapshot":
        backend = urlparse(uri).scheme
        manager = ArchiveManager.from_config(
            {
                "archive": {"enabled": True, "uri": uri, "backend": backend, "run_key": args.run_key, "retain_local_snapshots": args.retain_local},
                "seed": 0,
            },
            args.run_dir,
            args.library_dir,
        )
        assert manager is not None
        lock = ExperimentLock(args.run_dir, args.library_dir)
        lock.acquire()
        try:
            _print(manager.snapshot(args.task_cursor, status=args.status))
        finally:
            lock.release()
        return
    manifest = restore_snapshot(
        uri,
        args.destination if args.command == "restore" else Path("."),
        run_key=args.run_key,
        snapshot_id=args.snapshot,
        verify_only=args.command == "verify",
    )
    _print(manifest)


if __name__ == "__main__":
    main()
