"""library_gc: physically delete retired library entries nothing references anymore.

    uv run library_gc --dry-run          # inspect what WOULD go
    uv run library_gc                    # sweep, then prune dead router vertices
    uv run library_gc --library other/   # a different library dir

Mark-and-sweep from every live entry (plus protected keys) through payload refs, so a retired
dependency a live composition still names survives, and a whole superseded chain falls together.
By default the NEWEST results/*/checkpoint.json is scanned and any macro refs inside its pooled
genomes are protected, so resuming that run cannot dangle; --no-protect-checkpoint opts out.
After the sweep, persisted router state is reloaded tolerantly (dropping vertices whose entries
are gone, with their state rows and traffic) and re-saved pruned.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from ardevo.library import MODULE, ModuleLibrary, payload_refs
from ardevo.utils.logging import Logger

console = Logger.get_console()


def checkpoint_macro_refs(checkpoint_path: Path) -> set[str]:
    """Library keys named by macro refs inside a resumable checkpoint's pooled genomes (modules +
    species champions). These are the only checkpoint contents that resolve against the library on
    resume; absorbed_keys is a dedup guard and attempts are informational."""
    try:
        loop_state = json.loads(checkpoint_path.read_text()).get("loop_state", {})
    except (OSError, json.JSONDecodeError):
        return set()
    refs: set[str] = set()
    for module in loop_state.get("modules", []):
        refs |= payload_refs(MODULE, module.get("genome", {}))
    for genome in (loop_state.get("species_champions") or {}).values():
        refs |= payload_refs(MODULE, genome)
    return refs


def newest_checkpoint(results_root: Path) -> Path | None:
    candidates = sorted(results_root.glob("*/checkpoint.json"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def prune_router(library: ModuleLibrary, router_dir: Path) -> int:
    """Tolerantly reload the persisted router (dropping vertices whose entries are gone) and re-save
    the pruned state. Returns the number of vertices pruned."""
    from ardevo.routing import RouterService

    meta = json.loads((router_dir / "router_meta.json").read_text())
    before = len(meta.get("vertex_keys", []))
    service = RouterService(
        library,
        d_model=int(meta["d_model"]),
        top_k=int(meta["top_k"]),
        max_steps=int(meta["max_steps"]),
        adapter_rank=int(meta.get("adapter_rank", 0)),
        halting=bool(meta.get("halting", False)),
        edge_bias=bool(meta.get("edge_bias", False)),
        persist_dir=router_dir,
    )
    pruned = before - len(service.net._vertex_order)
    if pruned:
        service.save()
    return pruned


def run_gc(library_root: Path, *, dry_run: bool, protect_checkpoint: bool, results_root: Path | None = None) -> dict[str, Any]:
    library = ModuleLibrary(library_root)
    protect: set[str] = set()
    checkpoint_path = newest_checkpoint(results_root if results_root is not None else Path("results")) if protect_checkpoint else None
    if checkpoint_path is not None:
        protect = checkpoint_macro_refs(checkpoint_path)
    before = len(library)
    swept = library.collect_garbage(protect=protect, dry_run=dry_run)
    router_dir = library_root / "router"
    router_pruned = 0
    if not dry_run and swept and (router_dir / "router_meta.json").exists():
        router_pruned = prune_router(library, router_dir)
    return {
        "before": before,
        "swept": swept,
        "kept": before - len(swept),
        "router_pruned": router_pruned,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "protected": sorted(protect),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete retired library entries nothing references (mark-and-sweep; prunes dead router vertices).")
    parser.add_argument("--library", default="library", help="library dir to sweep")
    parser.add_argument("--dry-run", action="store_true", help="report what would be deleted without touching anything")
    parser.add_argument("--no-protect-checkpoint", action="store_true", help="do NOT protect macro refs from the newest results/*/checkpoint.json")
    args = parser.parse_args()

    library_root = Path(args.library)
    if not (library_root / "index.json").exists():
        console.print(f"[red]no library at {library_root}[/red]")
        return
    report = run_gc(library_root, dry_run=args.dry_run, protect_checkpoint=not args.no_protect_checkpoint)

    from rich.table import Table

    verb = "would delete" if args.dry_run else "deleted"
    table = Table(title=f"library_gc: {library_root} | {report['before']} entries -> {verb} {len(report['swept'])} | router vertices pruned: {report['router_pruned']}")
    table.add_column("swept key")
    for key in report["swept"]:
        table.add_row(key)
    console.print(table)
    if report["checkpoint"]:
        console.print(f"protected {len(report['protected'])} checkpoint macro refs from {report['checkpoint']}")


if __name__ == "__main__":
    main()
