"""render: re-render every library network after the fact, one full-size PNG per entry.

    uv run render                                  # one <key>.png per entry -> library/images/
    uv run render --images renders/ --gallery      # custom dir, plus a single contact-sheet PNG
    uv run render --overmind                        # also (re)render the whole routed model -> library/images/overmind.png

Everything goes through `ardevo.rendering`, so nested networks expand into callout boxes across the
top of each image and a broken entry degrades to a labeled box instead of killing the batch.
"""

import argparse
from pathlib import Path
from typing import Any

from ardevo.library import ModuleLibrary
from ardevo.rendering import DEFAULT_NODE_BUDGET, render_entry, render_library_gallery
from ardevo.utils.logging import Logger

console = Logger.get_console()


def render_all_entries(
    library: ModuleLibrary,
    directory: Path,
    *,
    include_retired: bool = False,
    include_dependencies: bool = True,
    node_budget: int = DEFAULT_NODE_BUDGET,
) -> list[dict[str, Any]]:
    """One full-size PNG per selected entry into `directory`; one row per entry, never raises."""
    rows: list[dict[str, Any]] = []
    for summary in library.summaries(include_retired=include_retired, include_dependencies=include_dependencies):
        key = summary["key"]
        row: dict[str, Any] = {"key": key, "entry_type": summary["entry_type"], "level": summary["level"]}
        try:
            path = render_entry(directory / f"{key}.png", library.load(key), library=library, node_budget=node_budget)
            row.update(status="OK", path=str(path))
        except Exception as error:
            row.update(status=f"FAIL:{type(error).__name__}", path="")
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the library: one PNG per entry, optionally a gallery contact sheet.")
    parser.add_argument("--library", default="library", help="library dir to render")
    parser.add_argument("--images", default=None, help="per-entry output dir (default: <library>/images)")
    parser.add_argument("--gallery", nargs="?", const="__default__", default=None, help="also write a contact-sheet PNG (default path: <library>/gallery.png)")
    parser.add_argument("--overmind", action="store_true", help="also (re)render the whole routed model to <library>/images/overmind.png from <library>/router state")
    parser.add_argument("--config", default=None, help="run config whose [orchestrator.routed] shapes a COLD overmind portrait when no router state exists")
    parser.add_argument("--columns", type=int, default=4, help="gallery columns")
    parser.add_argument("--include-retired", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-dependencies", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    library_root = Path(args.library)
    if not library_root.exists():
        console.print(f"[red]no library at {library_root}[/red] (point --library at a library dir)")
        return
    library = ModuleLibrary(library_root)

    images_dir = Path(args.images) if args.images else library_root / "images"
    rows = render_all_entries(library, images_dir, include_retired=args.include_retired, include_dependencies=args.include_dependencies)

    gallery_note = ""
    if args.gallery is not None:
        gallery_path = library_root / "gallery.png" if args.gallery == "__default__" else Path(args.gallery)
        render_library_gallery(library, gallery_path, columns=args.columns, include_retired=args.include_retired, include_dependencies=args.include_dependencies)
        gallery_note = f" | gallery -> {gallery_path}"

    if args.overmind:
        router_dir = library_root / "router"
        from ardevo.routing import RouterService

        if (router_dir / "router_meta.json").exists():
            import json

            meta = json.loads((router_dir / "router_meta.json").read_text())
            service = RouterService(
                library,
                d_model=int(meta["d_model"]),
                top_k=int(meta["top_k"]),
                max_steps=int(meta["max_steps"]),
                adapter_rank=int(meta.get("adapter_rank", 0)),
                halting=bool(meta.get("halting", False)),
                edge_bias=bool(meta.get("edge_bias", False)),
                persist_dir=router_dir,
                image_dir=images_dir,
            )
            service._rendered_vertex_count = -1  # force a render even if the count is unchanged
            service.render_overmind()
            gallery_note += f" | overmind -> {images_dir / 'overmind.png'}"
        else:
            # No persisted state: the router only saves on a real win, so mid-campaign libraries
            # often have vertices but no state file. Render a COLD portrait instead: the routable
            # substrate as it stands (every eligible entry a vertex, uniform feeds, no traffic).
            from ardevo.utils.config import Config

            routed = Config(conf_path=args.config).current.get("orchestrator", {}).get("routed", {})
            service = RouterService(
                library,
                d_model=int(routed.get("d_model", 64)),
                top_k=int(routed.get("top_k", 2)),
                max_steps=int(routed.get("max_steps", 4)),
                adapter_rank=int(routed.get("adapter_rank", 0)),
                halting=bool(routed.get("halting", False)),
                edge_bias=bool(routed.get("edge_bias", False)),
                persist_dir=None,  # read-only: a portrait must never mint router state
                image_dir=images_dir,
            )
            service.sync(include_compositions=bool(routed.get("include_compositions", True)), exclude_temporal=bool(routed.get("exclude_temporal", True)))
            if service.net._vertex_order:
                service.render_overmind()
                console.print(f"[yellow]no router state at {router_dir}[/yellow]: rendered a COLD portrait (no trained gate, uniform feeds)")
                gallery_note += f" | overmind (cold) -> {images_dir / 'overmind.png'}"
            else:
                console.print(f"[yellow]no router state at {router_dir}[/yellow] and no routable entries in the library (nothing to portrait)")

    from rich.table import Table

    table = Table(title=f"render: {library_root} ({len(library)} entries) -> {images_dir}{gallery_note}")
    columns = ["key", "entry_type", "level", "status", "path"]
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*(str(row.get(column, "")) for column in columns))
    console.print(table)


if __name__ == "__main__":
    main()
