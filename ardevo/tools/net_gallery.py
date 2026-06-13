"""render: re-render every library network after the fact, one full-size PNG per entry.

    uv run render                                  # one <key>.png per entry -> library/images/
    uv run render --images renders/ --gallery      # custom dir, plus a single contact-sheet PNG

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
