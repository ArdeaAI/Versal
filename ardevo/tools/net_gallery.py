"""net_gallery: re-render every library network after the fact, plus one contact-sheet PNG.

    uv run python -m ardevo.tools.net_gallery --library library --out library/gallery.png
    uv run python -m ardevo.tools.net_gallery --library library --per-entry renders/

The gallery is the single-file artistic overview of everything on the shelf; `--per-entry` writes
one full-size `<key>.png` per entry. Both go through `ardevo.rendering`, so nested networks expand
inline and a broken entry degrades to a labeled box instead of killing the sheet.
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
    parser = argparse.ArgumentParser(description="Render the library: one gallery PNG, optionally one PNG per entry.")
    parser.add_argument("--library", default="library", help="library dir to render")
    parser.add_argument("--out", default=None, help="gallery output path (default: <library>/gallery.png)")
    parser.add_argument("--per-entry", default=None, help="also write one <key>.png per entry into this directory")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--include-retired", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-dependencies", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    library_root = Path(args.library)
    if not library_root.exists():
        console.print(f"[red]no library at {library_root}[/red] (point --library at a library dir)")
        return
    library = ModuleLibrary(library_root)

    out_path = Path(args.out) if args.out else library_root / "gallery.png"
    gallery_path = render_library_gallery(library, out_path, columns=args.columns, include_retired=args.include_retired, include_dependencies=args.include_dependencies)

    rows: list[dict[str, Any]] = []
    if args.per_entry:
        rows = render_all_entries(library, Path(args.per_entry), include_retired=args.include_retired, include_dependencies=args.include_dependencies)

    from rich.table import Table

    table = Table(title=f"net gallery: {library_root} ({len(library)} entries) -> {gallery_path}")
    columns = ["key", "entry_type", "level", "status", "path"]
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*(str(row.get(column, "")) for column in columns))
    console.print(table)


if __name__ == "__main__":
    main()
