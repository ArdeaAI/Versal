"""motif_census: mine recurring substructures across library entries and report them.

    uv run motif_census                          # census + vocabulary tables, writes library/motifs.json
    uv run motif_census --render                 # also draw the atlas to library/images/motifs.png
    uv run motif_census --sizes 3,4,5 --top 5    # deeper mining, tighter tables
    uv run motif_census --include-retired        # tombstones are discovered structure too

Motifs are exact canonical labeled subgraphs counted by SUPPORT (distinct entries containing them),
grouped by diversity class so identity chains cannot drown the sections where a spontaneously
evolved skip/gating/attention-like discovery would land. The JSON report carries each motif's
canonical nodes/edges verbatim, so the paper can reconstruct any motif from the file alone.
"""

import argparse
import json
from pathlib import Path

from ardevo.library import ModuleLibrary
from ardevo.motifs import MAX_MOTIF_SIZE, CensusReport, MotifRecord, motif_census, report_to_dict
from ardevo.utils.logging import Logger

console = Logger.get_console()


def run_census(
    library_root: Path,
    *,
    sizes: tuple[int, ...] = (3, 4),
    min_support: int = 2,
    include_retired: bool = False,
    per_entry_cap: int = 50_000,
    json_out: Path | None = None,
) -> CensusReport:
    """The testable core: mine the library and (optionally) write the JSON report."""
    library = ModuleLibrary(library_root)
    report = motif_census(library, sizes=sizes, min_support=min_support, include_retired=include_retired, per_entry_cap=per_entry_cap)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report_to_dict(report), indent=2))
    return report


def _top_per_class(records: list[MotifRecord], top: int) -> list[MotifRecord]:
    kept: list[MotifRecord] = []
    counts: dict[str, int] = {}
    for record in records:  # records arrive class-grouped and ranked within class
        counts[record.diversity_class] = counts.get(record.diversity_class, 0) + 1
        if counts[record.diversity_class] <= top:
            kept.append(record)
    return kept


def _parse_sizes(text: str) -> tuple[int, ...]:
    sizes = tuple(int(part) for part in text.split(",") if part.strip())
    bad = [size for size in sizes if size < 2 or size > MAX_MOTIF_SIZE]
    if bad or not sizes:
        raise argparse.ArgumentTypeError(f"sizes must be in 2..{MAX_MOTIF_SIZE} (the canonicalizer is exact brute force), got {text!r}")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine recurring substructures (canonical motifs) and entry reuse across the library.")
    parser.add_argument("--library", default="library", help="library dir to mine")
    parser.add_argument("--sizes", type=_parse_sizes, default=(3, 4), help="comma-separated module motif sizes, each in 2..5 (default 3,4)")
    parser.add_argument("--min-support", type=int, default=2, help="a motif must recur across at least this many DISTINCT entries (default 2)")
    parser.add_argument("--top", type=int, default=10, help="rows per diversity class per table (default 10)")
    parser.add_argument("--include-retired", action=argparse.BooleanOptionalAction, default=False, help="mine tombstoned entries too")
    parser.add_argument("--per-entry-cap", type=int, default=50_000, help="max enumerated subgraphs per entry per size (deterministic truncation)")
    parser.add_argument("--json-out", default=None, help="report path (default <library>/motifs.json)")
    parser.add_argument("--render", action="store_true", help="also draw the motif atlas to <library>/images/motifs.png")
    args = parser.parse_args()

    library_root = Path(args.library)
    if not (library_root / "index.json").exists():
        console.print(f"[red]no library at {library_root}[/red]")
        return
    json_out = Path(args.json_out) if args.json_out else library_root / "motifs.json"
    report = run_census(
        library_root, sizes=args.sizes, min_support=args.min_support, include_retired=args.include_retired, per_entry_cap=args.per_entry_cap, json_out=json_out
    )

    from rich.table import Table

    def motif_table(title: str, records: list[MotifRecord]) -> Table:
        table = Table(title=title)
        for column in ("class", "fingerprint", "k", "support", "occurrences", "exemplars", "description"):
            table.add_column(column)
        for record in _top_per_class(records, args.top):
            table.add_row(
                record.diversity_class, record.fingerprint, str(record.size), str(record.support), str(record.occurrences), " ".join(record.exemplars), record.description
            )
        return table

    scanned = report.entries_scanned
    console.print(motif_table(f"module motifs ({scanned['modules']} entries, sizes {list(args.sizes)}, min support {args.min_support})", report.module_motifs))
    console.print(motif_table(f"composition motifs ({scanned['compositions']} entries)", report.composition_motifs))

    vocabulary = Table(title="vocabulary: who is built FROM whom")
    for column in ("key", "type", "level", "referenced by", "refs", "use_count", "max_fitness", "flags"):
        vocabulary.add_column(column)
    for row in report.vocabulary:
        flags = " ".join(flag for flag, on in (("retired", row["retired"]), ("dependency", row["dependency"])) if on)
        fitness = f"{row['max_attributed_fitness']:.2f}"
        vocabulary.add_row(row["key"], row["entry_type"], str(row["level"]), " ".join(row["referenced_by"]), str(row["reference_count"]), str(row["use_count"]), fitness, flags)
    console.print(vocabulary)

    if report.truncated_entries:
        console.print(f"[yellow]capped enumeration (supports are still honest, occurrences undercount): {report.truncated_entries}[/yellow]")
    console.print(f"report written to {json_out}")

    if args.render:
        from ardevo.rendering import render_motif_atlas

        atlas_records = _top_per_class(report.module_motifs, args.top) + _top_per_class(report.composition_motifs, args.top)
        atlas_path = render_motif_atlas(library_root / "images" / "motifs.png", atlas_records)
        console.print(f"atlas written to {atlas_path}")


if __name__ == "__main__":
    main()
