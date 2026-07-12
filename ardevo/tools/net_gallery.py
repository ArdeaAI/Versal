"""render: render library networks and the routed overmind after the fact.

    uv run render                                  # one <key>.png per entry -> library/images/
    uv run render --images renders/ --gallery      # custom dir, plus a single contact-sheet PNG
    uv run render --overmind                       # also (re)render the routed model
    uv run render --config configs/preflight.toml --metadata-overmind --images /tmp/ardevo-overmind-preview
    uv run render --config configs/preflight.toml --overmind-only --cold-overmind --images /tmp/ardevo-overmind-preview

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


def resolve_library_root(library: str | None, config: str | None) -> tuple[Path, dict[str, Any] | None]:
    """Resolve the explicit library, then the config library, then the canonical default."""
    runtime = None
    if config is not None:
        from ardevo.utils.config import Config

        runtime = Config(conf_path=config).current
    configured = (runtime or {}).get("orchestrator", {}).get("library_dir")
    return Path(library or configured or "library"), runtime


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


def _metadata_pathways(ordered_names: list[str], transitions: dict[str, dict[str, float]], *, per_vertex: int = 3, floor: float = 1e-3) -> list[tuple[int, int, float]]:
    """Normalize persisted transition traffic using the live router's portrait policy."""
    index_of = {name: index for index, name in enumerate(ordered_names)}
    raw = {
        (index_of[source], index_of[target]): float(weight)
        for source, targets in transitions.items()
        if source in index_of
        for target, weight in targets.items()
        if target in index_of and float(weight) > 0.0
    }
    if not raw:
        return []
    peak = max(raw.values())
    by_source: dict[int, list[tuple[int, int, float]]] = {}
    for (source, target), weight in raw.items():
        normalized = weight / peak
        if normalized >= floor:
            by_source.setdefault(source, []).append((source, target, normalized))
    return [edge for edges in by_source.values() for edge in sorted(edges, key=lambda edge: -edge[2])[:per_vertex]]


def render_overmind_from_metadata(library: ModuleLibrary, metadata_path: Path, out_path: Path) -> Path:
    """Render persisted traffic without loading the router's potentially large tensor state."""
    import json

    from ardevo.rendering import OvermindVertex, OvermindView, render_overmind
    from ardevo.routing import mean_firing_step, overmind_vertex_order, sanitize_key

    meta = json.loads(metadata_path.read_text())
    summaries = {summary["key"]: summary for summary in library.summaries(include_retired=True)}
    original_order = [str(key) for key in meta.get("vertex_keys", []) if str(key) in summaries]
    key_by_name = {sanitize_key(key): key for key in original_order}
    latent_order = list(key_by_name)
    retired = {name for name, key in key_by_name.items() if summaries[key].get("retired", False)}
    step_usage = {str(name): [float(value) for value in values] for name, values in meta.get("step_usage_totals", {}).items()}
    ordered_names = overmind_vertex_order(latent_order, retired, step_usage)

    usage = {str(name): float(value) for name, value in meta.get("usage_totals", {}).items()}
    total_usage = sum(usage.get(name, 0.0) for name in ordered_names) or 1.0
    entry_raw = {name: values[0] for name, values in step_usage.items() if values}
    exit_raw = {name: values[-1] for name, values in step_usage.items() if values}
    if not entry_raw:
        entry_raw = dict(usage)
        exit_raw = dict(usage)
    entry_peak = max(entry_raw.values(), default=0.0) or 1.0
    exit_peak = max(exit_raw.values(), default=0.0) or 1.0
    rank = {name: index for index, name in enumerate(latent_order)}

    vertices = []
    for name in ordered_names:
        key = key_by_name[name]
        is_retired = name in retired
        share = usage.get(name, 0.0) / total_usage
        vertices.append(
            OvermindVertex(
                key=key,
                label=f"{key}  ({share:.0%})" + ("  retired" if is_retired else ""),
                retired=is_retired,
                usage=share,
                entry_share=entry_raw.get(name, 0.0) / entry_peak,
                exit_share=exit_raw.get(name, 0.0) / exit_peak,
                mean_step=mean_firing_step(step_usage.get(name, [])),
                embedding_rank=rank[name],
            )
        )

    def surface_keys(field: str) -> list[str]:
        return [str(item.get("key", "")) if isinstance(item, dict) else str(item) for item in meta.get(field, [])]

    view = OvermindView(
        vertices=vertices,
        input_signatures=surface_keys("input_adapter_keys"),
        output_signatures=surface_keys("output_head_keys"),
        d_model=int(meta["d_model"]),
        top_k=int(meta["top_k"]),
        max_steps=int(meta["max_steps"]),
        pathways=_metadata_pathways(ordered_names, meta.get("transition_totals", {})),
    )
    return render_overmind(out_path, view, library=library)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the library: one PNG per entry, optionally a gallery contact sheet.")
    parser.add_argument("--library", default=None, help="library dir to render (default: [orchestrator] library_dir from --config, else library)")
    parser.add_argument("--images", default=None, help="per-entry output dir (default: <library>/images)")
    parser.add_argument("--gallery", nargs="?", const="__default__", default=None, help="also write a contact-sheet PNG (default path: <library>/gallery.png)")
    parser.add_argument("--overmind", action="store_true", help="also (re)render the whole routed model to <library>/images/overmind.png from <library>/router state")
    parser.add_argument("--overmind-only", action="store_true", help="render only overmind.png; implies --overmind and skips entry/gallery renders")
    parser.add_argument("--metadata-overmind", action="store_true", help="render only overmind.png from router metadata, without loading router_state.pt")
    parser.add_argument("--cold-overmind", action="store_true", help="ignore persisted router state and render the current library with an untrained router")
    parser.add_argument("--config", default=None, help="run config that selects the default library and shapes a cold overmind portrait")
    parser.add_argument("--columns", type=int, default=4, help="gallery columns")
    parser.add_argument("--include-retired", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-dependencies", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.overmind_only or args.metadata_overmind:
        args.overmind = True
        args.overmind_only = True
    if args.metadata_overmind and args.cold_overmind:
        parser.error("--metadata-overmind cannot be combined with --cold-overmind")
    if args.cold_overmind and not args.overmind:
        parser.error("--cold-overmind requires --overmind or --overmind-only")
    if args.overmind_only and args.gallery is not None:
        parser.error("--overmind-only cannot be combined with --gallery")

    library_root, runtime = resolve_library_root(args.library, args.config)
    if not library_root.exists():
        console.print(f"[red]no library at {library_root}[/red] (point --library at a library dir)")
        return
    library = ModuleLibrary(library_root)

    images_dir = Path(args.images) if args.images else library_root / "images"
    rows = [] if args.overmind_only else render_all_entries(library, images_dir, include_retired=args.include_retired, include_dependencies=args.include_dependencies)

    gallery_note = ""
    if args.gallery is not None:
        gallery_path = library_root / "gallery.png" if args.gallery == "__default__" else Path(args.gallery)
        render_library_gallery(library, gallery_path, columns=args.columns, include_retired=args.include_retired, include_dependencies=args.include_dependencies)
        gallery_note = f" | gallery -> {gallery_path}"

    if args.overmind:
        router_dir = library_root / "router"
        router_meta = router_dir / "router_meta.json"

        if args.metadata_overmind:
            if not router_meta.exists():
                console.print(f"[red]no router metadata at {router_meta}[/red] (use --cold-overmind for a structural portrait)")
                return
            render_overmind_from_metadata(library, router_meta, images_dir / "overmind.png")
            gallery_note += f" | overmind (metadata) -> {images_dir / 'overmind.png'}"
        elif not args.cold_overmind and router_meta.exists():
            import json

            from ardevo.routing import RouterService

            meta = json.loads(router_meta.read_text())
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
            # A cold portrait reads the current library but never loads or writes router state. Keep
            # image_dir unset during sync so adding experts cannot trigger an implicit first render.
            from ardevo.routing import RouterService
            from ardevo.utils.config import Config

            routed = (runtime or Config().current).get("orchestrator", {}).get("routed", {})
            service = RouterService(
                library,
                d_model=int(routed.get("d_model", 64)),
                top_k=int(routed.get("top_k", 2)),
                max_steps=int(routed.get("max_steps", 4)),
                adapter_rank=int(routed.get("adapter_rank", 0)),
                halting=bool(routed.get("halting", False)),
                edge_bias=bool(routed.get("edge_bias", False)),
                persist_dir=None,  # read-only: a portrait must never mint router state
                image_dir=None,
            )
            service.sync(include_compositions=bool(routed.get("include_compositions", True)), exclude_temporal=bool(routed.get("exclude_temporal", True)))
            if service.net._vertex_order:
                service.image_dir = images_dir
                service.render_overmind()
                reason = "persisted router state ignored" if args.cold_overmind else f"no router state at {router_dir}"
                console.print(f"[yellow]{reason}[/yellow]: rendered a COLD portrait (no trained gate or observed traffic)")
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
