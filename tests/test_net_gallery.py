"""net_gallery CLI seam: render selection is explicit and persistent state stays isolated."""

import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest

import versal.rendering as rendering
import versal.routing as routing
from versal.evolution.genome import Genome, genome_to_dict
from versal.library import MODULE, ModuleLibrary
from versal.tools import net_gallery
from versal.tools.net_gallery import refresh_results_run, render_all_entries

_FIXTURE_LIBRARY = Path(__file__).parent / "fixtures" / "library_v1"
_IO = {"inputs": [{"signature": "BINARY|K", "width": 2}], "output": {"signature": "BINARY|K", "width": 1}}


def test_render_all_entries_reports_rows(tmp_path: Path) -> None:
    library = ModuleLibrary(_FIXTURE_LIBRARY)
    rows = render_all_entries(library, tmp_path / "renders")
    assert len(rows) == len(library)
    assert all(row["status"] == "OK" for row in rows)
    for row in rows:
        assert Path(row["path"]).exists() and Path(row["path"]).stat().st_size > 0


def test_render_all_entries_reports_density_portrait_as_ok(tmp_path: Path, solving_genome: Genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    key = library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=_IO, provenance={})

    rows = render_all_entries(library, tmp_path / "renders", node_budget=1)

    assert rows == [{"key": key, "entry_type": MODULE, "level": 1, "status": "OK", "path": str(tmp_path / "renders" / f"{key}.png")}]
    assert Path(rows[0]["path"]).stat().st_size > 0


def test_render_all_entries_filters_retired(tmp_path: Path, solving_genome: Genome, linear_genome: Genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    kept = library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=_IO, provenance={})
    retired = library.add(entry_type=MODULE, payload=genome_to_dict(linear_genome), io=_IO, provenance={})
    library.retire(retired)

    rows = render_all_entries(library, tmp_path / "renders")
    assert [row["key"] for row in rows] == [kept]
    rows_all = render_all_entries(library, tmp_path / "renders_all", include_retired=True)
    assert {row["key"] for row in rows_all} == {kept, retired}


def test_refresh_results_run_rebuilds_network_and_speciation_from_json(
    tmp_path: Path,
    solving_genome: Genome,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from versal import results

    library = ModuleLibrary(tmp_path / "library")
    key = library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=_IO, provenance={})
    task = tmp_path / "run" / "task_0003"
    task.mkdir(parents=True)
    (task / "stats.json").write_text(json.dumps({"task_cursor": 3, "library": {"net_key": key}}))
    (task / "checkpoint.json").write_text(json.dumps({"loop_state": {"module_species_history": [{"0": 2}, {"0": 1, "1": 1}]}}))
    captured: dict[str, Any] = {}

    def capture_network(directory, genome, *, title, library, max_inline_depth):
        captured["network"] = (directory, len(genome.nodes), title, library.root, max_inline_depth)
        return directory / "net.png"

    def capture_speciation(directory, history, *, title):
        captured["speciation"] = (directory, history, title)
        return directory / "speciation.png"

    monkeypatch.setattr(rendering, "render_network", capture_network)
    monkeypatch.setattr(results, "render_speciation", capture_speciation)

    rows = refresh_results_run(library, task.parent, max_inline_depth=7)

    assert rows == [{"task": "task_0003", "net": "OK", "speciation": "OK"}]
    assert captured["network"] == (task, len(solving_genome.nodes), f"orchestrated task 3: module {key}", library.root, 7)
    assert captured["speciation"] == (task, [{0: 2}, {0: 1, 1: 1}], "module species through task 3")


def test_refresh_results_run_preserves_existing_images_when_sources_fail(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "library")
    task = tmp_path / "run" / "task_0001"
    task.mkdir(parents=True)
    (task / "stats.json").write_text(json.dumps({"task_cursor": 1, "library": {"net_key": "missing"}}))
    (task / "checkpoint.json").write_text("not json")
    (task / "net.png").write_bytes(b"old net")
    (task / "speciation.png").write_bytes(b"old chart")

    rows = refresh_results_run(library, task.parent)

    assert rows == [{"task": "task_0001", "net": "FAIL:KeyError", "speciation": "FAIL:JSONDecodeError"}]
    assert (task / "net.png").read_bytes() == b"old net"
    assert (task / "speciation.png").read_bytes() == b"old chart"


class _LibrarySpy:
    roots: list[Path] = []

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.roots.append(self.root)

    def __len__(self) -> int:
        return 0


def _write_config(path: Path, library_root: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "[orchestrator]",
                f"library_dir = {json.dumps(str(library_root))}",
                "[orchestrator.routed]",
                "d_model = 16",
                "top_k = 1",
                "max_steps = 2",
            )
        )
    )


@pytest.fixture
def isolated_cli(monkeypatch: pytest.MonkeyPatch) -> type[_LibrarySpy]:
    """Replace filesystem/render dependencies so CLI tests cannot touch a live campaign."""
    _LibrarySpy.roots.clear()
    monkeypatch.setattr(net_gallery, "ModuleLibrary", _LibrarySpy)
    monkeypatch.setattr(net_gallery, "render_all_entries", lambda *_args, **_kwargs: [])
    return _LibrarySpy


def test_cli_uses_configured_library_when_library_flag_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli: type[_LibrarySpy],
) -> None:
    configured = tmp_path / "configured-library"
    configured.mkdir()
    config = tmp_path / "run.toml"
    _write_config(config, configured)
    monkeypatch.setattr(sys, "argv", ["render", "--config", str(config)])

    net_gallery.main()

    assert isolated_cli.roots == [configured]


def test_cli_explicit_library_overrides_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli: type[_LibrarySpy],
) -> None:
    configured = tmp_path / "configured-library"
    configured.mkdir()
    explicit = tmp_path / "explicit-library"
    explicit.mkdir()
    config = tmp_path / "run.toml"
    _write_config(config, configured)
    monkeypatch.setattr(sys, "argv", ["render", "--config", str(config), "--library", str(explicit)])

    net_gallery.main()

    assert isolated_cli.roots == [explicit]


def test_cli_results_run_routes_historical_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli: type[_LibrarySpy],
) -> None:
    library_root = tmp_path / "library"
    library_root.mkdir()
    run_directory = tmp_path / "results" / "run"
    run_directory.mkdir(parents=True)
    captured: dict[str, Any] = {}

    def capture(library, run, **kwargs):
        captured.update(library=library.root, run=run, kwargs=kwargs)
        return []

    monkeypatch.setattr(net_gallery, "refresh_results_run", capture)
    monkeypatch.setattr(sys, "argv", ["render", "--library", str(library_root), "--results-run", str(run_directory), "--results-only"])

    net_gallery.main()

    assert captured == {"library": library_root, "run": run_directory, "kwargs": {}}


def test_cold_overmind_only_skips_entry_gallery_and_persisted_router(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli: type[_LibrarySpy],
) -> None:
    library_root = tmp_path / "library"
    router_dir = library_root / "router"
    router_dir.mkdir(parents=True)
    # Deliberately invalid: --cold-overmind must not even parse persisted metadata.
    (router_dir / "router_meta.json").write_text("not json")
    config = tmp_path / "run.toml"
    _write_config(config, library_root)
    images = tmp_path / "preview"
    calls: dict[str, Any] = {"renders": 0, "syncs": 0}

    def forbidden_entry_render(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("--overmind-only must skip per-entry rendering")

    class RouterSpy:
        instances: list["RouterSpy"] = []

        def __init__(self, _library: Any, **kwargs: Any) -> None:
            calls["persist_dir"] = kwargs["persist_dir"]
            self.image_dir = kwargs["image_dir"]
            self.net = type("NetSpy", (), {"_vertex_order": ["module"]})()
            self.instances.append(self)

        def sync(self, **_kwargs: Any) -> int:
            calls["syncs"] += 1
            return 1

        def render_overmind(self) -> None:
            calls["renders"] += 1

    monkeypatch.setattr(net_gallery, "render_all_entries", forbidden_entry_render)
    monkeypatch.setattr(routing, "RouterService", RouterSpy)
    monkeypatch.setattr(
        sys,
        "argv",
        ["render", "--config", str(config), "--images", str(images), "--overmind-only", "--cold-overmind"],
    )

    net_gallery.main()

    assert isolated_cli.roots == [library_root]
    assert calls == {"renders": 1, "syncs": 1, "persist_dir": None}
    assert RouterSpy.instances[0].image_dir == images


def test_metadata_overmind_is_isolated_from_entry_and_router_state_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli: type[_LibrarySpy],
) -> None:
    library_root = tmp_path / "library"
    router_dir = library_root / "router"
    router_dir.mkdir(parents=True)
    metadata = {"d_model": 64, "vertex_order": ["m1_fixture"]}
    metadata_path = router_dir / "router_meta.json"
    metadata_path.write_text(json.dumps(metadata))
    # Metadata rendering must remain usable even when the heavyweight tensor state is unreadable.
    (router_dir / "router_state.pt").write_text("not a torch checkpoint")
    config = tmp_path / "run.toml"
    _write_config(config, library_root)
    images = tmp_path / "preview"
    calls: dict[str, Any] = {}

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("metadata-only rendering entered a heavyweight render path")

    def metadata_renderer(_library: Any, source: Path, destination: Path) -> Path:
        calls["metadata"] = json.loads(source.read_text())
        calls["source"] = source
        calls["destination"] = destination
        return destination

    monkeypatch.setattr(net_gallery, "render_all_entries", forbidden)
    monkeypatch.setattr(net_gallery, "render_library_gallery", forbidden)
    monkeypatch.setattr(net_gallery, "render_overmind_from_metadata", metadata_renderer, raising=False)
    monkeypatch.setattr(routing, "RouterService", forbidden)
    monkeypatch.setattr(routing.torch, "load", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["render", "--config", str(config), "--images", str(images), "--metadata-overmind"],
    )

    net_gallery.main()

    assert isolated_cli.roots == [library_root]
    assert calls == {
        "metadata": metadata,
        "source": metadata_path,
        "destination": images / "overmind.png",
    }


def test_render_overmind_from_metadata_builds_traffic_view_without_rendering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summaries = [
        {"key": "m1_alpha", "retired": False},
        {"key": "m1_beta", "retired": False},
        {"key": "m1_retired", "retired": True},
        {"key": "m1_evicted", "retired": False},
    ]

    class MetadataLibrarySpy:
        def summaries(self, *, include_retired: bool = False) -> list[dict[str, Any]]:
            assert include_retired is True
            return summaries

    meta = {
        "d_model": 64,
        "top_k": 2,
        "max_steps": 4,
        "vertex_keys": ["m1_alpha", "m1_beta", "m1_retired", "m1_missing"],
        "input_adapter_keys": [{"key": "BINARY|K:2", "width": 2}, "REAL|K:3"],
        "output_head_keys": [{"key": "BINARY|K:1", "width": 1}],
        "usage_totals": {"m1_alpha": 2.0, "m1_beta": 6.0, "m1_retired": 2.0},
        "step_usage_totals": {
            "m1_alpha": [0.0, 2.0],
            "m1_beta": [4.0, 0.0],
            "m1_retired": [1.0, 1.0],
        },
        "transition_totals": {
            "m1_alpha": {"m1_beta": 3.0, "m1_retired": 1.0},
            "m1_beta": {"m1_alpha": 6.0},
            "m1_missing": {"m1_beta": 100.0},
        },
        "evicted": {"m1_evicted": {"route_epoch": 8, "reuse_epoch": 4}},
    }
    metadata_path = tmp_path / "router_meta.json"
    metadata_path.write_text(json.dumps(meta))
    out_path = tmp_path / "preview" / "overmind.png"
    captured: dict[str, Any] = {}

    def capture_render(path: Path, view: rendering.OvermindView, *, library: Any) -> Path:
        captured.update(path=path, view=view, library=library)
        return path

    monkeypatch.setattr(rendering, "render_overmind", capture_render)
    library_spy = MetadataLibrarySpy()
    library = cast(ModuleLibrary, library_spy)

    result = net_gallery.render_overmind_from_metadata(library, metadata_path, out_path)

    assert result == out_path
    assert captured["path"] == out_path
    assert captured["library"] is library_spy
    view = captured["view"]
    assert [vertex.key for vertex in view.vertices] == ["m1_beta", "m1_alpha", "m1_retired", "m1_evicted"]
    assert [vertex.usage for vertex in view.vertices] == pytest.approx([0.6, 0.2, 0.2, 0.0])
    assert [vertex.mean_step for vertex in view.vertices[:3]] == pytest.approx([0.0, 1.0, 0.5])
    assert view.vertices[-1].mean_step is None
    assert view.vertices[-2].retired is True and view.vertices[-1].retired is True
    assert "route evicted" in view.vertices[-1].label
    assert view.input_signatures == ["BINARY|K:2", "REAL|K:3"]
    assert view.output_signatures == ["BINARY|K:1"]
    assert view.pathways == pytest.approx([(1, 0, 0.5), (1, 2, 1.0 / 6.0), (0, 1, 1.0)])
    assert (view.d_model, view.top_k, view.max_steps) == (64, 2, 4)
