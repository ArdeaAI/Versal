"""net_gallery CLI seam: per-entry renders report honest rows and respect the index filters."""

from pathlib import Path

from ardevo.evolution.genome import Genome, genome_to_dict
from ardevo.library import MODULE, ModuleLibrary
from ardevo.tools.net_gallery import render_all_entries

_FIXTURE_LIBRARY = Path(__file__).parent / "fixtures" / "library_v1"
_IO = {"inputs": [{"signature": "BINARY|K", "width": 2}], "output": {"signature": "BINARY|K", "width": 1}}


def test_render_all_entries_reports_rows(tmp_path: Path) -> None:
    library = ModuleLibrary(_FIXTURE_LIBRARY)
    rows = render_all_entries(library, tmp_path / "renders")
    assert len(rows) == len(library)
    assert all(row["status"] == "OK" for row in rows)
    for row in rows:
        assert Path(row["path"]).exists() and Path(row["path"]).stat().st_size > 0


def test_render_all_entries_filters_retired(tmp_path: Path, solving_genome: Genome, linear_genome: Genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    kept = library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=_IO, provenance={})
    retired = library.add(entry_type=MODULE, payload=genome_to_dict(linear_genome), io=_IO, provenance={})
    library.retire(retired)

    rows = render_all_entries(library, tmp_path / "renders")
    assert [row["key"] for row in rows] == [kept]
    rows_all = render_all_entries(library, tmp_path / "renders_all", include_retired=True)
    assert {row["key"] for row in rows_all} == {kept, retired}
