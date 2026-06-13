"""Open-ended QD archive: admission keeps behaviorally DIVERSE stepping stones (the DGM idea), not
just the top-k by metric, so recombination has varied parts to build cross-task solutions from."""

from pathlib import Path

from ardevo.evolution.genome import ConnectionGene, Genome, NodeGene, NodeKind
from ardevo.library import LIBRARY_ADMISSION, MODULE, ModuleLibrary
from ardevo.orchestrator import _functional_token, _genome_behavior

_IO = {"inputs": [{"signature": "BINARY|K", "width": 2}], "output": {"signature": "BINARY|K", "width": 1}}


def _seed(library: ModuleLibrary, tag: int, behavior: list[str], metric: float, robustness: float) -> str:
    """Add a distinct entry (unique payload) carrying a behavior niche, as the orchestrator would."""
    payload = {"nodes": [], "connections": [], "macros": [], "tag": tag}
    return library.add(entry_type=MODULE, payload=payload, io=_IO, provenance={"accepted_metric": metric, "weight_robustness": robustness, "behavior": behavior})


def test_archive_keeps_distinct_behavior_niches(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    policy = LIBRARY_ADMISSION.get("archive")(min_metric=0.0, min_robustness=0.0, per_niche_cap=1, max_per_signature=10)
    for tag, behavior in enumerate((["ff"], ["rec"], ["refine"], ["macro"])):
        decision = policy(library, entry_type=MODULE, io=_IO, provenance={"accepted_metric": 0.9, "weight_robustness": 0.5, "behavior": behavior})
        assert decision.admit and decision.retire == ()  # each new niche is a fresh stepping stone
        _seed(library, tag, behavior, 0.9, 0.5)
    assert len(library) == 4  # four diverse entries on ONE io shape: a flat top-3 cap would lose one


def test_archive_replaces_only_within_a_full_niche(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    policy = LIBRARY_ADMISSION.get("archive")(per_niche_cap=1, max_per_signature=10)
    strong = _seed(library, 0, ["ff"], metric=0.95, robustness=0.6)
    # A weaker same-niche candidate is rejected; a stronger one tombstones the incumbent.
    weak = policy(library, entry_type=MODULE, io=_IO, provenance={"accepted_metric": 0.9, "weight_robustness": 0.4, "behavior": ["ff"]})
    assert not weak.admit
    better = policy(library, entry_type=MODULE, io=_IO, provenance={"accepted_metric": 0.99, "weight_robustness": 0.8, "behavior": ["ff"]})
    assert better.admit and better.retire == (strong,)


def test_archive_bounds_total_per_signature(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    policy = LIBRARY_ADMISSION.get("archive")(per_niche_cap=1, max_per_signature=3)
    for tag, behavior in enumerate((["a"], ["b"], ["c"])):
        _seed(library, tag, behavior, metric=0.9, robustness=0.5)
    # Signature is full (3) across 3 niches: a new niche admits only by tombstoning the weakest.
    full = policy(library, entry_type=MODULE, io=_IO, provenance={"accepted_metric": 0.99, "weight_robustness": 0.9, "behavior": ["d"]})
    assert full.admit and len(full.retire) == 1
    # A new niche no better than the weakest incumbent is refused: the archive stays bounded.
    refused = policy(library, entry_type=MODULE, io=_IO, provenance={"accepted_metric": 0.5, "weight_robustness": 0.1, "behavior": ["e"]})
    assert not refused.admit


def test_default_policy_would_collapse_the_diversity(tmp_path: Path) -> None:
    """Contrast: the flat default cap rejects a 4th same-io entry once 3 exist, regardless of how
    behaviorally novel it is. This is the diversity the archive preserves."""
    library = ModuleLibrary(tmp_path / "lib")
    default = LIBRARY_ADMISSION.get("default")(min_metric=0.0, min_robustness=0.0, per_signature_cap=3)
    for tag, behavior in enumerate((["a"], ["b"], ["c"])):
        _seed(library, tag, behavior, metric=0.9, robustness=0.5)
    novel = default(library, entry_type=MODULE, io=_IO, provenance={"accepted_metric": 0.9, "weight_robustness": 0.5, "behavior": ["d"]})
    assert not novel.admit  # default tossed a behaviorally-novel stepping stone the archive keeps


def test_genome_behavior_descriptor_distinguishes_kinds() -> None:
    plain = Genome(
        nodes={0: NodeGene(0, NodeKind.INPUT, "identity"), 1: NodeGene(1, NodeKind.OUTPUT, "identity")},
        connections=[ConnectionGene(0, 1, 1.0, True, 0)],
    )
    fancy = Genome(
        nodes={0: NodeGene(0, NodeKind.INPUT, "identity"), 1: NodeGene(1, NodeKind.OUTPUT, "identity"), 2: NodeGene(2, NodeKind.HIDDEN, "tanh", aggregation="product")},
        connections=[ConnectionGene(0, 2, 1.0, True, 0), ConnectionGene(2, 1, 1.0, True, 1), ConnectionGene(2, 2, 0.5, True, 2, recurrent=True)],
    )
    fancy.refine_steps = 3
    assert _genome_behavior(plain) == ["h0", "ff", "single", "sum", "flat"]
    fancy_behavior = _genome_behavior(fancy)
    assert "rec" in fancy_behavior and "refine" in fancy_behavior and "prod" in fancy_behavior
    assert _genome_behavior(plain) != fancy_behavior  # distinct niches


def test_behavior_is_persisted_in_the_index(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    key = _seed(library, 0, ["h1", "rec", "refine"], metric=0.9, robustness=0.5)
    summary = next(row for row in library.summaries() if row["key"] == key)
    assert summary["behavior"] == ["h1", "rec", "refine"]


# --- Pillar D: functional fingerprint niching ---------------------------------------------------

_FN_IO = {"inputs": [{"signature": "CONTINUOUS|C", "width": 2}], "output": {"signature": "CATEGORICAL|C", "width": 1}}


def _flat(weight: float) -> Genome:
    nodes = {0: NodeGene(0, NodeKind.INPUT, "identity"), 1: NodeGene(1, NodeKind.INPUT, "identity"), 2: NodeGene(2, NodeKind.OUTPUT, "identity")}
    return Genome(nodes=nodes, connections=[ConnectionGene(0, 2, weight, True, 0), ConnectionGene(1, 2, weight, True, 1)])


def test_functional_token_is_deterministic_and_distinguishes_functions() -> None:
    positive = _functional_token(_flat(2.0), _FN_IO)
    negative = _functional_token(_flat(-2.0), _FN_IO)
    assert positive is not None and negative is not None
    assert positive == _functional_token(_flat(2.0), _FN_IO)  # deterministic per io signature (fixed probe)
    assert positive != negative  # opposite functions land in different niches


def test_functional_token_skips_time_axis_io() -> None:
    temporal_io = {"inputs": [{"signature": "CONTINUOUS|C,T", "width": 4}], "output": {"signature": "CATEGORICAL|C", "width": 1}}
    assert _functional_token(_flat(1.0), temporal_io) is None  # static probe does not apply to TIME-axis io


def test_refine_stepping_stones_admit_at_a_lower_floor(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    policy = LIBRARY_ADMISSION.get("archive")(min_metric=0.9, min_robustness=0.0, per_niche_cap=2, max_per_signature=10, refine_depth_threshold=0.85)
    # A 0.87 deep-refine stepping stone clears the relaxed 0.85 floor; a 0.87 feedforward one does not.
    refine = policy(library, entry_type=MODULE, io=_IO, provenance={"accepted_metric": 0.87, "weight_robustness": 0.3, "behavior": ["h0", "refine"]})
    feedforward = policy(library, entry_type=MODULE, io=_IO, provenance={"accepted_metric": 0.87, "weight_robustness": 0.3, "behavior": ["h0", "single"]})
    assert refine.admit and not feedforward.admit
