"""Typed graph grammar induction, persistence, programs, and compilation."""

import json
import random
from pathlib import Path
from typing import Any

import pytest

from ardevo.evolution.composition import CompositionGenome, comp_topological_order
from ardevo.evolution.genome import Genome, InnovationTracker, topological_order
from ardevo.grammar import (
    Grammar,
    GrammarError,
    Program,
    ProgramEdge,
    ProgramNode,
    compile_program,
    crossover_program,
    induce_grammar,
    load_grammar,
    mutate_program,
    rebuild_grammar,
    repeat_production,
    save_grammar,
    seed_program,
    validate_program,
)
from ardevo.library import COMPOSITION, MODULE, ModuleLibrary

_IO = {"inputs": [{"signature": "BINARY|K", "width": 1}], "output": {"signature": "BINARY|K", "width": 1}}
_WIDE_IO = {"inputs": [{"signature": "BINARY|K", "width": 2}], "output": {"signature": "BINARY|K", "width": 1}}


def _gadget(offset: int, weight: float) -> dict[str, Any]:
    ids = [offset + index for index in range(4)]
    return {
        "nodes": [
            {"id": ids[0], "kind": "input", "activation": "identity"},
            {"id": ids[1], "kind": "hidden", "activation": "tanh"},
            {"id": ids[2], "kind": "hidden", "activation": "tanh", "aggregation": "product"},
            {"id": ids[3], "kind": "output", "activation": "identity"},
        ],
        "connections": [
            {"in": ids[0], "out": ids[1], "weight": weight, "enabled": True, "innovation": 0},
            {"in": ids[1], "out": ids[2], "weight": weight, "enabled": True, "innovation": 1},
            {"in": ids[0], "out": ids[2], "weight": weight, "enabled": True, "innovation": 2},
            {"in": ids[2], "out": ids[3], "weight": weight, "enabled": True, "innovation": 3},
        ],
    }


def _chain(weight: float) -> dict[str, Any]:
    return {
        "nodes": [
            {"id": 0, "kind": "input", "activation": "identity"},
            {"id": 1, "kind": "output", "activation": "identity"},
        ],
        "connections": [{"in": 0, "out": 1, "weight": weight, "enabled": True, "innovation": 0}],
    }


def _composition(glue: float) -> dict[str, Any]:
    return {
        "nodes": [
            {"id": 0, "kind": "input", "ref": "BINARY|K", "in_width": 0, "out_width": 2, "aggregation": "sum", "trainable": False},
            {
                "id": 1,
                "kind": "module",
                "ref": "library:m1_aaaaaaaaaaaa",
                "in_width": 2,
                "out_width": 1,
                "aggregation": "sum",
                "trainable": False,
            },
            {"id": 2, "kind": "output", "ref": "head", "in_width": 1, "out_width": 0, "aggregation": "sum", "trainable": False},
        ],
        "edges": [
            {"in": 0, "out": 1, "enabled": True, "innovation": 0, "glue": [glue, glue], "glue_rank": 0},
            {"in": 1, "out": 2, "enabled": True, "innovation": 1, "glue": [glue], "glue_rank": 0},
        ],
    }


def _independent_gadget_library(root: Path) -> tuple[ModuleLibrary, str, str, str]:
    library = ModuleLibrary(root)
    first = library.add(entry_type=MODULE, payload=_gadget(0, 0.5), io=_IO, provenance={"task": "a"})
    refinement = library.add(entry_type=MODULE, payload=_gadget(10, 1.0), io=_IO, provenance={"task": "a2", "refined_from": first})
    independent = library.add(entry_type=MODULE, payload=_gadget(20, 1.5), io=_IO, provenance={"task": "b"})
    return library, first, refinement, independent


def test_induction_counts_distinct_lineage_roots_and_infers_ports(tmp_path: Path) -> None:
    library, first, refinement, independent = _independent_gadget_library(tmp_path / "library")
    grammar = induce_grammar(library, module_sizes=(4,), composition_sizes=())

    assert len(grammar.productions) == 1
    production = grammar.productions[0]
    assert production.source_kind == MODULE
    assert production.support == 2
    assert production.occurrences == 3
    assert production.lineage_roots == tuple(sorted((first, independent)))
    assert production.exemplars == tuple(sorted((first, refinement, independent)))
    assert production.mdl_gain > 0
    assert [(port.name, port.direction, port.signature, port.width, port.role) for port in production.ports] == [
        ("in0", "input", "BINARY|K", 1, "terminal"),
        ("out0", "output", "BINARY|K", 1, "terminal"),
    ]


def test_refinements_alone_do_not_meet_independent_support(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "library")
    root = library.add(entry_type=MODULE, payload=_gadget(0, 0.5), io=_IO, provenance={})
    library.add(entry_type=MODULE, payload=_gadget(10, 1.0), io=_IO, provenance={"refined_from": root})

    assert induce_grammar(library, module_sizes=(4,), composition_sizes=()).productions == ()


def test_non_positive_mdl_motif_is_not_promoted(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "library")
    library.add(entry_type=MODULE, payload=_chain(0.5), io=_IO, provenance={})
    library.add(entry_type=MODULE, payload=_chain(1.5), io=_IO, provenance={})

    # Two nodes + one edge cost three tokens; declaring two typed ports removes the apparent saving.
    assert induce_grammar(library, module_sizes=(2,), composition_sizes=()).productions == ()


def test_submotif_infers_cut_port_and_compiles_external_boundary(tmp_path: Path) -> None:
    library, *_keys = _independent_gadget_library(tmp_path / "library")
    grammar = induce_grammar(library, module_sizes=(3,), composition_sizes=())

    assert len(grammar.productions) == 1
    production = grammar.productions[0]
    assert any(port.role == "cut" for port in production.ports)
    tracker = InnovationTracker(_next_node_id=100)
    compiled = compile_program(seed_program(production), grammar, tracker=tracker)
    assert isinstance(compiled, Genome)
    assert min(compiled.nodes) == 100
    assert len(compiled.input_ids) == 1 and len(compiled.output_ids) == 1
    assert tracker.new_node_id() == max(compiled.nodes) + 1


def test_grammar_round_trip_and_atomic_rebuild_are_deterministic(tmp_path: Path) -> None:
    library, *_keys = _independent_gadget_library(tmp_path / "library")
    first = induce_grammar(library, module_sizes=(4,), composition_sizes=())
    second = induce_grammar(library, module_sizes=(4,), composition_sizes=())
    assert first.to_dict() == second.to_dict()

    path = save_grammar(first, library)
    assert path == library.root / "grammar" / "grammar.json"
    assert load_grammar(library) == first
    assert not list(path.parent.glob("*.tmp"))
    rebuilt = rebuild_grammar(library, module_sizes=(4,), composition_sizes=())
    assert rebuilt == first
    assert json.loads(path.read_text()) == first.to_dict()


def test_missing_and_unknown_grammar_versions_fail_loudly(tmp_path: Path) -> None:
    assert load_grammar(tmp_path / "missing", missing_ok=True) == Grammar.empty()
    with pytest.raises(FileNotFoundError):
        load_grammar(tmp_path / "missing")
    with pytest.raises(GrammarError, match="unsupported grammar version"):
        Grammar.from_dict({"version": 999})


def test_program_serde_validation_and_flat_compilation(tmp_path: Path) -> None:
    library, *_keys = _independent_gadget_library(tmp_path / "library")
    grammar = induce_grammar(library, module_sizes=(4,), composition_sizes=())
    production = grammar.productions[0]
    program = Program(
        nodes=(ProgramNode(1, production.key), ProgramNode(0, production.key)),
        edges=(ProgramEdge(0, "out0", 1, "in0"),),
    )
    validate_program(program, grammar)
    round_trip = Program.from_dict(json.loads(json.dumps(program.to_dict())))
    assert round_trip.to_dict() == program.to_dict()

    compiled = compile_program(program, grammar)
    assert isinstance(compiled, Genome)
    assert len(compiled.input_ids) == 1 and len(compiled.output_ids) == 1
    assert len(compiled.nodes) == 8
    topological_order(compiled)


def test_program_rejects_cycles_and_claimed_input_twice(tmp_path: Path) -> None:
    library, *_keys = _independent_gadget_library(tmp_path / "library")
    grammar = induce_grammar(library, module_sizes=(4,), composition_sizes=())
    key = grammar.productions[0].key
    cyclic = Program(
        nodes=(ProgramNode(0, key), ProgramNode(1, key)),
        edges=(ProgramEdge(0, "out0", 1, "in0"), ProgramEdge(1, "out0", 0, "in0")),
    )
    with pytest.raises(GrammarError, match="cycle"):
        validate_program(cyclic, grammar)

    claimed = Program(
        nodes=(ProgramNode(0, key), ProgramNode(1, key), ProgramNode(2, key)),
        edges=(ProgramEdge(0, "out0", 2, "in0"), ProgramEdge(1, "out0", 2, "in0")),
    )
    with pytest.raises(GrammarError, match="connected more than once"):
        validate_program(claimed, grammar)


def test_repeat_mutation_and_crossover_preserve_valid_programs(tmp_path: Path) -> None:
    library, *_keys = _independent_gadget_library(tmp_path / "library")
    grammar = induce_grammar(library, module_sizes=(4,), composition_sizes=())
    key = grammar.productions[0].key
    parent = Program(nodes=(ProgramNode(0, key), ProgramNode(1, key)), edges=(ProgramEdge(0, "out0", 1, "in0"),))

    repeated = repeat_production(parent, grammar, rng=random.Random(0))
    assert len(repeated.nodes) == 3 and len(repeated.edges) == 2
    validate_program(repeated, grammar)
    assert mutate_program(parent, grammar, rng=random.Random(0), operator="repeat") == repeated
    child = crossover_program(repeated, repeated, grammar, rng=random.Random(1))
    assert child == repeated


def test_composition_motif_ports_and_library_backed_compilation(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "library")
    library.add(entry_type=COMPOSITION, payload=_composition(0.5), io=_WIDE_IO, provenance={}, level=2)
    library.add(entry_type=COMPOSITION, payload=_composition(1.5), io=_WIDE_IO, provenance={}, level=2)
    grammar = induce_grammar(library, module_sizes=(), composition_sizes=(3,))

    assert len(grammar.productions) == 1
    production = grammar.productions[0]
    assert [(port.direction, port.signature, port.width) for port in production.ports] == [
        ("input", "BINARY|K", 2),
        ("output", "BINARY|K", 1),
    ]
    compiled = compile_program(seed_program(production), grammar, library=library, rng=random.Random(0))
    assert isinstance(compiled, CompositionGenome)
    assert len(compiled.input_ids) == 1 and len(compiled.output_ids) == 1 and len(compiled.module_ids) == 1
    comp_topological_order(compiled)


def test_compile_needs_library_when_rule_cannot_flatten(tmp_path: Path) -> None:
    library = ModuleLibrary(tmp_path / "library")
    library.add(entry_type=COMPOSITION, payload=_composition(0.5), io=_WIDE_IO, provenance={}, level=2)
    library.add(entry_type=COMPOSITION, payload=_composition(1.5), io=_WIDE_IO, provenance={}, level=2)
    grammar = induce_grammar(library, module_sizes=(), composition_sizes=(3,))

    with pytest.raises(GrammarError, match="cannot expand"):
        compile_program(seed_program(grammar.productions[0]), grammar)
