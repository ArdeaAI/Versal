"""
Bounded preparation preserves the original grammar and survives interruptions.
"""

import copy
import json
from typing import Any

from tests.test_grammar import _independent_gadget_library
from versal.grammar import _canonical_mapping, _ports_for_occurrence, induce_grammar
from versal.grammar_work import GrammarPreparation, OccurrenceCursor
from versal.motifs import canonical_form, enumerate_connected_subgraphs, module_motif_graph


def test_cursor_matches_original_enumeration_and_port_typing(tmp_path):
    library, key, _, _ = _independent_gadget_library(tmp_path)
    entry = library.load(key)
    for cap in (1, 2, 50):
        cursor = OccurrenceCursor(entry, (2, 3, 4), cap, "root")
        labels, edges = module_motif_graph(entry.payload)
        expected = []
        for size in (2, 3, 4):
            subsets, _ = enumerate_connected_subgraphs(cursor.skeleton, size, cap)
            for subset in subsets:
                members = tuple(sorted(subset))
                local = {node: i for i, node in enumerate(members)}
                local_labels = [labels[node] for node in members]
                local_edges = [(local[a], local[b], mask) for (a, b), mask in edges.items() if a in local and b in local]
                motif = canonical_form(local_labels, local_edges)
                ports = _ports_for_occurrence(entry, labels, edges, members, _canonical_mapping(local_labels, local_edges, motif))
                expected.append((motif, ports))
        actual = []
        while not cursor.done:
            item = cursor.step()
            if item:
                actual.append((item.motif, item.ports))
            # Reconstruct every step to exercise the persisted DFS stack, including
            # positions where a branch has been entered but has not emitted a motif.
            cursor = OccurrenceCursor(entry, (2, 3, 4), cap, "root", json.loads(json.dumps(cursor.state_dict())))
        assert actual == expected


def test_preparation_resume_and_cache_match_complete_induction(tmp_path):
    library, _, _, _ = _independent_gadget_library(tmp_path)
    params: dict[str, Any] = dict(module_sizes=(2, 3, 4), composition_sizes=(2, 3), per_entry_cap=50, min_lineage_support=2)
    expected = induce_grammar(library, **params).to_dict()
    work = GrammarPreparation(library, params)
    turns = 0
    while not work.advance(should_stop=lambda: False, items=3, seconds=10):
        turns += 1
        assert turns < 500
        work = GrammarPreparation(library, params, copy.deepcopy(work.state_dict()))
    assert work.grammar is not None
    assert work.grammar.to_dict() == expected
    cached = GrammarPreparation(library, params)
    while not cached.advance(should_stop=lambda: False, items=3, seconds=10):
        pass
    assert cached.grammar is not None
    assert cached.grammar.to_dict() == expected


def test_cancelled_preparation_does_not_advance_or_publish(tmp_path):
    library, _, _, _ = _independent_gadget_library(tmp_path)
    params: dict[str, Any] = dict(module_sizes=(3,), composition_sizes=(3,), per_entry_cap=50, min_lineage_support=2)
    work = GrammarPreparation(library, params)
    before = copy.deepcopy(work.state_dict())
    assert not work.advance(should_stop=lambda: True)
    assert work.state_dict() == before
    assert not (tmp_path / "grammar/grammar.json").exists()


def test_grammar_preparation_spends_its_share_without_starving_search(tmp_path, monkeypatch, xor_task):
    from tests.test_interleaved import _policy
    from tests.test_orchestrator import _orchestrator
    from versal.orchestrator import comp_task_spec
    from versal.strategy_common import StrategyResult
    from versal.strategy_sessions import SESSION_STRATEGY, GrammarSession, StrategySession

    calls = []

    class Session(StrategySession):
        def _advance(self, runtime):
            calls.append(self.role)
            return StrategyResult(self.role, 0.25, 1, champion_metrics={"support_accuracy": 0.25})

    for name in ("direct", "spatial"):
        monkeypatch.setitem(SESSION_STRATEGY._items, name, Session)
    orchestrator = _orchestrator(
        tmp_path,
        table=_policy(evolve=["grammar", "direct", "spatial"], evolve_budget={"grammar": 1, "direct": 1, "spatial": 1}, max_depth=0),
    )
    grammar = dict(orchestrator.strategies)["grammar"]

    def unfinished(runtime):
        calls.append("prepare")
        return False

    monkeypatch.setattr(grammar, "prepare", unfinished)
    spec = comp_task_spec(xor_task, include_query=False)
    session = GrammarSession(grammar, xor_task, spec, seed=0)
    assert session.ready(orchestrator._runtime()) and calls == []
    assert orchestrator.search is not None
    result = orchestrator.search.run(xor_task, spec, 9)
    assert len(calls) == 9
    assert all(calls.count(name) == 3 for name in ("prepare", "direct", "spatial"))
    assert result.generations_used == 6
    assert result.strategy_work["grammar"]["preparation_steps"] == 3
    assert result.strategy_work["grammar"]["generations"] == 0


def test_spatial_repetition_does_not_multiply_independent_grammar_evidence(tmp_path):
    from dataclasses import replace

    from tests.test_spatial_recipe import recipe, task
    from versal.evolution.genome import genome_to_dict
    from versal.library import MODULE, ModuleLibrary, task_io

    library = ModuleLibrary(tmp_path)
    genome = recipe()
    for weight in (0.2, 0.5):
        genome.connections[0] = replace(genome.connections[0], weight=weight)
        library.add(entry_type=MODULE, payload=genome_to_dict(genome), io=task_io(task()), provenance={"search_lineage": "one-search"})
    assert induce_grammar(library, module_sizes=(3,), composition_sizes=()).productions == ()
    genome.connections[0] = replace(genome.connections[0], weight=0.8)
    library.add(entry_type=MODULE, payload=genome_to_dict(genome), io=task_io(task()), provenance={"search_lineage": "independent-search"})
    grammar = induce_grammar(library, module_sizes=(3,), composition_sizes=())
    assert grammar.productions
    assert all(production.support == 2 for production in grammar.productions)


def test_catalog_token_tracks_lineage_but_not_usage(tmp_path):
    from versal.grammar_work import catalog_token

    library, key, _, _ = _independent_gadget_library(tmp_path)
    before = catalog_token(library)
    library._index[key]["use_count"] = 99
    assert catalog_token(library) == before
    library._index[key]["search_lineage"] = "new-evidence-owner"
    assert catalog_token(library) != before
