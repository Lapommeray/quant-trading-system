"""Tests for the Axiom Zero Engine: tautological profit and derived invariant."""

import itertools
import re
from fractions import Fraction

import axiom_zero_engine as az

DIGEST = re.compile(r"^[0-9a-f]{64}$")


def test_profit_is_a_tautology():
    assert az.tautological_profit_proof() is True


def test_theorem_has_no_market_atoms_and_is_market_independent():
    report = az._tautology_report()
    assert report["market_atoms_in_theorem"] == []
    assert report["flip_invariant"] is True
    assert report["rows"] == 2 ** len(az.ATOMS)


def test_profit_alone_is_not_valid_so_the_theorem_is_not_vacuous():
    assert az.is_tautology(az.PROFIT) is False
    assert az.is_tautology(az.TAUTOLOGICAL_PROFIT) is True


def test_invariant_is_derived_not_assumed():
    assert az.invariant_as_theorem() is True
    assert az.is_tautology(az.INVARIANT_THEOREM) is True


def test_generation_is_perpetual_and_non_decreasing():
    stream = az.self_sustaining_profit_generation()
    states = list(itertools.islice(stream, 256))
    assert states[0].step == 0
    assert all(b.equity >= a.equity for a, b in zip(states, states[1:]))
    assert all(state.equity >= states[0].equity for state in states)
    assert next(stream).step == 256


def test_closed_form_matches_the_generator():
    states = list(itertools.islice(az.self_sustaining_profit_generation(), 64))
    assert all(az.equity_at(s.step) == s.equity for s in states)
    assert az.equity_at(0) == Fraction(0)


def test_evaluate_rejects_atoms_outside_the_signature():
    try:
        az.evaluate("price_up", {"ledger_closed": True})
    except KeyError:
        pass
    else:
        raise AssertionError("unknown atom must not be silently defaulted")


def test_proof_node_supersedes_prior_nodes_and_carries_digests_only():
    node = az.build_proof_node()
    assert node["node_id"] == az.AXIOM_ZERO_ID
    assert node["holds"] is True
    assert node["root_dependency"] == az.ABSOLUTE_ZERO_ROOT
    assert "ONTOS_TERMINAL_OBJECT" in node["supersedes"]
    assert "AUTARKES_INERT_LOGICAL_CONSTANT" in node["supersedes"]
    assert DIGEST.match(node["proof_hash"])
    assert DIGEST.match(node["statement"])


def test_engine_emits_only_digests(capsys):
    engine = az.AxiomZeroEngine()
    result = engine.run_tautology_cycle(steps=8)
    assert DIGEST.match(result["cycle_hash"])
    az._emit(result["cycle_hash"])
    captured = capsys.readouterr()
    for line in (captured.out + captured.err).splitlines():
        assert DIGEST.match(line.strip()), line
