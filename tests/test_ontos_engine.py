"""Tests for the Ontos Engine terminal object proof."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ontos_engine as ontos  # noqa: E402


def test_categorical_terminal_proof():
    assert ontos.categorical_terminal_proof() is True


def test_existence_necessitation():
    assert ontos.existence_necessitation() is True


def test_hom_into_core_is_singleton():
    core = ontos.singularity_core()
    for obj in ontos.canonical_witnesses():
        arrows = ontos.hom(obj, core)
        assert len(arrows) == 1
        assert arrows[0].table == {state: "*" for state in obj.states}


def test_invariant_violating_system_is_not_an_object():
    rogue = ontos.SystemObject("rogue", {"s0": 100000.0, "s1": 90000.0}, 100000.0)
    assert rogue.satisfies_invariant() is False


def test_proof_node_carries_only_digests():
    node = ontos.build_proof_node()
    assert node["node_id"] == "ONTOS_TERMINAL_OBJECT"
    assert node["holds"] is True
    assert ontos._is_digest(node["statement"])
    assert ontos._is_digest(node["proof_hash"])


def test_logged_records_are_hashes_only(caplog):
    ontos.setup_logging()
    with caplog.at_level("INFO", logger="Ontos"):
        ontos.categorical_terminal_proof()
    assert caplog.records
    for record in caplog.records:
        assert ontos._is_digest(ontos._HashOnlyFormatter().format(record))


def test_embed_in_proof_network():
    ok, digest = ontos.embed_in_proof_network()
    assert ok is True
    assert ontos._is_digest(digest)
