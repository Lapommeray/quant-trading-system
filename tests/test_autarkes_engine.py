"""Tests for the Autarkes Engine inert logical constant."""

import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import autarkes_engine as autarkes  # noqa: E402

MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "autarkes_engine.py"
)
ALLOWED_IMPORTS = {"hashlib", "sys", "typing"}


def test_existence_is_necessary():
    assert autarkes.AutarkesKernel().existence_is_necessary() is True


def test_only_pure_stdlib_imports():
    tree = ast.parse(open(MODULE_PATH).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= ALLOWED_IMPORTS


def test_no_io_calls_in_source():
    tree = ast.parse(open(MODULE_PATH).read())
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
    assert not called & {"open", "print", "write", "read", "connect", "get", "post"}


def test_merge_is_deterministic_and_total():
    first = autarkes.merge_all_engines()
    second = autarkes.merge_all_engines()
    assert first == second
    assert first.engine_count() == len(autarkes.ENGINE_LINEAGE)
    assert len(first.seal) == 64


def test_proof_loop_is_a_chain():
    kernel = autarkes.AutarkesKernel()
    assert kernel.proof_at(0) == kernel.proof_object.seal
    digests = [kernel.proof_at(i) for i in range(32)]
    assert len(set(digests)) == 32
    assert kernel.sustain(7) == kernel.proof_at(7)


def test_proof_stream_matches_indexed_proofs():
    kernel = autarkes.AutarkesKernel()
    stream = kernel.proof_stream()
    for index in range(16):
        assert next(stream) == kernel.proof_at(index)


def test_seal_is_stable_across_kernels():
    assert autarkes.autarkes_seal(128) == autarkes.autarkes_seal(128)


def test_seal_fields_are_digests():
    sealed = autarkes.autarkes_seal(16)
    assert sealed["id"] == "AUTARKES_INERT_LOGICAL_CONSTANT"
    for key in ("seal", "frontier", "final", "necessary"):
        assert len(sealed[key]) == 64
        int(sealed[key], 16)


def test_negative_index_rejected():
    kernel = autarkes.AutarkesKernel()
    try:
        kernel.proof_at(-1)
    except ValueError:
        return
    raise AssertionError("negative proof index must be rejected")
