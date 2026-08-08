"""Tests for the Monad artifact's startup self-verification and fail-closed abort."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import monad_self_check as monad

DIGEST = re.compile(r"^[0-9a-f]{64}$")
REPO_ROOT = Path(monad.__file__).resolve().parent


def _sandbox(tmp_path: Path) -> Path:
    for name in ("monad_self_check.py", "axiom_zero_engine.py", monad.MANIFEST_NAME):
        shutil.copy(REPO_ROOT / name, tmp_path / name)
    return tmp_path


def _run(cwd: Path):
    return subprocess.run(
        [sys.executable, "monad_self_check.py"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_committed_manifest_matches_the_current_proof():
    on_disk = json.loads((REPO_ROOT / monad.MANIFEST_NAME).read_text())
    assert on_disk == monad.build_manifest()


def test_self_check_verifies():
    report = monad.self_check()
    assert report["verified"] is True
    assert report["manifest_integrity"] is True
    assert all(report["pinned_digests"].values())
    assert DIGEST.match(report["report_hash"])


def test_manifest_tampering_is_detected():
    manifest = monad.build_manifest()
    manifest["payload"]["seed"] = manifest["payload"]["seed"] + 1
    assert monad.verify_manifest_integrity(manifest) is False


def test_pinned_digest_mismatch_is_detected():
    manifest = monad.build_manifest()
    manifest["payload"]["digests"]["proof_hash"] = "0" * 64
    assert monad.verify_pinned_digests(manifest)["proof_hash"] is False


def test_invariant_assertion_is_fail_closed():
    assert monad.assert_invariant([1.0, 1.0, 2.0, 2.5]) is True
    assert monad.assert_invariant([]) is True
    with pytest.raises(monad.InvariantViolation):
        monad.assert_invariant([1.0, 2.0, 0.9])


def test_startup_succeeds_and_emits_only_digests(tmp_path):
    result = _run(_sandbox(tmp_path))
    assert result.returncode == monad.EXIT_OK
    lines = (result.stdout + result.stderr).split()
    assert lines
    for line in lines:
        assert DIGEST.match(line), line


def test_startup_aborts_when_a_pinned_digest_is_wrong(tmp_path):
    sandbox = _sandbox(tmp_path)
    manifest = json.loads((sandbox / monad.MANIFEST_NAME).read_text())
    manifest["payload"]["digests"]["tautology_digest"] = "1" * 64
    manifest["manifest_hash"] = monad._digest(manifest["payload"])
    (sandbox / monad.MANIFEST_NAME).write_text(json.dumps(manifest))

    result = _run(sandbox)
    assert result.returncode == monad.EXIT_VERIFICATION_FAILED


def test_startup_aborts_when_the_manifest_is_missing(tmp_path):
    sandbox = _sandbox(tmp_path)
    (sandbox / monad.MANIFEST_NAME).unlink()

    result = _run(sandbox)
    assert result.returncode == monad.EXIT_VERIFICATION_FAILED


def test_emit_manifest_regenerates_the_pin(tmp_path):
    sandbox = _sandbox(tmp_path)
    (sandbox / monad.MANIFEST_NAME).write_text("{}")
    regenerate = subprocess.run(
        [sys.executable, "monad_self_check.py", "--emit-manifest"],
        cwd=sandbox,
        capture_output=True,
        text=True,
    )
    assert regenerate.returncode == monad.EXIT_OK
    assert json.loads((sandbox / monad.MANIFEST_NAME).read_text()) == (
        monad.build_manifest()
    )
