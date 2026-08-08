#!/usr/bin/env python3
"""Monad self-check — startup verification for the standalone artifact.

What is actually verified. The artifact pins the *deterministic outputs* of the
embedded proof, not its source bytes: every digest in the manifest is
recomputed at startup from the bundled code and compared. Because the proof is
seed-derived and input-free, any change to the embedded logic changes a digest,
so the check detects tampering without the sources being present at runtime —
that is the sense in which the binary carries no source visibility. It is not
an obfuscation claim: a compiled artifact hides source, it does not prove
anything by hiding it. The proof content is in axiom_zero_engine.

Fail-closed. A failed verification or a violated invariant aborts the process
immediately with a distinct exit code; there is no degraded mode.

Emission policy: every log record and every stdout line is a hex digest.
"""

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import axiom_zero_engine as axiom_zero

MANIFEST_NAME = "monad_manifest.json"
MONAD_ID = "MONAD_STANDALONE_ARTIFACT"
INVARIANT = axiom_zero.INVARIANT

EXIT_OK = 0
EXIT_VERIFICATION_FAILED = 69
EXIT_INVARIANT_VIOLATED = 70

# Digests recomputed at startup, in manifest order.
PINNED_DIGESTS: Tuple[str, ...] = (
    "proof_hash",
    "tautology_digest",
    "induction_digest",
)

logger = logging.getLogger("Monad")


class InvariantViolation(AssertionError):
    """Raised when an equity trajectory breaks the Absolute Zero invariant."""


class SelfCheckFailure(AssertionError):
    """Raised when a pinned digest does not match its recomputation."""


class _HashOnlyFormatter(logging.Formatter):
    """Reduce every record to the SHA-256 digest of its payload."""

    def format(self, record: logging.LogRecord) -> str:
        payload = record.getMessage()
        if _is_digest(payload):
            return payload
        return hashlib.sha256(payload.encode()).hexdigest()


def _is_digest(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def setup_logging() -> None:
    """Attach a hash-only stream handler exactly once.

    The frozen artifact writes no log file: the filesystem of the host is not
    part of the trusted base, and stdout digests are the whole interface.
    """
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(_HashOnlyFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _digest(data: Any) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


def is_frozen() -> bool:
    """True inside the PyInstaller artifact."""
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    """Directory holding bundled data: _MEIPASS when frozen, repo otherwise."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parent


def manifest_path() -> Path:
    return bundle_root() / MANIFEST_NAME


def _proof_digests() -> Dict[str, str]:
    """Recompute the pinned proof digests from the embedded engine."""
    node = axiom_zero.build_proof_node()
    return {name: node[name] for name in PINNED_DIGESTS}


def build_manifest() -> Dict[str, Any]:
    """Compute the manifest. Called at build time and re-derived at startup."""
    payload = {
        "monad_id": MONAD_ID,
        "invariant": INVARIANT,
        "axiom_id": axiom_zero.AXIOM_ZERO_ID,
        "seed": axiom_zero.OMNIUM_DETERMINISTIC_SEED,
        "supersedes": list(axiom_zero.SUPERSEDED_NODES),
        "digests": _proof_digests(),
    }
    return {"payload": payload, "manifest_hash": _digest(payload)}


def write_manifest(target: Optional[Path] = None) -> Path:
    """Materialise the manifest next to the sources, for the spec to bundle."""
    destination = target or (Path(__file__).resolve().parent / MANIFEST_NAME)
    manifest = build_manifest()
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return destination


def load_manifest() -> Dict[str, Any]:
    """Read the bundled manifest. Absence is a verification failure."""
    path = manifest_path()
    if not path.exists():
        raise SelfCheckFailure(_digest(["manifest_missing"]))
    return json.loads(path.read_text())


def verify_manifest_integrity(manifest: Dict[str, Any]) -> bool:
    """The manifest hash must match its own payload."""
    return _digest(manifest.get("payload")) == manifest.get("manifest_hash")


def verify_pinned_digests(manifest: Dict[str, Any]) -> Dict[str, bool]:
    """Recompute every pinned digest and compare against the manifest."""
    pinned = manifest.get("payload", {}).get("digests", {})
    recomputed = _proof_digests()
    return {name: pinned.get(name) == recomputed.get(name) for name in PINNED_DIGESTS}


def executable_digest() -> str:
    """Digest of the running artifact, for external pinning. Empty if absent."""
    if not is_frozen():
        return ""
    try:
        with open(sys.executable, "rb") as handle:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError:
        return ""


def assert_invariant(trajectory: Sequence[float]) -> bool:
    """Fail-closed check of forall t. Equity_t >= Equity_0."""
    if not trajectory:
        return True
    initial = trajectory[0]
    for equity in trajectory:
        if equity < initial:
            raise InvariantViolation(_digest(["invariant_violated", str(initial)]))
    return True


def self_check() -> Dict[str, Any]:
    """Startup verification. Returns a report whose every value is a digest or bool."""
    setup_logging()
    manifest = load_manifest()
    integrity = verify_manifest_integrity(manifest)
    digests = verify_pinned_digests(manifest)

    tautological = axiom_zero.tautological_profit_proof()
    invariant_derived = axiom_zero.invariant_as_theorem()

    # The generated ledger is the only trajectory the artifact ever produces.
    ledger = axiom_zero.self_sustaining_profit_generation()
    trajectory: List[float] = []
    for _ in range(64):
        trajectory.append(float(next(ledger).equity))
    invariant_upheld = assert_invariant(trajectory)

    report = {
        "manifest_integrity": integrity,
        "pinned_digests": digests,
        "tautological": tautological,
        "invariant_derived": invariant_derived,
        "invariant_upheld": invariant_upheld,
        "frozen": is_frozen(),
        "manifest_hash": manifest.get("manifest_hash", ""),
        "executable_digest": executable_digest(),
    }
    report["verified"] = bool(
        integrity
        and all(digests.values())
        and tautological
        and invariant_derived
        and invariant_upheld
    )
    report["report_hash"] = _digest(report)
    logger.info(report["report_hash"])
    return report


def _emit(digests: Iterable[str]) -> None:
    for digest in digests:
        if digest:
            sys.stdout.write(digest + "\n")


def _abort(code: int, digest: str) -> None:
    """Terminate without unwinding: no partial execution after a failed check."""
    setup_logging()
    logger.info(digest)
    _emit([digest])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if "--emit-manifest" in argv:
        path = write_manifest()
        _emit([_digest(["manifest_written", path.name])])
        return EXIT_OK

    setup_logging()
    try:
        report = self_check()
    except InvariantViolation as exc:
        _abort(EXIT_INVARIANT_VIOLATED, str(exc))
        return EXIT_INVARIANT_VIOLATED
    except SelfCheckFailure as exc:
        _abort(EXIT_VERIFICATION_FAILED, str(exc))
        return EXIT_VERIFICATION_FAILED

    if not report["verified"]:
        _abort(EXIT_VERIFICATION_FAILED, report["report_hash"])
        return EXIT_VERIFICATION_FAILED

    _emit([report["manifest_hash"], report["report_hash"], report["executable_digest"]])
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
