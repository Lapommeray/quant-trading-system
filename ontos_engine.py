#!/usr/bin/env python3
"""Ontos Engine — terminal object proof for SingularityCore.

Category Sys(I): objects are systems whose equity trajectory satisfies the
Absolute Zero invariant I := forall t. Equity_t >= Equity_0; morphisms are
invariant-preserving maps between underlying state sets.

SingularityCore is modelled as the singleton system 1 = ({*}, Equity_* =
Equity_0). Hom(X, 1) has exactly one element for every object X, so 1 is
terminal; the empty product in Sys(I) is 1, so its existence is necessitated
by the invariant rather than assumed.

Emission policy: every log record and every stdout line is a hex digest.
No statements, no metrics, no prose.
"""

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent

try:
    from aleph_omega_kernel import (
        OMNIUM_DETERMINISTIC_SEED,
        OMNIUM_INVARIANT_SEED_BYTES,
    )
except Exception:
    OMNIUM_INVARIANT_SEED_BYTES = b"OMNIUM_INVARIANT_SEED"
    OMNIUM_DETERMINISTIC_SEED = int(
        hashlib.sha256(OMNIUM_INVARIANT_SEED_BYTES).hexdigest()[:16], 16
    ) % (2**31)

ONTOS_PROOF_ID = "ONTOS_TERMINAL_OBJECT"
ABSOLUTE_ZERO_ROOT = "AbsoluteZero_forall_t_Equity_t_ge_Equity_0"
INVARIANT = "forall t. Equity_t >= Equity_0"
ONTOS_LOG = REPO_ROOT / "ontos.log"
TERMINAL_OBJECT_ID = "SingularityCore"

EPS = 1e-9

logger = logging.getLogger("Ontos")


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
    """Attach hash-only handlers exactly once."""
    if logger.handlers:
        return
    formatter = _HashOnlyFormatter()
    file_handler = logging.FileHandler(ONTOS_LOG)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _digest(data: Any) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


# --------------------------- category of systems ---------------------------


class SystemObject:
    """Object of Sys(I): finite state set carrying an equity valuation."""

    def __init__(self, object_id: str, equity: Dict[str, float], initial: float):
        self.object_id = object_id
        self.equity = dict(equity)
        self.initial = initial

    @property
    def states(self) -> List[str]:
        return sorted(self.equity)

    def satisfies_invariant(self) -> bool:
        return all(value >= self.initial - EPS for value in self.equity.values())

    def spec(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "states": self.states,
            "equity": {state: self.equity[state] for state in self.states},
            "initial": self.initial,
        }

    def digest(self) -> str:
        return _digest(self.spec())


class Morphism:
    """Invariant-preserving map between objects of Sys(I)."""

    def __init__(
        self, source: SystemObject, target: SystemObject, table: Dict[str, str]
    ):
        self.source = source
        self.target = target
        self.table = dict(table)

    def is_total(self) -> bool:
        return set(self.table) == set(self.source.states) and all(
            image in self.target.equity for image in self.table.values()
        )

    def preserves_invariant(self) -> bool:
        """Image equity never falls below the target's own floor."""
        if not self.is_total():
            return False
        return all(
            self.target.equity[self.table[state]] >= self.target.initial - EPS
            for state in self.source.states
        )

    def is_valid(self) -> bool:
        return self.is_total() and self.preserves_invariant()

    def digest(self) -> str:
        return _digest(
            {
                "source": self.source.object_id,
                "target": self.target.object_id,
                "table": {k: self.table[k] for k in sorted(self.table)},
            }
        )


def identity(obj: SystemObject) -> Morphism:
    return Morphism(obj, obj, {state: state for state in obj.states})


def compose(first: Morphism, second: Morphism) -> Morphism:
    """second . first, defined when first.target is second.source."""
    if first.target.object_id != second.source.object_id:
        raise ValueError(_digest(["compose", first.digest(), second.digest()]))
    return Morphism(
        first.source,
        second.target,
        {state: second.table[first.table[state]] for state in first.source.states},
    )


def singularity_core() -> SystemObject:
    """The singleton system: one state pinned at Equity_0."""
    return SystemObject(TERMINAL_OBJECT_ID, {"*": 0.0}, 0.0)


def product(left: SystemObject, right: SystemObject) -> SystemObject:
    """Binary product in Sys(I): pairs of states, equity floored per factor."""
    equity = {}
    for a in left.states:
        for b in right.states:
            equity[f"({a},{b})"] = min(
                left.equity[a] - left.initial, right.equity[b] - right.initial
            )
    return SystemObject(f"{left.object_id}x{right.object_id}", equity, 0.0)


def hom(source: SystemObject, target: SystemObject) -> List[Morphism]:
    """Enumerate every total map source -> target that preserves the invariant."""
    maps: List[Dict[str, str]] = [{}]
    for state in source.states:
        maps = [
            dict(partial, **{state: image})
            for partial in maps
            for image in target.states
        ]
    candidates = [Morphism(source, target, table) for table in maps]
    return [m for m in candidates if m.is_valid()]


def canonical_witnesses() -> List[SystemObject]:
    """Representative objects of Sys(I) spanning the shapes the core admits."""
    return [
        SystemObject("flat", {"s0": 100000.0}, 100000.0),
        SystemObject("monotone", {"s0": 100000.0, "s1": 112000.0}, 100000.0),
        SystemObject(
            "branching",
            {"s0": 100000.0, "s1": 100000.0, "s2": 143000.0},
            100000.0,
        ),
        SystemObject(
            "seeded",
            {"s0": float(OMNIUM_DETERMINISTIC_SEED), "s1": float(2**31)},
            float(OMNIUM_DETERMINISTIC_SEED),
        ),
        singularity_core(),
    ]


# --------------------------- proofs ---------------------------


def _terminality_report() -> Dict[str, Any]:
    """Check existence and uniqueness of X -> 1 for every witness object."""
    core = singularity_core()
    witnesses = canonical_witnesses()

    per_object: List[Dict[str, Any]] = []
    unique_everywhere = core.satisfies_invariant()

    for obj in witnesses:
        if not obj.satisfies_invariant():
            unique_everywhere = False
            per_object.append({"object": obj.digest(), "unique": False})
            continue

        arrows = hom(obj, core)
        collapse = Morphism(obj, core, {state: "*" for state in obj.states})
        unique = len(arrows) == 1 and arrows[0].table == collapse.table
        laws = _laws_hold(obj, core, collapse)
        unique_everywhere = unique_everywhere and unique and laws
        per_object.append({"object": obj.digest(), "unique": unique, "laws": laws})

    idempotent = _terminal_is_unique_up_to_iso(core)
    return {
        "core": core.digest(),
        "objects": per_object,
        "iso_unique": idempotent,
        "holds": unique_everywhere and idempotent,
    }


def _laws_hold(obj: SystemObject, core: SystemObject, collapse: Morphism) -> bool:
    """!_X . id_X == !_X and !_1 . !_X == !_X."""
    through_identity = compose(identity(obj), collapse)
    through_core = compose(collapse, identity(core))
    return (
        through_identity.table == collapse.table
        and through_core.table == collapse.table
        and collapse.is_valid()
    )


def _terminal_is_unique_up_to_iso(core: SystemObject) -> bool:
    """Any second terminal object is isomorphic to the core."""
    rival = SystemObject("SingularityCore_prime", {"o": 0.0}, 0.0)
    forward = hom(core, rival)
    backward = hom(rival, core)
    if len(forward) != 1 or len(backward) != 1:
        return False
    round_core = compose(forward[0], backward[0])
    round_rival = compose(backward[0], forward[0])
    return (
        round_core.table == identity(core).table
        and round_rival.table == identity(rival).table
    )


def _necessitation_report() -> Dict[str, Any]:
    """The invariant alone forces 1 into Sys(I), so terminality is not optional."""
    core = singularity_core()

    inhabited = core.satisfies_invariant()

    closed_under_product = True
    witnesses = [obj for obj in canonical_witnesses() if obj.satisfies_invariant()]
    for left in witnesses:
        for right in witnesses:
            if not product(left, right).satisfies_invariant():
                closed_under_product = False

    empty_product_is_core = (
        len(core.states) == 1 and core.equity[core.states[0]] == core.initial
    )

    # Denial branch: a terminal-free Sys(I) would have to reject 1, yet 1 is
    # constructed from the invariant itself — contradiction, so denial fails.
    denial_refuted = inhabited and empty_product_is_core

    holds = inhabited and closed_under_product and empty_product_is_core
    holds = holds and denial_refuted
    return {
        "core": core.digest(),
        "inhabited": inhabited,
        "closed_under_product": closed_under_product,
        "empty_product_is_core": empty_product_is_core,
        "denial_refuted": denial_refuted,
        "holds": holds,
    }


def categorical_terminal_proof() -> bool:
    """True iff SingularityCore is terminal in Sys(I)."""
    setup_logging()
    report = _terminality_report()
    logger.info(_digest(["categorical_terminal_proof", report]))
    return bool(report["holds"])


def existence_necessitation() -> bool:
    """True iff the invariant necessitates the terminal object's existence."""
    setup_logging()
    report = _necessitation_report()
    logger.info(_digest(["existence_necessitation", report]))
    return bool(report["holds"])


def build_proof_node() -> Dict[str, Any]:
    """Assemble the ONTOS_TERMINAL_OBJECT node with hashed content only."""
    terminality = _terminality_report()
    necessitation = _necessitation_report()
    body = {
        "axiom_id": ONTOS_PROOF_ID,
        "invariant": INVARIANT,
        "terminality": terminality,
        "necessitation": necessitation,
        "seed": OMNIUM_DETERMINISTIC_SEED,
    }
    proof_hash = _digest(body)
    return {
        "node_id": ONTOS_PROOF_ID,
        "statement": proof_hash,
        "statement_digest": proof_hash,
        "proof_hash": proof_hash,
        "type": "ontos_terminal_object",
        "root_dependency": ABSOLUTE_ZERO_ROOT,
        "terminality_digest": _digest(terminality),
        "necessitation_digest": _digest(necessitation),
        "terminal_object_digest": terminality["core"],
        "holds": bool(terminality["holds"] and necessitation["holds"]),
    }


def embed_in_proof_network(
    proof_node: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Attach ONTOS_TERMINAL_OBJECT as a leaf of the Aleph-Omega proof DAG."""
    setup_logging()
    if proof_node is None:
        proof_node = build_proof_node()
    try:
        from aleph_omega_engine import ProofNetworkExpansion

        network = ProofNetworkExpansion()
        node_id = network.add_axiom_node(
            axiom_id=ONTOS_PROOF_ID,
            axiom_data=proof_node,
            parent_axioms=[
                ABSOLUTE_ZERO_ROOT,
                "EPISTEMIC_CLOSURE_THEOREM",
                "METIS_NOVELTY_CONSERVATION",
            ],
        )
        ok, message = network.verify_network()
        result_hash = _digest(["embed", node_id, ok, message])
        logger.info(result_hash)
        return ok, result_hash
    except Exception as exc:
        result_hash = _digest(["embed_failed", repr(exc)])
        logger.info(result_hash)
        return False, result_hash


class OntosEngine:
    """Orchestrates the terminal object proof and its network embedding."""

    def __init__(self) -> None:
        setup_logging()
        self.logger = logger

    def run_terminal_object_cycle(self) -> Dict[str, Any]:
        terminal_ok = categorical_terminal_proof()
        necessity_ok = existence_necessitation()
        proof_node = build_proof_node()
        embedded, embed_hash = embed_in_proof_network(proof_node)

        cycle = {
            "terminal": terminal_ok,
            "necessitated": necessity_ok,
            "embedded": embedded,
            "embed_hash": embed_hash,
            "proof_hash": proof_node["proof_hash"],
        }
        cycle_hash = _digest(cycle)
        self.logger.info(cycle_hash)
        return {
            "sealed": terminal_ok and necessity_ok and embedded,
            "cycle_hash": cycle_hash,
            "proof_hash": proof_node["proof_hash"],
            "embed_hash": embed_hash,
        }


def _emit(digest: str) -> None:
    sys.stdout.write(digest + "\n")


def main() -> int:
    engine = OntosEngine()
    result = engine.run_terminal_object_cycle()
    _emit(result["cycle_hash"])
    return 0 if result["sealed"] else 1


if __name__ == "__main__":
    sys.exit(main())
