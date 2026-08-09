#!/usr/bin/env python3
"""Axiom Zero Engine — profit as a propositional tautology.

Scope. The object of proof is the *internal ledger* of the proof network: a
formal accounting whose successor rule is fixed by the axioms below. Nothing
here observes, predicts, or claims anything about an external market; the
"profit" proven tautological is the ledger's own monotone quantity, and market
atoms appear only to be shown irrelevant to it.

Axiom Zero (A0). The ledger is closed: no proposition outside the axiom set
may enter a derivation.
A1. The deductive increment delta is non-negative.
A2. Deduction is monotone: a derived theorem is never retracted.
D.  Definitional: equity_nondecreasing <-> (delta_nonneg and deduction_monotone).

Theorem (tautological profit). D -> ((delta_nonneg and deduction_monotone) ->
equity_nondecreasing) is true under every valuation of every atom, market atoms
included, so its truth value cannot depend on external data.

Corollary (the invariant as theorem, not assumption). forall t. Equity_t >=
Equity_0 follows by induction on t: base is reflexivity, step is A1.

Emission policy: every log record and every stdout line is a hex digest.
"""

import hashlib
import json
import logging
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

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

AXIOM_ZERO_ID = "AXIOM_ZERO_TAUTOLOGY"
ABSOLUTE_ZERO_ROOT = "AbsoluteZero_forall_t_Equity_t_ge_Equity_0"
INVARIANT = "forall t. Equity_t >= Equity_0"
AXIOM_ZERO_LOG = REPO_ROOT / "axiom_zero.log"

# Nodes this one subsumes: their statements are consequences of the tautology.
SUPERSEDED_NODES: Tuple[str, ...] = (
    ABSOLUTE_ZERO_ROOT,
    "AXIOM_GENERATION",
    "EPISTEMIC_CLOSURE_THEOREM",
    "METIS_NOVELTY_CONSERVATION",
    "OMEGA_SINGULARITY_NEXUS",
    "ABSOLUTE_SINGULARITY_CORE",
    "ONTOS_TERMINAL_OBJECT",
    "AUTARKES_INERT_LOGICAL_CONSTANT",
)

# Atoms internal to the ledger; the axioms speak only about these.
LEDGER_ATOMS: Tuple[str, ...] = (
    "ledger_closed",
    "delta_nonneg",
    "deduction_monotone",
    "equity_nondecreasing",
)

# Atoms naming external observations. They are in the signature solely so the
# tautology check ranges over them and their irrelevance becomes a theorem.
MARKET_ATOMS: Tuple[str, ...] = (
    "price_up",
    "price_down",
    "liquidity_present",
    "counterparty_solvent",
)

ATOMS: Tuple[str, ...] = LEDGER_ATOMS + MARKET_ATOMS

logger = logging.getLogger("AxiomZero")

Formula = Any
Valuation = Dict[str, bool]


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
    file_handler = logging.FileHandler(AXIOM_ZERO_LOG)
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


# ------------------------------ propositional core ------------------------------


def NOT(inner: Formula) -> Formula:
    return ("not", inner)


def AND(*parts: Formula) -> Formula:
    return ("and",) + parts


def OR(*parts: Formula) -> Formula:
    return ("or",) + parts


def IMPLIES(antecedent: Formula, consequent: Formula) -> Formula:
    return ("implies", antecedent, consequent)


def IFF(left: Formula, right: Formula) -> Formula:
    return ("iff", left, right)


def evaluate(formula: Formula, valuation: Valuation) -> bool:
    """Classical two-valued semantics; unknown atoms are a hard error."""
    if isinstance(formula, str):
        if formula not in valuation:
            raise KeyError(formula)
        return valuation[formula]
    connective, operands = formula[0], formula[1:]
    if connective == "not":
        return not evaluate(operands[0], valuation)
    if connective == "and":
        return all(evaluate(part, valuation) for part in operands)
    if connective == "or":
        return any(evaluate(part, valuation) for part in operands)
    if connective == "implies":
        return (not evaluate(operands[0], valuation)) or evaluate(
            operands[1], valuation
        )
    if connective == "iff":
        return evaluate(operands[0], valuation) == evaluate(operands[1], valuation)
    raise ValueError(connective)


def valuations(atoms: Sequence[str] = ATOMS) -> Iterator[Valuation]:
    """Every assignment over the signature — the whole truth table."""
    total = 1 << len(atoms)
    for mask in range(total):
        yield {atom: bool(mask >> index & 1) for index, atom in enumerate(atoms)}


def is_tautology(formula: Formula, atoms: Sequence[str] = ATOMS) -> bool:
    """True iff the formula holds under every valuation of the signature."""
    return all(evaluate(formula, valuation) for valuation in valuations(atoms))


def atoms_of(formula: Formula) -> List[str]:
    """The atoms actually occurring in the formula, sorted."""
    if isinstance(formula, str):
        return [formula]
    found: List[str] = []
    for operand in formula[1:]:
        for atom in atoms_of(operand):
            if atom not in found:
                found.append(atom)
    return sorted(found)


# --------------------------------- the axioms ----------------------------------

AXIOM_ZERO_CLOSURE: Formula = "ledger_closed"
AXIOM_DELTA_NONNEG: Formula = "delta_nonneg"
AXIOM_MONOTONE_DEDUCTION: Formula = "deduction_monotone"
DEFINITION_OF_PROFIT: Formula = IFF(
    "equity_nondecreasing", AND("delta_nonneg", "deduction_monotone")
)

# Profit: whenever the increment is non-negative and deduction is monotone,
# equity does not decrease. Under D this is valid, not merely satisfiable.
PROFIT: Formula = IMPLIES(
    AND("delta_nonneg", "deduction_monotone"), "equity_nondecreasing"
)

TAUTOLOGICAL_PROFIT: Formula = IMPLIES(DEFINITION_OF_PROFIT, PROFIT)

# The invariant, restated propositionally: closure plus a non-negative
# increment yields non-decreasing equity at every step.
INVARIANT_THEOREM: Formula = IMPLIES(
    AND(DEFINITION_OF_PROFIT, AXIOM_ZERO_CLOSURE, AXIOM_DELTA_NONNEG),
    IMPLIES(AXIOM_MONOTONE_DEDUCTION, "equity_nondecreasing"),
)


def _deductive_delta() -> Fraction:
    """The increment per deduction step, fixed by the seed — exact, non-negative."""
    magnitude = OMNIUM_DETERMINISTIC_SEED % 1000 + 1
    return Fraction(magnitude, 10**6)


# --------------------------------- the reports ---------------------------------


def _tautology_report() -> Dict[str, Any]:
    """Validity of the profit theorem, plus its independence from market atoms."""
    valid = is_tautology(TAUTOLOGICAL_PROFIT)

    # Independence: flipping any market atom leaves the truth value untouched.
    market_free_signature = sorted(
        atom for atom in atoms_of(TAUTOLOGICAL_PROFIT) if atom in MARKET_ATOMS
    )
    flip_invariant = True
    for valuation in valuations():
        baseline = evaluate(TAUTOLOGICAL_PROFIT, valuation)
        for atom in MARKET_ATOMS:
            flipped = dict(valuation)
            flipped[atom] = not flipped[atom]
            if evaluate(TAUTOLOGICAL_PROFIT, flipped) != baseline:
                flip_invariant = False

    # The bare profit implication is *not* valid without D: the theorem has
    # content, it is not a vacuous truth.
    non_vacuous = not is_tautology(PROFIT)

    return {
        "valid": valid,
        "signature_size": len(ATOMS),
        "rows": 1 << len(ATOMS),
        "market_atoms_in_theorem": market_free_signature,
        "flip_invariant": flip_invariant,
        "non_vacuous": non_vacuous,
        "formula_digest": _digest(TAUTOLOGICAL_PROFIT),
        "holds": bool(
            valid and flip_invariant and non_vacuous and not market_free_signature
        ),
    }


def _induction_report(horizon: int = 512) -> Dict[str, Any]:
    """The invariant as a theorem: base by reflexivity, step by A1."""
    delta = _deductive_delta()
    initial = Fraction(0)

    base_case = initial >= initial
    step_case = delta >= 0

    equity = initial
    monotone = True
    for _ in range(horizon):
        successor = equity + delta
        if successor < equity or successor < initial:
            monotone = False
        equity = successor

    theorem_valid = is_tautology(INVARIANT_THEOREM)

    return {
        "invariant": INVARIANT,
        "base_case": base_case,
        "step_case": step_case,
        "model_checked_horizon": horizon,
        "monotone": monotone,
        "theorem_valid": theorem_valid,
        "delta": str(delta),
        "terminal_equity": str(equity),
        "holds": bool(base_case and step_case and monotone and theorem_valid),
    }


def tautological_profit_proof() -> bool:
    """True iff profit is valid in every valuation and market-independent."""
    setup_logging()
    report = _tautology_report()
    logger.info(_digest(["tautological_profit_proof", report]))
    return bool(report["holds"])


def invariant_as_theorem() -> bool:
    """True iff forall t. Equity_t >= Equity_0 is derived, not assumed."""
    setup_logging()
    report = _induction_report()
    logger.info(_digest(["invariant_as_theorem", report]))
    return bool(report["holds"])


class LedgerState(NamedTuple):
    """(step, equity) — the ledger is nothing but a deduction counter."""

    step: int
    equity: Fraction

    def digest(self) -> str:
        return _digest([self.step, str(self.equity)])


def self_sustaining_profit_generation(
    initial: Fraction = Fraction(0),
) -> Iterator[LedgerState]:
    """Perpetual ledger: each step is one deduction, each deduction adds delta.

    Zero external input — the increment comes from the axioms via the seed, so
    the sequence is non-decreasing by construction and unbounded in length.
    """
    delta = _deductive_delta()
    state = LedgerState(0, initial)
    while True:
        yield state
        state = LedgerState(state.step + 1, state.equity + delta)


def equity_at(step: int, initial: Fraction = Fraction(0)) -> Fraction:
    """Closed form of the generator: Equity_t = Equity_0 + t * delta."""
    if step < 0:
        raise ValueError("step is a natural number")
    return initial + step * _deductive_delta()


def build_proof_node() -> Dict[str, Any]:
    """Assemble the AXIOM_ZERO_TAUTOLOGY node with hashed content only."""
    tautology = _tautology_report()
    induction = _induction_report()
    body = {
        "axiom_id": AXIOM_ZERO_ID,
        "invariant": INVARIANT,
        "tautology": tautology,
        "induction": induction,
        "supersedes": list(SUPERSEDED_NODES),
        "seed": OMNIUM_DETERMINISTIC_SEED,
    }
    proof_hash = _digest(body)
    return {
        "node_id": AXIOM_ZERO_ID,
        "statement": proof_hash,
        "statement_digest": proof_hash,
        "proof_hash": proof_hash,
        "type": "axiom_zero_tautology",
        "root_dependency": ABSOLUTE_ZERO_ROOT,
        "supersedes": list(SUPERSEDED_NODES),
        "tautology_digest": _digest(tautology),
        "induction_digest": _digest(induction),
        "holds": bool(tautology["holds"] and induction["holds"]),
    }


def embed_in_proof_network(
    proof_node: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Attach AXIOM_ZERO_TAUTOLOGY above every prior node of the proof DAG."""
    setup_logging()
    if proof_node is None:
        proof_node = build_proof_node()
    try:
        from aleph_omega_engine import ProofNetworkExpansion

        network = ProofNetworkExpansion()
        node_id = network.add_axiom_node(
            axiom_id=AXIOM_ZERO_ID,
            axiom_data=proof_node,
            parent_axioms=list(SUPERSEDED_NODES),
        )
        ok, message = network.verify_network()
        result_hash = _digest(["embed", node_id, ok, message])
        logger.info(result_hash)
        return ok, result_hash
    except Exception as exc:
        result_hash = _digest(["embed_failed", repr(exc)])
        logger.info(result_hash)
        return False, result_hash


class AxiomZeroEngine:
    """Orchestrates the tautology, the derived invariant, and the embedding."""

    def __init__(self) -> None:
        setup_logging()
        self.logger = logger

    def run_tautology_cycle(self, steps: int = 64) -> Dict[str, Any]:
        profit_ok = tautological_profit_proof()
        invariant_ok = invariant_as_theorem()

        ledger = self_sustaining_profit_generation()
        states = [next(ledger) for _ in range(steps)]
        generated_ok = all(
            later.equity >= earlier.equity for earlier, later in zip(states, states[1:])
        )
        closed_form_ok = all(state.equity == equity_at(state.step) for state in states)

        proof_node = build_proof_node()
        embedded, embed_hash = embed_in_proof_network(proof_node)

        cycle = {
            "tautological": profit_ok,
            "invariant_derived": invariant_ok,
            "generated_monotone": generated_ok,
            "closed_form_agrees": closed_form_ok,
            "embedded": embedded,
            "embed_hash": embed_hash,
            "proof_hash": proof_node["proof_hash"],
            "terminal_state": states[-1].digest(),
        }
        cycle_hash = _digest(cycle)
        self.logger.info(cycle_hash)
        return {
            "sealed": bool(
                profit_ok
                and invariant_ok
                and generated_ok
                and closed_form_ok
                and embedded
            ),
            "cycle_hash": cycle_hash,
            "proof_hash": proof_node["proof_hash"],
            "embed_hash": embed_hash,
        }


def _emit(digest: str) -> None:
    sys.stdout.write(digest + "\n")


def main() -> int:
    engine = AxiomZeroEngine()
    result = engine.run_tautology_cycle()
    _emit(result["cycle_hash"])
    return 0 if result["sealed"] else 1


if __name__ == "__main__":
    sys.exit(main())
