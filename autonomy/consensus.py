"""
Consensus engine - weighted voting across modules.

Pure Python, no QuantConnect dependency. Fails closed if no modules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ConsensusResult:
    symbol: str
    final_signal: str | None  # BUY, SELL, None
    confidence: float
    weighted_confidence: float
    votes: Dict[str, float]
    module_results: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "final_signal": self.final_signal,
            "confidence": self.confidence,
            "weighted_confidence": self.weighted_confidence,
            "votes": self.votes,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
        }


class ConsensusEngine:
    """Weighted voting consensus. No QC dependency."""

    def __init__(self, min_weighted_confidence: float = 0.30):
        self.min_weighted_confidence = min_weighted_confidence

    def compute(
        self,
        symbol: str,
        directions: Dict[str, str],
        confidences: Dict[str, float],
        weights: Dict[str, float],
        module_results: Dict[str, Any] | None = None,
        latency_ms: float = 0.0,
    ) -> ConsensusResult:
        votes: Dict[str, float] = {"BUY": 0.0, "SELL": 0.0, "NEUTRAL": 0.0}

        for mod_name, direction in directions.items():
            dir_up = direction.upper()
            if dir_up not in votes:
                continue
            w = weights.get(mod_name, 0.0) * confidences.get(mod_name, 0.0)
            votes[dir_up] += w

        total_conf = sum(confidences.values()) / len(confidences) if confidences else 0.0
        weighted_conf = sum(votes.values())

        final = max(votes, key=lambda k: votes[k]) if weighted_conf > 0 else "NEUTRAL"
        if final == "NEUTRAL" or weighted_conf < self.min_weighted_confidence:
            final_signal = None
        else:
            final_signal = final

        return ConsensusResult(
            symbol=symbol,
            final_signal=final_signal,
            confidence=total_conf,
            weighted_confidence=weighted_conf,
            votes=votes,
            module_results=module_results or {},
            latency_ms=latency_ms,
        )
