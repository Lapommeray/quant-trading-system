"""Conservative built-in modules for the canonical organism.

The historical repository contains many optional research modules, several of
which require heavyweight or legacy dependencies.  These small modules are
always importable and provide a real interconnection baseline: momentum
proposes direction, while the regime risk gate can veto participation during
abnormal volatility.  They also implement the common learning/adaptive hooks
so the same self-coding coordinator can service every registered module.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Mapping

from .market import MarketRegime, extract_closes
from .organism import BaseModule, ModuleResult, register_module


def _returns(closes: list[float]) -> list[float]:
    return [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
        if closes[i - 1] > 0 and closes[i] > 0
    ]


@register_module(name="momentum_alpha")
class MomentumAlphaModule(BaseModule):
    """Price-only baseline alpha with bounded confidence."""

    module_name = "momentum_alpha"
    category = "alpha"
    version = "2.0.0"
    regime_affinity = {
        "low_bull": 1.15,
        "low_bear": 1.15,
        "high_bull": 0.85,
        "high_bear": 0.85,
        "low_range": 0.75,
        "high_range": 0.60,
        "default": 1.0,
    }

    def __init__(self, config=None, event_bus=None):
        super().__init__(config=config, event_bus=event_bus)
        self.current_regime = "unknown"
        self.last_outcome: Dict[str, Any] = {}

    def generate_signal(
        self, symbol: str, history_data: Dict[str, Any]
    ) -> ModuleResult:
        closes = extract_closes(history_data)
        adaptive = self.config.get("adaptive", {})
        lookback = int(adaptive.get("lookback", 20) or 20)
        lookback = max(8, min(100, lookback))
        if len(closes) < lookback:
            return ModuleResult(
                self.module_name, "NEUTRAL", 0.0, {"reason": "insufficient_history"}
            )

        recent = closes[-lookback:]
        half = max(3, lookback // 2)
        fast = statistics.fmean(recent[-half:])
        slow = statistics.fmean(recent)
        edge = (fast - slow) / max(abs(slow), 1e-12)
        volatility = statistics.pstdev(_returns(recent)) if len(recent) > 2 else 0.0
        # A small edge should remain neutral rather than being forced into a
        # trade.  Confidence is bounded and regime/risk tuning can only make
        # it more selective.
        threshold = max(0.0005, float(adaptive.get("confidence_floor", 0.60)) - 0.58)
        signal = "NEUTRAL"
        if edge > threshold:
            signal = "BUY"
        elif edge < -threshold:
            signal = "SELL"
        confidence = (
            min(0.95, 0.50 + min(0.40, abs(edge) * 18.0))
            if signal != "NEUTRAL"
            else 0.50
        )
        if volatility > 0.02:
            confidence *= 0.85
        return ModuleResult(
            module_name=self.module_name,
            signal=signal,
            confidence=max(0.0, min(1.0, confidence)),
            features={
                "fast_mean": fast,
                "slow_mean": slow,
                "edge": edge,
                "volatility": volatility,
                "lookback": lookback,
                "regime": self.current_regime,
            },
        )

    def on_event(self, event_type: str, payload: Dict[str, Any]):
        if event_type == "MARKET_REGIME":
            self.current_regime = str(payload.get("regime", "unknown"))

    def learn_from_outcome(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        self.last_outcome = dict(outcome)
        return {
            "module": self.module_name,
            "learned": True,
            "reward": outcome.get("reward"),
        }

    def apply_adaptive_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        applied = super().apply_adaptive_parameters(parameters)
        if "lookback" in applied:
            applied["lookback"] = int(max(8, min(100, applied["lookback"])))
            self.config["adaptive"]["lookback"] = applied["lookback"]
        return applied

    def self_improve(self, performance_history):
        return {
            "module": self.module_name,
            "improved": bool(self.last_outcome),
            "mode": "bounded_parameter_tuning",
        }


@register_module(name="regime_risk_gate")
class RegimeRiskGateModule(BaseModule):
    """Veto-style module that communicates volatility risk through consensus."""

    module_name = "regime_risk_gate"
    category = "risk_gate"
    version = "1.1.0"
    regime_affinity = {
        "high_range": 1.20,
        "high_bull": 1.15,
        "high_bear": 1.15,
        "default": 1.0,
    }

    def __init__(self, config=None, event_bus=None):
        super().__init__(config=config, event_bus=event_bus)
        self.regime = MarketRegime()

    def generate_signal(
        self, symbol: str, history_data: Dict[str, Any]
    ) -> ModuleResult:
        closes = extract_closes(history_data)
        returns = _returns(closes[-40:])
        volatility = statistics.pstdev(returns) if len(returns) > 1 else 0.0
        high_volatility = volatility >= 0.02 or self.regime.volatility == "high"
        # NEUTRAL is a deliberate veto vote.  It never generates an order on
        # its own, and a separate risk engine still remains authoritative.
        # A calm-market neutral vote is intentionally low weight; in a
        # high-volatility regime it becomes a strong veto vote.
        confidence = 0.95 if high_volatility else 0.45
        return ModuleResult(
            module_name=self.module_name,
            signal="NEUTRAL",
            confidence=confidence,
            features={
                "volatility": volatility,
                "high_volatility": high_volatility,
                "regime": self.regime.label,
            },
        )

    def on_event(self, event_type: str, payload: Dict[str, Any]):
        if event_type == "MARKET_REGIME":
            data = payload.get("regime_data", {})
            if isinstance(data, Mapping):
                self.regime = MarketRegime(
                    label=str(data.get("label", payload.get("regime", "unknown"))),
                    direction=str(data.get("direction", "unknown")),
                    volatility=str(data.get("volatility", "unknown")),
                    trend_score=float(data.get("trend_score", 0.0) or 0.0),
                    volatility_score=float(data.get("volatility_score", 0.0) or 0.0),
                    sample_size=int(data.get("sample_size", 0) or 0),
                    timestamp=float(data.get("timestamp", 0.0) or 0.0),
                )

    def self_improve(self, performance_history):
        return {"module": self.module_name, "improved": False, "mode": "veto_only"}


__all__ = ["MomentumAlphaModule", "RegimeRiskGateModule"]
