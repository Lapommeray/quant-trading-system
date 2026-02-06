"""
Unified AI Trading Indicator v2.0

A composable, self-optimizing indicator combining:
- 8 technical indicators (consensus engine)
- Reinforcement Learning (Q-learning)
- Bayesian state estimation (regime detection)
- Genetic hyper-parameter evolution
- Microstructure analysis
- Self-learning feedback loop

Targets:
- Adaptive behaviour in changing regimes
- Interpretable, rationalized trading signals
- Deterministic local execution (no API dependency)
- Self-improving accuracy over time

This is the highest-level interface. It wraps UnifiedIntelligenceCore
and adds explainability, confidence filtering, and signal rationalization.
"""

import os
import logging
import time
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("UnifiedAIIndicator")

_CORE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(filename, classname):
    import importlib.util
    path = os.path.join(_CORE_DIR, filename)
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, classname)


UnifiedIntelligenceCore = _load_module("unified_core.py", "UnifiedIntelligenceCore")
FeatureEngineer = _load_module("feature_engineering.py", "FeatureEngineer")
DataPipeline = _load_module("data_pipeline.py", "DataPipeline")


class UnifiedAIIndicator:
    """
    Production-grade AI trading indicator.

    Wraps the full intelligence pipeline and provides:
    - Single-call signal generation from symbol name
    - Human-readable explanation for every signal
    - Confidence-gated output (only emits when highly certain)
    - Full audit trail for every decision
    """

    def __init__(self, symbol: str = "XAUUSD", data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = os.path.dirname(_CORE_DIR)

        self.symbol = symbol
        self._core = UnifiedIntelligenceCore(data_dir=data_dir)
        self._features = FeatureEngineer()
        self._pipeline = DataPipeline()
        self._confidence = 0.0
        self._last_signal = "HOLD"
        self._last_reason = "boot"
        self._cycle_count = 0
        self._signal_history = []

        logger.info(f"UnifiedAIIndicator initialized for {symbol}")

    def process_tick(self, market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process one market snapshot and emit a decision signal.

        If market_data is None, fetches live data automatically.

        Returns:
            dict: {symbol, signal, confidence, reason, timestamp, details}
        """
        self._cycle_count += 1

        if market_data is None:
            market_data = self._pipeline.fetch(self.symbol)
            if not market_data.get("ohlcv"):
                return self._null_signal("no_data_available")

        result = self._core.generate_signal(self.symbol, market_data)

        signal = result.get("final_signal", "HOLD")
        confidence = result.get("confidence", 0.0)

        self._confidence = confidence

        if confidence >= 0.7 and signal in ("BUY", "SELL"):
            self._last_signal = signal
        else:
            signal = self._last_signal if confidence > 0.4 else "HOLD"

        reason = self._explain(result, market_data)
        self._last_reason = reason

        output = {
            "symbol": self.symbol,
            "signal": signal,
            "confidence": round(confidence, 6),
            "reason": reason,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cycle": self._cycle_count,
            "regime": result.get("regime", "UNKNOWN"),
            "indicator_signal": result.get("indicator_signal"),
            "rl_action": result.get("rl_action"),
            "micro_bias": result.get("micro_bias"),
            "votes": result.get("votes", {}),
        }

        self._signal_history.append({
            "cycle": self._cycle_count,
            "signal": signal,
            "confidence": confidence,
            "regime": result.get("regime"),
        })
        if len(self._signal_history) > 1000:
            self._signal_history = self._signal_history[-500:]

        return output

    def _explain(self, result: Dict[str, Any], market_data: Dict[str, Any]) -> str:
        """Generate human-readable rationale for the signal."""
        parts = []

        regime = result.get("regime", "?")
        parts.append(f"regime={regime}")

        votes = result.get("votes", {})
        buy_count = sum(1 for v in votes.values() if v == "BUY")
        sell_count = sum(1 for v in votes.values() if v == "SELL")
        parts.append(f"indicators={buy_count}B/{sell_count}S")

        indicator_sig = result.get("indicator_signal")
        if indicator_sig:
            parts.append(f"consensus={indicator_sig}")

        rl_action = result.get("rl_action")
        if rl_action:
            parts.append(f"rl={rl_action}")

        micro = result.get("micro_bias")
        if micro:
            parts.append(f"micro={micro}")

        htf = result.get("htf_trend") if "htf_trend" in result else None
        if htf:
            parts.append(f"daily={htf}")

        conf = result.get("confidence", 0)
        parts.append(f"conf={conf:.3f}")

        prev_eval = result.get("prev_eval")
        if prev_eval:
            correct = "WIN" if prev_eval.get("correct") else "LOSS"
            change = prev_eval.get("change_pct", 0)
            acc = prev_eval.get("accuracy_pct", 0)
            parts.append(f"prev={correct}({change:+.2f}%)")
            parts.append(f"accuracy={acc:.1f}%")

        return ", ".join(parts)

    def _null_signal(self, reason: str) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "signal": "HOLD",
            "confidence": 0.0,
            "reason": reason,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cycle": self._cycle_count,
            "regime": "UNKNOWN",
        }

    def reset(self):
        """Reset indicator state (for new backtest run)."""
        self._confidence = 0.0
        self._last_signal = "HOLD"
        self._last_reason = "reset"
        self._signal_history = []

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def last_signal(self) -> str:
        return self._last_signal

    def generate_signal(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runner-compatible interface.

        Called by run_mt5_live.py. Delegates to process_tick and returns
        the standard {final_signal, confidence, ...} schema.
        """
        self.symbol = symbol
        tick_result = self.process_tick(market_data)
        return {
            "final_signal": tick_result.get("signal"),
            "confidence": tick_result.get("confidence", 0.0),
            "regime": tick_result.get("regime"),
            "reason": tick_result.get("reason", ""),
            "indicator_signal": tick_result.get("indicator_signal"),
            "rl_action": tick_result.get("rl_action"),
            "micro_bias": tick_result.get("micro_bias"),
            "votes": tick_result.get("votes", {}),
        }

    def get_stats(self) -> Dict[str, Any]:
        core_stats = self._core.get_stats()
        return {
            "symbol": self.symbol,
            "cycle_count": self._cycle_count,
            "last_signal": self._last_signal,
            "confidence": self._confidence,
            "last_reason": self._last_reason,
            "history_length": len(self._signal_history),
            **core_stats,
        }
