"""
State Buffer - Short-Term Market Memory

Maintains a sliding window of recent market states across ticks/bars,
providing EMA-smoothed memory for the decision engine.

This gives the indicator "short-term memory" — awareness of what happened
in the last N bars without needing a full LSTM. Uses exponential moving
averages and rolling statistics to capture momentum drift, regime shifts,
and volatility clustering.

Pure Python/numpy. No external AI.
"""

import logging
from collections import deque
from typing import Dict, Any, Optional, List

import numpy as np

logger = logging.getLogger("StateBuffer")


class StateBuffer:
    """
    EMA-based short-term market memory.

    Tracks rolling statistics of features, price action, and signals
    to provide context-aware inputs to the decision engine.
    """

    def __init__(self, max_bars: int = 500, ema_spans: Optional[List[int]] = None):
        self.max_bars = max_bars
        self.ema_spans = ema_spans or [5, 10, 20, 50]

        self._price_history: deque = deque(maxlen=max_bars)
        self._feature_history: deque = deque(maxlen=max_bars)
        self._signal_history: deque = deque(maxlen=max_bars)
        self._confidence_history: deque = deque(maxlen=max_bars)

        self._ema_values: Dict[str, Dict[int, float]] = {}
        self._regime_memory: deque = deque(maxlen=100)

        logger.info(f"StateBuffer initialized: max_bars={max_bars}, ema_spans={self.ema_spans}")

    def record(
        self,
        price: float,
        features: Dict[str, float],
        signal: str = "HOLD",
        confidence: float = 0.0,
        regime: str = "UNKNOWN",
    ):
        """Record a new observation into the buffer."""
        self._price_history.append(price)
        self._feature_history.append(features)
        self._signal_history.append(signal)
        self._confidence_history.append(confidence)
        self._regime_memory.append(regime)

        for key, value in features.items():
            if key not in self._ema_values:
                self._ema_values[key] = {}
            for span in self.ema_spans:
                alpha = 2.0 / (span + 1)
                prev = self._ema_values[key].get(span, value)
                self._ema_values[key][span] = alpha * value + (1 - alpha) * prev

    def get_memory_features(self) -> Dict[str, float]:
        """
        Extract memory-derived features for the decision engine.

        Returns a dict of features derived from the buffer's rolling state.
        """
        result: Dict[str, float] = {}

        prices = list(self._price_history)
        if len(prices) >= 5:
            arr = np.array(prices[-50:], dtype=np.float64)
            result["mem_price_mean"] = float(np.mean(arr))
            result["mem_price_std"] = float(np.std(arr))
            result["mem_price_zscore"] = float(
                (arr[-1] - np.mean(arr)) / np.std(arr)
            ) if np.std(arr) > 0 else 0.0

            if len(prices) >= 20:
                short = np.mean(arr[-5:])
                long = np.mean(arr[-20:])
                result["mem_momentum_drift"] = float(
                    (short - long) / long
                ) if long != 0 else 0.0
            else:
                result["mem_momentum_drift"] = 0.0

            returns = np.diff(arr) / arr[:-1]
            returns = returns[~np.isnan(returns)]
            if len(returns) >= 5:
                result["mem_vol_cluster"] = float(np.std(returns[-5:]) / (np.std(returns) + 1e-10))
            else:
                result["mem_vol_cluster"] = 1.0
        else:
            result["mem_price_mean"] = 0.0
            result["mem_price_std"] = 0.0
            result["mem_price_zscore"] = 0.0
            result["mem_momentum_drift"] = 0.0
            result["mem_vol_cluster"] = 1.0

        for key in ["momentum", "depth_imbalance", "bar_volatility"]:
            for span in self.ema_spans:
                ema_val = self._ema_values.get(key, {}).get(span, 0.0)
                result[f"mem_{key}_ema{span}"] = float(ema_val)

        confs = list(self._confidence_history)
        if len(confs) >= 5:
            result["mem_avg_confidence"] = float(np.mean(confs[-10:]))
            result["mem_confidence_trend"] = float(
                np.mean(confs[-3:]) - np.mean(confs[-10:])
            )
        else:
            result["mem_avg_confidence"] = 0.5
            result["mem_confidence_trend"] = 0.0

        signals = list(self._signal_history)
        if len(signals) >= 5:
            recent = signals[-10:]
            result["mem_buy_ratio"] = sum(1 for s in recent if s == "BUY") / len(recent)
            result["mem_sell_ratio"] = sum(1 for s in recent if s == "SELL") / len(recent)
        else:
            result["mem_buy_ratio"] = 0.0
            result["mem_sell_ratio"] = 0.0

        regimes = list(self._regime_memory)
        if len(regimes) >= 5:
            recent_r = regimes[-10:]
            result["mem_regime_stability"] = max(
                recent_r.count("BULL"),
                recent_r.count("BEAR"),
                recent_r.count("RANGE"),
            ) / len(recent_r)
        else:
            result["mem_regime_stability"] = 0.0

        return result

    def get_signal_consistency(self, lookback: int = 10) -> float:
        """
        How consistent recent signals have been (0=chaotic, 1=perfectly stable).
        """
        signals = list(self._signal_history)
        if len(signals) < 3:
            return 0.0

        recent = signals[-lookback:]
        if not recent:
            return 0.0

        most_common_count = max(
            recent.count("BUY"),
            recent.count("SELL"),
            recent.count("HOLD"),
        )
        return most_common_count / len(recent)

    def get_regime_distribution(self, lookback: int = 50) -> Dict[str, float]:
        """Probability distribution of regimes over recent history."""
        regimes = list(self._regime_memory)
        if not regimes:
            return {"BULL": 0.33, "BEAR": 0.33, "RANGE": 0.34}

        recent = regimes[-lookback:]
        n = len(recent)
        return {
            "BULL": recent.count("BULL") / n,
            "BEAR": recent.count("BEAR") / n,
            "RANGE": recent.count("RANGE") / n,
        }

    @property
    def bar_count(self) -> int:
        return len(self._price_history)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "bars_stored": len(self._price_history),
            "features_tracked": len(self._ema_values),
            "signal_consistency": round(self.get_signal_consistency(), 3),
            "regime_distribution": self.get_regime_distribution(),
        }
