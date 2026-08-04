"""Small dependency-free market-regime detector.

The detector is intentionally conservative.  It does not predict prices and it
never changes leverage, order size, or safety limits.  Its only responsibility
is to describe the recent observation window so modules can tune their
confidence and the organism can compare performance under similar conditions.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

_CLOSE_KEYS = ("close", "Close", "last", "price", "Price")


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _from_frame(data: Any) -> List[float]:
    """Read a pandas-like frame without importing pandas."""

    try:
        columns = list(data.columns)
    except Exception:
        return []
    column = next((name for name in _CLOSE_KEYS if name in columns), None)
    if column is None:
        lower = {str(name).lower(): name for name in columns}
        column = next(
            (lower[name.lower()] for name in _CLOSE_KEYS if name.lower() in lower), None
        )
    if column is None:
        return []
    try:
        values = data[column].tolist()
    except Exception:
        try:
            values = list(data[column])
        except Exception:
            return []
    return [
        number
        for value in values
        if (number := _number(value)) is not None and number > 0
    ]


def extract_closes(data: Any) -> List[float]:
    """Extract close prices from frames, mappings, or simple bar sequences.

    ``history_data`` in this repository is usually a mapping of timeframe to a
    pandas DataFrame.  Supporting plain dictionaries and lists makes the
    organism useful in tests and in lightweight data adapters too.
    """

    if data is None:
        return []
    frame_values = _from_frame(data)
    if frame_values:
        return frame_values
    if isinstance(data, Mapping):
        for key in _CLOSE_KEYS:
            if key in data:
                value = data[key]
                if isinstance(value, (str, bytes)):
                    number = _number(value)
                    return [number] if number is not None and number > 0 else []
                try:
                    values = list(value)
                except TypeError:
                    number = _number(value)
                    return [number] if number is not None and number > 0 else []
                return [
                    number
                    for item in values
                    if (number := _number(item)) is not None and number > 0
                ]

        # Prefer a lower timeframe when several frames are available.  A
        # direct frame may also appear as the value of a mapping.
        preferred = sorted(data.items(), key=lambda item: str(item[0]))
        for _, value in preferred:
            values = extract_closes(value)
            if values:
                return values
        return []
    if isinstance(data, (str, bytes)):
        number = _number(data)
        return [number] if number is not None and number > 0 else []
    try:
        items = list(data)
    except TypeError:
        number = _number(data)
        return [number] if number is not None and number > 0 else []

    closes: List[float] = []
    for item in items:
        if isinstance(item, Mapping):
            value = next((item.get(key) for key in _CLOSE_KEYS if key in item), None)
            number = _number(value)
        else:
            # A scalar sequence is interpreted as close prices.  For OHLCV
            # tuples the fourth value is the conventional close.
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                number = _number(item[3])
            else:
                number = _number(item)
        if number is not None and number > 0:
            closes.append(number)
    return closes


@dataclass(frozen=True)
class MarketRegime:
    """Descriptive snapshot used by adaptive modules."""

    label: str = "unknown"
    direction: str = "unknown"
    volatility: str = "unknown"
    trend_score: float = 0.0
    volatility_score: float = 0.0
    return_mean: float = 0.0
    return_std: float = 0.0
    sample_size: int = 0
    timestamp: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def regime(self) -> str:
        """Alias used by older adapters."""
        return self.label

    @property
    def is_unknown(self) -> bool:
        return self.label == "unknown"

    @property
    def is_high_volatility(self) -> bool:
        return self.volatility == "high"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "direction": self.direction,
            "volatility": self.volatility,
            "trend_score": self.trend_score,
            "volatility_score": self.volatility_score,
            "return_mean": self.return_mean,
            "return_std": self.return_std,
            "sample_size": self.sample_size,
            "timestamp": self.timestamp,
            "metrics": dict(self.metrics),
        }


class MarketRegimeDetector:
    """Detect trend/range and calm/volatile state from recent closes."""

    def __init__(
        self,
        window: int = 50,
        *,
        minimum_samples: int = 10,
        high_volatility_threshold: float = 0.02,
        trend_threshold: float = 0.35,
        clock=None,
    ) -> None:
        self.window = max(5, int(window))
        self.minimum_samples = max(3, int(minimum_samples))
        self.high_volatility_threshold = max(0.000001, float(high_volatility_threshold))
        self.trend_threshold = max(0.01, float(trend_threshold))
        self.clock = clock
        self.current = MarketRegime()
        self.history: List[MarketRegime] = []

    def detect(self, history_data: Any) -> MarketRegime:
        closes = extract_closes(history_data)[-self.window :]
        if len(closes) < self.minimum_samples:
            self.current = MarketRegime(sample_size=len(closes), timestamp=self._now())
            self.history.append(self.current)
            return self.current

        returns = [
            math.log(closes[index] / closes[index - 1])
            for index in range(1, len(closes))
            if closes[index - 1] > 0
        ]
        if not returns:
            self.current = MarketRegime(sample_size=len(closes), timestamp=self._now())
            self.history.append(self.current)
            return self.current

        mean_return = statistics.fmean(returns)
        std_return = statistics.pstdev(returns) if len(returns) > 1 else 0.0
        # Scale the mean by recent noise.  This is a signal-strength measure,
        # not a forecast probability.
        trend_score = mean_return / max(std_return, 1e-9) * math.sqrt(len(returns))
        trend_score = max(-1.0, min(1.0, trend_score))
        volatility_score = max(
            0.0, min(1.0, std_return / self.high_volatility_threshold)
        )

        if trend_score >= self.trend_threshold:
            direction = "bull"
        elif trend_score <= -self.trend_threshold:
            direction = "bear"
        else:
            direction = "range"
        volatility = "high" if std_return >= self.high_volatility_threshold else "low"
        label = f"{volatility}_{direction}"

        metrics = {
            "abs_return_mean": statistics.fmean(abs(value) for value in returns),
            "last_return": returns[-1],
            "price_change": closes[-1] / closes[0] - 1.0,
        }
        self.current = MarketRegime(
            label=label,
            direction=direction,
            volatility=volatility,
            trend_score=trend_score,
            volatility_score=volatility_score,
            return_mean=mean_return,
            return_std=std_return,
            sample_size=len(closes),
            timestamp=self._now(),
            metrics=metrics,
        )
        self.history.append(self.current)
        if len(self.history) > 200:
            self.history = self.history[-200:]
        return self.current

    # ``update`` is a convenient streaming spelling for ``detect``.
    update = detect

    def _now(self) -> float:
        return float(self.clock()) if self.clock else __import__("time").time()

    def module_affinity(
        self, module: Any, regime: Optional[MarketRegime] = None
    ) -> float:
        """Return a bounded affinity multiplier declared by a module."""

        regime = regime or self.current
        affinities = getattr(module, "regime_affinity", {}) or {}
        if not isinstance(affinities, Mapping):
            return 1.0
        value = affinities.get(regime.label, affinities.get("default", 1.0))
        try:
            return max(0.5, min(1.5, float(value)))
        except (TypeError, ValueError):
            return 1.0

    def adaptive_parameters(
        self, regime: Optional[MarketRegime] = None
    ) -> Dict[str, Any]:
        """Parameters that are safe for modules to consume at runtime."""

        regime = regime or self.current
        if regime.is_unknown:
            return {
                "confidence_floor": 0.60,
                "weight_multiplier": 1.0,
                "high_volatility": False,
            }
        if regime.is_high_volatility:
            return {
                "confidence_floor": 0.68,
                "weight_multiplier": 0.85,
                "high_volatility": True,
            }
        if regime.direction == "range":
            return {
                "confidence_floor": 0.63,
                "weight_multiplier": 0.95,
                "high_volatility": False,
            }
        return {
            "confidence_floor": 0.58,
            "weight_multiplier": 1.05,
            "high_volatility": False,
        }


__all__ = ["MarketRegime", "MarketRegimeDetector", "extract_closes"]
