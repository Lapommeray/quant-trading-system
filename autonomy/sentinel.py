"""Multi-timeframe panic sentinel and survival-mode state machine."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from .market import extract_closes


@dataclass(frozen=True)
class SentinelConfig:
    timeframes: tuple[str, ...] = ("1m", "15m", "1h")
    sigma_threshold: float = 3.0
    minimum_samples: int = 20
    stabilization_observations: int = 3
    heartbeat_timeout_seconds: float = 30.0


@dataclass(frozen=True)
class SentinelDecision:
    survival_mode: bool
    changed: bool
    cleared: bool
    spikes: Dict[str, bool] = field(default_factory=dict)
    z_scores: Dict[str, float] = field(default_factory=dict)
    reason: str = ""
    risk_profile: Dict[str, float] = field(default_factory=dict)
    heartbeat_ok: bool = True
    error: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "survival_mode": self.survival_mode,
            "changed": self.changed,
            "cleared": self.cleared,
            "spikes": dict(self.spikes),
            "z_scores": dict(self.z_scores),
            "reason": self.reason,
            "risk_profile": dict(self.risk_profile),
            "heartbeat_ok": self.heartbeat_ok,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class MultiTimeframeSentinel:
    """Trigger survival mode only when all configured timeframes spike."""

    RISK_PROFILE = {
        "max_leverage": 1.0,
        "max_position_pct": 0.01,
        "max_daily_loss_pct": 0.02,
        "stop_multiplier": 0.50,
    }

    def __init__(
        self, config: Optional[SentinelConfig] = None, clock=time.time
    ) -> None:
        self.config = config or SentinelConfig()
        self.clock = clock
        self.survival_mode = False
        self._stable_count = 0
        self._last_heartbeat = 0.0
        self._healthy = False
        self._last_error = ""
        self.last_decision = SentinelDecision(False, False, False, heartbeat_ok=False)

    def evaluate(self, history_data: Any) -> SentinelDecision:
        self._last_heartbeat = float(self.clock())
        try:
            spikes: Dict[str, bool] = {}
            z_scores: Dict[str, float] = {}
            eligible: Dict[str, bool] = {}
            for timeframe in self.config.timeframes:
                if isinstance(history_data, Mapping) and timeframe in history_data:
                    frame = history_data[timeframe]
                else:
                    frame = history_data
                z_score, spike, enough = self._measure_spike(frame)
                z_scores[timeframe] = z_score
                spikes[timeframe] = spike
                eligible[timeframe] = enough

            all_spike = bool(spikes) and all(spikes.values()) and all(eligible.values())
            changed = False
            cleared = False
            reason = "stable"
            if not self.survival_mode and all_spike:
                self.survival_mode = True
                self._stable_count = 0
                changed = True
                reason = "3-sigma volatility spike across all sentinel timeframes"
            elif self.survival_mode:
                if all(eligible.values()) and not all_spike:
                    self._stable_count += 1
                else:
                    self._stable_count = 0
                if self._stable_count >= max(1, self.config.stabilization_observations):
                    self.survival_mode = False
                    self._stable_count = 0
                    changed = True
                    cleared = True
                    reason = "sentinel observed stabilized volatility"
                else:
                    reason = "survival mode active pending stabilization"
            self._healthy = True
            self._last_error = ""
            decision = SentinelDecision(
                survival_mode=self.survival_mode,
                changed=changed,
                cleared=cleared,
                spikes=spikes,
                z_scores=z_scores,
                reason=reason,
                risk_profile=dict(self.RISK_PROFILE),
                heartbeat_ok=True,
                timestamp=float(self.clock()),
            )
        except Exception as exc:
            self._healthy = False
            self._last_error = str(exc)
            decision = self.force_survival(
                f"sentinel failure: {exc}", heartbeat_ok=False, error=str(exc)
            )
        self.last_decision = decision
        return decision

    def heartbeat_ok(self) -> bool:
        return bool(
            self._healthy
            and self._last_heartbeat > 0
            and float(self.clock()) - self._last_heartbeat
            <= max(1.0, self.config.heartbeat_timeout_seconds)
        )

    def force_survival(
        self,
        reason: str = "manual panic",
        *,
        heartbeat_ok: Optional[bool] = None,
        error: Optional[str] = None,
    ) -> SentinelDecision:
        changed = not self.survival_mode
        self.survival_mode = True
        self._stable_count = 0
        self.last_decision = SentinelDecision(
            survival_mode=True,
            changed=changed,
            cleared=False,
            reason=reason,
            risk_profile=dict(self.RISK_PROFILE),
            heartbeat_ok=self.heartbeat_ok() if heartbeat_ok is None else heartbeat_ok,
            error=self._last_error if error is None else error,
            timestamp=float(self.clock()),
        )
        return self.last_decision

    def _measure_spike(self, frame: Any) -> tuple[float, bool, bool]:
        closes = extract_closes(frame)
        if len(closes) < self.config.minimum_samples + 1:
            return 0.0, False, False
        returns = [
            math.log(closes[index] / closes[index - 1])
            for index in range(1, len(closes))
            if closes[index - 1] > 0 and closes[index] > 0
        ]
        if len(returns) < self.config.minimum_samples:
            return 0.0, False, False
        baseline = returns[:-1]
        latest = returns[-1]
        mean = statistics.fmean(baseline)
        std = statistics.pstdev(baseline) if len(baseline) > 1 else 0.0
        if std <= 1e-12:
            z_score = float("inf") if abs(latest - mean) > 1e-12 else 0.0
        else:
            z_score = abs(latest - mean) / std
        return z_score, z_score >= self.config.sigma_threshold, True

    def get_status(self) -> Dict[str, Any]:
        return {
            "survival_mode": self.survival_mode,
            "stable_count": self._stable_count,
            "stabilization_required": self.config.stabilization_observations,
            "heartbeat_ok": self.heartbeat_ok(),
            "heartbeat_age_seconds": (
                max(0.0, float(self.clock()) - self._last_heartbeat)
                if self._last_heartbeat
                else None
            ),
            "last_error": self._last_error,
            "last_decision": self.last_decision.to_dict(),
        }


# Semantic alias for integrations.
PanicSentinel = MultiTimeframeSentinel

__all__ = [
    "MultiTimeframeSentinel",
    "PanicSentinel",
    "SentinelConfig",
    "SentinelDecision",
]
