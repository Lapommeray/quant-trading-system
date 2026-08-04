"""Immutable-in-configuration financial guardrails for autonomous execution.

Python cannot provide a literal hardware fuse inside a user-space process, so
this module uses a frozen limits object, private state, and a mandatory call
site in the execution engine.  Generated code is never given a reference to
this object.  Emergency stops and survival mode can tighten limits, but no
runtime path can loosen them.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class GuardrailLimits:
    max_position_pct: float = 0.02
    max_daily_trades: int = 50
    max_daily_loss_pct: float = 0.05
    max_leverage: float = 1.0
    min_seconds_between_trades: float = 10.0
    survival_position_pct: float = 0.01
    survival_leverage: float = 1.0


class CapitalTracker:
    """UTC-day capital and order-rate accounting."""

    def __init__(self, initial_equity: float = 0.0, clock=time.time) -> None:
        self.initial_equity = max(0.0, float(initial_equity))
        self.today_trades = 0
        self.today_pnl = 0.0
        self.last_trade_time: Optional[float] = None
        self.day_start = self._today()
        self.clock = clock

    def _today(self):
        return datetime.now(timezone.utc).date()

    def reset_if_new_day(self) -> None:
        today = self._today()
        if today != self.day_start:
            self.today_trades = 0
            self.today_pnl = 0.0
            self.last_trade_time = None
            self.day_start = today

    def set_equity(self, equity: float) -> None:
        equity = max(0.0, float(equity))
        if self.initial_equity <= 0 and equity > 0:
            self.initial_equity = equity

    def can_trade(self, max_daily_trades: int) -> bool:
        self.reset_if_new_day()
        return self.today_trades < max_daily_trades

    def rate_limit_ok(self, minimum_seconds: float) -> bool:
        if self.last_trade_time is None:
            return True
        return float(self.clock()) - self.last_trade_time >= minimum_seconds

    def record_trade(self, pnl: float = 0.0) -> None:
        self.reset_if_new_day()
        self.today_trades += 1
        self.today_pnl += float(pnl)
        self.last_trade_time = float(self.clock())

    def daily_loss_pct(self) -> float:
        if self.initial_equity <= 0:
            return 0.0
        return max(0.0, -self.today_pnl / self.initial_equity)

    def status(self) -> Dict[str, Any]:
        self.reset_if_new_day()
        return {
            "day": self.day_start.isoformat(),
            "trades": self.today_trades,
            "pnl": self.today_pnl,
            "initial_equity": self.initial_equity,
            "daily_loss_pct": self.daily_loss_pct(),
            "last_trade_time": self.last_trade_time,
        }


class AutonomousGuardrails:
    """Non-loosenable limits used immediately before an order is sent."""

    # Public constants are intentionally conservative.  The frozen limits
    # object can only tighten these values for a particular deployment.
    MAX_POSITION_PCT = 0.02
    MAX_DAILY_TRADES = 50
    MAX_DAILY_LOSS_PCT = 0.05
    MAX_LEVERAGE = 1.0
    MIN_SECONDS_BETWEEN_TRADES = 10.0
    MAX_MEMORY_MB = 500
    MAX_CPU_PERCENT = 25
    MODULE_TIMEOUT_SEC = 5
    MAX_CODE_LINES = 200
    MAX_COMPLEXITY = 10
    MAX_NESTED_DEPTH = 3

    __slots__ = (
        "_lock",
        "_limits",
        "_capital_tracker",
        "_violations",
        "_emergency_stop",
        "_survival_mode",
        "_audit_path",
        "_clock",
    )

    def __init__(
        self,
        *,
        initial_equity: float = 0.0,
        audit_path: Optional[str | Path] = None,
        limits: Optional[GuardrailLimits] = None,
        clock=time.time,
    ) -> None:
        base_limits = GuardrailLimits()
        requested = limits or base_limits
        # A caller may tighten limits but never loosen the compiled defaults.
        immutable_limits = GuardrailLimits(
            max_position_pct=min(
                requested.max_position_pct, base_limits.max_position_pct
            ),
            max_daily_trades=min(
                requested.max_daily_trades, base_limits.max_daily_trades
            ),
            max_daily_loss_pct=min(
                requested.max_daily_loss_pct, base_limits.max_daily_loss_pct
            ),
            max_leverage=min(requested.max_leverage, base_limits.max_leverage),
            min_seconds_between_trades=max(
                requested.min_seconds_between_trades,
                base_limits.min_seconds_between_trades,
            ),
            survival_position_pct=min(
                requested.survival_position_pct,
                base_limits.survival_position_pct,
            ),
            survival_leverage=min(
                requested.survival_leverage, base_limits.survival_leverage
            ),
        )
        object.__setattr__(self, "_lock", threading.RLock())
        object.__setattr__(self, "_limits", immutable_limits)
        object.__setattr__(
            self, "_capital_tracker", CapitalTracker(initial_equity, clock)
        )
        object.__setattr__(self, "_violations", [])
        emergency_from_env = os.getenv("EMERGENCY_STOP", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        object.__setattr__(self, "_emergency_stop", emergency_from_env)
        object.__setattr__(self, "_survival_mode", False)
        object.__setattr__(
            self,
            "_audit_path",
            Path(audit_path or "audit_logs/violations.jsonl"),
        )
        object.__setattr__(self, "_clock", clock)

    def __setattr__(self, name: str, value: Any) -> None:
        # Limits/state cannot be monkey-patched by a generated module.  The
        # controlled state transitions below use object.__setattr__ internally.
        raise PermissionError(f"guardrail state is immutable: {name}")

    @property
    def limits(self) -> GuardrailLimits:
        return self._limits

    @property
    def emergency_stop(self) -> bool:
        return self._emergency_stop

    @property
    def survival_mode(self) -> bool:
        return self._survival_mode

    def set_survival_mode(self, active: bool, reason: str = "") -> None:
        with self._lock:
            object.__setattr__(self, "_survival_mode", bool(active))
            self._log_violation("survival_mode", reason or bool(active))

    def validate_trade(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate a trade before it reaches an exchange adapter."""

        with self._lock:
            if self._emergency_stop:
                return False, "EMERGENCY_STOP_ACTIVE"

            position_pct = float(params.get("position_size_pct", 0.0) or 0.0)
            leverage = float(params.get("leverage", 1.0) or 1.0)
            if position_pct < 0 or leverage <= 0:
                self._log_violation("invalid_trade_parameters", params)
                return False, "Invalid position or leverage"

            max_position = (
                self._limits.survival_position_pct
                if self._survival_mode
                else self._limits.max_position_pct
            )
            max_leverage = (
                self._limits.survival_leverage
                if self._survival_mode
                else self._limits.max_leverage
            )
            if position_pct > max_position:
                self._log_violation("position_size", position_pct)
                return (
                    False,
                    f"Position {position_pct:.2%} exceeds max {max_position:.2%}",
                )
            if leverage > max_leverage:
                self._log_violation("leverage", leverage)
                return (
                    False,
                    f"Leverage {leverage:.2f}x exceeds max {max_leverage:.2f}x",
                )
            if not self._capital_tracker.can_trade(self._limits.max_daily_trades):
                return False, self.block_reason()
            if not self._capital_tracker.rate_limit_ok(
                self._limits.min_seconds_between_trades
            ):
                return (
                    False,
                    f"Rate limit: {self._limits.min_seconds_between_trades:g}s between trades",
                )
            if (
                self._capital_tracker.daily_loss_pct()
                >= self._limits.max_daily_loss_pct
            ):
                object.__setattr__(self, "_emergency_stop", True)
                return False, "Daily loss limit reached"
            return True, "OK"

    def record_trade_result(self, pnl: float, *, count_trade: bool = True) -> None:
        with self._lock:
            if count_trade:
                self._capital_tracker.record_trade(float(pnl))
            else:
                self._capital_tracker.today_pnl += float(pnl)
            if (
                self._capital_tracker.daily_loss_pct()
                >= self._limits.max_daily_loss_pct
            ):
                self.trigger_emergency_stop(
                    f"Daily loss {self._capital_tracker.daily_loss_pct():.2%}"
                )

    def set_equity(self, equity: float) -> None:
        with self._lock:
            self._capital_tracker.set_equity(equity)

    def trigger_emergency_stop(self, reason: str) -> None:
        with self._lock:
            object.__setattr__(self, "_emergency_stop", True)
            self._log_violation("emergency_stop", reason)

    def reset_emergency_stop(self, confirmation: str) -> bool:
        """Manual-only reset; generated code has no confirmation token."""

        if confirmation != "MANUAL_RESET":
            return False
        with self._lock:
            object.__setattr__(self, "_emergency_stop", False)
            self._log_violation("manual_reset", "MANUAL_RESET")
        return True

    def _log_violation(self, rule: str, value: Any) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rule": rule,
            "value": value,
        }
        self._violations.append(entry)
        if self._audit_path is not None:
            try:
                self._audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self._audit_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(entry, sort_keys=True) + "\n")
            except OSError:
                # A logging failure must not loosen a guardrail.
                pass

    def block_reason(self) -> str:
        return f"Daily trade limit reached: {self._capital_tracker.today_trades}/{self._limits.max_daily_trades}"

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "limits": asdict(self._limits),
                "emergency_stop": self._emergency_stop,
                "survival_mode": self._survival_mode,
                "capital": self._capital_tracker.status(),
                "violations": list(self._violations[-50:]),
            }


__all__ = ["AutonomousGuardrails", "CapitalTracker", "GuardrailLimits"]
