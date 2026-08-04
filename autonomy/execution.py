"""
Autonomy execution - event-driven routing to OKX live trader, fail-closed.

Signals, fills and reconciliation metadata travel over the same canonical bus
as the organism.  This lets the learning store correlate a realized outcome
with every module that contributed to a signal without coupling modules to the
exchange adapter.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

try:
    from okx_live.trader import OKXLiveTrader
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from okx_live.trader import OKXLiveTrader
    except Exception:
        OKXLiveTrader = None  # type: ignore

from .audit import AuditTrail
from .guardrails import AutonomousGuardrails
from .organism import get_event_bus


@dataclass
class ExecutorConfig:
    min_confidence: float = 0.65
    max_quantity: Optional[float] = None
    allowed_symbols: Optional[list] = None
    blocked_symbols: Optional[list] = None
    require_weighted_confidence: float = 0.30
    fail_closed: bool = True


class AutonomousExecutor:
    def __init__(
        self,
        okx_engine: Optional[Any] = None,
        event_bus: Optional[Any] = None,
        config: Optional[ExecutorConfig] = None,
    ):
        self.config = config or ExecutorConfig()
        self.event_bus = event_bus or get_event_bus()

        # okx_engine param can be trader instance for backwards compat
        if okx_engine is None:
            if OKXLiveTrader is None:
                raise RuntimeError("OKXLiveTrader not available - fail-closed")
            okx_engine = OKXLiveTrader()

        # Validate real trading (fail-closed)
        if hasattr(okx_engine, "engine") and hasattr(
            okx_engine.engine, "is_simulation"
        ):
            if okx_engine.engine.is_simulation and self.config.fail_closed:
                raise RuntimeError("Engine simulation but fail_closed=True")

        self.okx_trader = okx_engine
        self.guardrails = AutonomousGuardrails()
        self.audit_trail = AuditTrail("audit_logs/trade_audit.jsonl")
        self._survival_mode = False
        self._running = False
        self._stats = {
            "signals_received": 0,
            "orders_placed": 0,
            "orders_blocked": 0,
            "last_signal": None,
        }
        self._lock = threading.RLock()

        self.event_bus.subscribe("SIGNAL_GENERATED", self._on_signal)
        self.event_bus.subscribe("KILL_SWITCH", self._on_kill_switch)
        self.event_bus.subscribe("SURVIVAL_MODE", self._on_survival_mode)
        self.event_bus.subscribe("CLEAR_SURVIVAL_MODE", self._on_clear_survival_mode)
        log.info(
            "AutonomousExecutor subscribed (fail_closed=%s)", self.config.fail_closed
        )

    def start(self):
        if self._running:
            return
        if not self.okx_trader.is_connected():
            connected = self.okx_trader.connect()
            if not connected and self.config.fail_closed:
                raise RuntimeError("OKX trader failed to connect and fail_closed=True")
        self._running = True
        self.event_bus.publish(
            "EXECUTOR_STARTED", {"real_trading": True}, source="AutonomousExecutor"
        )

    def stop(self):
        self._running = False
        log.info("AutonomousExecutor stopped stats=%s", self._stats)

    def _on_signal(self, event: Any):
        payload = event.payload if hasattr(event, "payload") else event
        if not isinstance(payload, dict):
            return
        with self._lock:
            self._stats["signals_received"] += 1
            self._stats["last_signal"] = payload

        symbol = payload.get("symbol", "BTC/USDT")
        self.audit_trail.record(
            "TRADE_ATTEMPT",
            {
                "symbol": symbol,
                "signal": payload.get("final_signal"),
                "payload": payload,
            },
            source="AutonomousExecutor",
        )
        if self._survival_mode or payload.get("survival_mode"):
            self.audit_trail.record(
                "TRADE_BLOCKED",
                {"symbol": symbol, "reason": "survival mode"},
                source="AutonomousExecutor",
            )
            with self._lock:
                self._stats["orders_blocked"] += 1
            return
        final_signal = payload.get("final_signal")
        confidence = payload.get("confidence", 0.0)
        weighted_conf = payload.get("weighted_confidence", 0.0)

        if self.config.allowed_symbols and symbol not in self.config.allowed_symbols:
            return
        if self.config.blocked_symbols and symbol in self.config.blocked_symbols:
            return
        if confidence < self.config.min_confidence:
            with self._lock:
                self._stats["orders_blocked"] += 1
            return
        if weighted_conf < self.config.require_weighted_confidence:
            with self._lock:
                self._stats["orders_blocked"] += 1
            return
        if not final_signal or final_signal.upper() not in ("BUY", "SELL"):
            return
        allowed, reason = self.guardrails.validate_trade(
            {
                "position_size_pct": payload.get(
                    "position_size_pct", payload.get("position_size", 0.0)
                ),
                "leverage": payload.get("leverage", 1.0),
            }
        )
        if not allowed:
            log.warning("Autonomous guardrail blocked %s: %s", symbol, reason)
            self.audit_trail.record(
                "TRADE_BLOCKED",
                {"symbol": symbol, "reason": reason},
                source="AutonomousGuardrails",
            )
            with self._lock:
                self._stats["orders_blocked"] += 1
            return

        try:
            result = self.okx_trader.place_order_from_signal(
                payload, max_quantity=self.config.max_quantity
            )
            if result:
                result_success = (
                    bool(result.get("success"))
                    if isinstance(result, dict)
                    else bool(getattr(result, "success", False))
                )
                if result_success:
                    with self._lock:
                        self._stats["orders_placed"] += 1
                    # The live engine emits its own fill event, but it does
                    # not know which consensus prediction caused the order.
                    # Emit a correlated reconciliation event for learning.
                    result_payload = (
                        result.to_dict()
                        if hasattr(result, "to_dict")
                        else dict(result) if isinstance(result, dict) else {}
                    )
                    result_payload.update(
                        {
                            "symbol": symbol,
                            "prediction_ids": payload.get("prediction_ids", {}),
                            "source_module": payload.get("source_module"),
                            "regime": payload.get("regime", "unknown"),
                            "outcome_type": "fill_reconciliation",
                        }
                    )
                    self.audit_trail.record(
                        "TRADE_PLACED",
                        {"symbol": symbol, "result": result_payload},
                        source="AutonomousExecutor",
                    )
                    try:
                        realized_pnl = float(result_payload.get("pnl") or 0.0)
                    except (TypeError, ValueError):
                        realized_pnl = 0.0
                    self.guardrails.record_trade_result(realized_pnl)
                    self.event_bus.publish(
                        "TRADE_OUTCOME", result_payload, source="AutonomousExecutor"
                    )
                else:
                    self.audit_trail.record(
                        "TRADE_BLOCKED",
                        {"symbol": symbol, "reason": "exchange rejected"},
                        source="AutonomousExecutor",
                    )
                    with self._lock:
                        self._stats["orders_blocked"] += 1
        except Exception as exc:
            log.exception("Executor failed for %s: %s", symbol, exc)
            with self._lock:
                self._stats["orders_blocked"] += 1
            if self.config.fail_closed:
                raise

    def _on_kill_switch(self, event: Any):
        payload = event.payload if hasattr(event, "payload") else {}
        reason = (
            payload.get("reason", "unknown") if isinstance(payload, dict) else "unknown"
        )
        log.critical("Kill switch in executor: %s", reason)
        if hasattr(self.okx_trader, "activate_kill_switch"):
            self.okx_trader.activate_kill_switch(f"Kill switch: {reason}")

    def _on_survival_mode(self, event: Any):
        payload = event.payload if hasattr(event, "payload") else {}
        reason = (
            payload.get("reason", "panic sentinel")
            if isinstance(payload, dict)
            else "panic sentinel"
        )
        self._survival_mode = True
        self.guardrails.set_survival_mode(True, str(reason))
        if hasattr(self.okx_trader, "set_survival_mode"):
            self.okx_trader.set_survival_mode(True, str(reason))
        log.critical("Executor entered survival mode: %s", reason)

    def _on_clear_survival_mode(self, event: Any):
        self._survival_mode = False
        self.guardrails.set_survival_mode(False, "sentinel stabilized")
        if hasattr(self.okx_trader, "set_survival_mode"):
            self.okx_trader.set_survival_mode(False, "sentinel stabilized")
        log.warning("Executor left survival mode after sentinel stabilization")

    def get_stats(self):
        with self._lock:
            base = dict(self._stats)
        base["survival_mode"] = self._survival_mode
        base["guardrails"] = self.guardrails.get_status()
        base["audit"] = self.audit_trail.status()
        try:
            base.update(self.okx_trader.get_status())
        except Exception:
            pass
        return base


# Alias
EventDrivenExecutor = AutonomousExecutor
ExecutorConfig = ExecutorConfig
