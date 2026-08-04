"""
Autonomy execution - event-driven routing to OKX live (fails closed).

No QuantConnect dependency. Requires real OKX credentials and real market data.
Will NOT fallback to simulation or synthetic data.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

try:
    from okx_live.engine import OKXLiveEngine  # type: ignore
except ImportError:
    try:
        from okx_live.engine import OKXLiveEngine  # type: ignore
    except Exception:
        # Will be imported lazily via local path
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        try:
            from okx_live.engine import OKXLiveEngine  # type: ignore
        except Exception:
            OKXLiveEngine = None  # type: ignore

try:
    from core.event_bus import EventBus, get_event_bus  # type: ignore
    EVENTBUS_AVAILABLE = True
except ImportError:
    EVENTBUS_AVAILABLE = False
    EventBus = None  # type: ignore
    get_event_bus = None  # type: ignore


@dataclass
class ExecutorConfig:
    min_confidence: float = 0.65
    max_quantity: Optional[float] = None
    allowed_symbols: Optional[list] = None
    blocked_symbols: Optional[list] = None
    require_weighted_confidence: float = 0.30
    fail_closed: bool = True  # if True, raise if engine not ready


class AutonomousExecutor:
    """Event-driven executor for real OKX trading - fails closed."""

    def __init__(
        self,
        okx_engine: Optional[Any] = None,
        event_bus: Optional[Any] = None,
        config: Optional[ExecutorConfig] = None,
    ):
        self.config = config or ExecutorConfig()
        self.event_bus = event_bus
        if self.event_bus is None and EVENTBUS_AVAILABLE and get_event_bus:
            try:
                self.event_bus = get_event_bus()
            except Exception:
                self.event_bus = None

        # Engine must be real - no simulation fallback
        if okx_engine is None:
            if OKXLiveEngine is None:
                raise RuntimeError("OKXLiveEngine not available - cannot create AutonomousExecutor in fail-closed mode")
            okx_engine = OKXLiveEngine(paper_mode=False, event_bus=self.event_bus)

        # Validate engine is real, not simulation
        if hasattr(okx_engine, "is_simulation") and okx_engine.is_simulation:
            if self.config.fail_closed:
                raise RuntimeError("Engine is in simulation mode but fail_closed=True requires real trading")

        self.okx_engine = okx_engine
        self._running = False
        self._stats = {"signals_received": 0, "orders_placed": 0, "orders_blocked": 0, "last_signal": None}
        self._lock = threading.RLock()

        if self.event_bus:
            self.event_bus.subscribe("SIGNAL_GENERATED", self._on_signal)
            self.event_bus.subscribe("KILL_SWITCH", self._on_kill_switch)
            log.info("AutonomousExecutor subscribed to SIGNAL_GENERATED (fail_closed=%s)", self.config.fail_closed)

    def start(self):
        if self._running:
            return
        # Fail closed if not connected or credentials missing
        if not self.okx_engine.is_connected():
            connected = self.okx_engine.connect()
            if not connected and self.config.fail_closed:
                raise RuntimeError("OKX engine failed to connect and fail_closed=True - aborting start")

        self._running = True
        log.info("AutonomousExecutor started - real trading, fail_closed=%s", self.config.fail_closed)
        if self.event_bus:
            self.event_bus.publish("EXECUTOR_STARTED", {"real_trading": True}, source="AutonomousExecutor")

    def stop(self):
        self._running = False
        log.info("AutonomousExecutor stopped stats=%s", self._stats)

    def _on_signal(self, event: Any):
        payload = event.payload if hasattr(event, "payload") else event
        if not isinstance(payload, dict):
            log.warning("Signal payload not dict: %s", payload)
            return

        with self._lock:
            self._stats["signals_received"] += 1
            self._stats["last_signal"] = payload

        symbol = payload.get("symbol", "BTC/USDT")
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

        try:
            result = self.okx_engine.place_order_from_signal(payload, max_quantity=self.config.max_quantity)
            if result:
                if result.success:
                    with self._lock:
                        self._stats["orders_placed"] += 1
                    log.info("REAL OKX order placed %s %s conf=%.2f qty=%.6f", symbol, final_signal, confidence, result.filled_quantity)
                else:
                    with self._lock:
                        self._stats["orders_blocked"] += 1
                    log.warning("REAL OKX order blocked %s: %s", symbol, result.message)
            else:
                with self._lock:
                    self._stats["orders_blocked"] += 1
        except Exception as exc:
            log.exception("AutonomousExecutor failed for %s: %s", symbol, exc)
            with self._lock:
                self._stats["orders_blocked"] += 1
            if self.config.fail_closed:
                raise

    def _on_kill_switch(self, event: Any):
        payload = event.payload if hasattr(event, "payload") else {}
        reason = payload.get("reason", "unknown") if isinstance(payload, dict) else "unknown"
        log.critical("Kill switch received in AutonomousExecutor: %s", reason)
        if hasattr(self.okx_engine, "activate_kill_switch"):
            self.okx_engine.activate_kill_switch(f"Kill switch propagated: {reason}")

    def execute_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Direct execution - still fails closed on safety violation."""
        result = self.okx_engine.place_order_from_signal(signal, max_quantity=self.config.max_quantity)
        if result:
            if result.success:
                self._stats["orders_placed"] += 1
            else:
                self._stats["orders_blocked"] += 1
            return result.to_dict()
        return None

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            base = dict(self._stats)
        try:
            base.update(self.okx_engine.get_status())
        except Exception:
            pass
        return base


# Alias for backwards compat with earlier execution module
EventDrivenExecutor = AutonomousExecutor
