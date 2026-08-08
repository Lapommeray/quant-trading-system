"""
Event-Driven OKX Executor - connects Organism signals to OKX Engine.

Flow:
Organism.generate_consensus_signal() -> publishes SIGNAL_GENERATED
EventDrivenExecutor subscribes -> validates -> routes to OKXEngine.place_order_from_signal()

Also supports MT5 fallback and paper mode.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

try:
    from .okx_engine import OKXEngine, OKXOrderRequest, OrderSide, OrderType  # type: ignore
except ImportError:
    from execution.okx_engine import OKXEngine, OKXOrderRequest, OrderSide, OrderType  # type: ignore

try:
    from core.event_bus import EventBus, Event, get_event_bus  # type: ignore
except ImportError:
    EventBus = None  # type: ignore
    Event = None  # type: ignore
    get_event_bus = None  # type: ignore

try:
    from safety_governance import SafetyGovernanceSystem  # type: ignore

    SAFETY = True
except ImportError:
    SAFETY = False


@dataclass
class ExecutorConfig:
    min_confidence: float = 0.60
    max_quantity: Optional[float] = None  # cap per order
    allowed_symbols: Optional[list] = None  # None = all
    blocked_symbols: Optional[list] = None
    require_organism_weight: float = 0.0  # minimum organism weighted confidence


class EventDrivenExecutor:
    """Listens to signal events and executes via OKX."""

    def __init__(
        self,
        okx_engine: Optional[OKXEngine] = None,
        event_bus: Optional[Any] = None,
        config: Optional[ExecutorConfig] = None,
    ):
        self.config = config or ExecutorConfig()
        self.okx_engine = okx_engine or OKXEngine()
        if event_bus is None and get_event_bus:
            try:
                self.event_bus = get_event_bus()
            except Exception:
                self.event_bus = None
        else:
            self.event_bus = event_bus

        self._running = False
        self._stats = {
            "signals_received": 0,
            "orders_placed": 0,
            "orders_blocked": 0,
            "last_signal": None,
        }
        self._lock = threading.RLock()

        # Wire subscriptions
        if self.event_bus:
            self.event_bus.subscribe("SIGNAL_GENERATED", self._on_signal)
            self.event_bus.subscribe("KILL_SWITCH", self._on_kill_switch)
            log.info("EventDrivenExecutor subscribed to SIGNAL_GENERATED")

    def start(self):
        if self._running:
            return
        if not self.okx_engine.is_connected():
            self.okx_engine.connect()
        self._running = True
        log.info("EventDrivenExecutor started paper=%s", self.okx_engine.paper_mode)
        if self.event_bus:
            self.event_bus.publish(
                "EXECUTOR_STARTED",
                {"paper": self.okx_engine.paper_mode},
                source="EventDrivenExecutor",
            )

    def stop(self):
        self._running = False
        log.info("EventDrivenExecutor stopped stats=%s", self._stats)

    # ---- event handlers ----
    def _on_signal(self, event: Any):
        """Handle SIGNAL_GENERATED event."""
        with self._lock:
            self._stats["signals_received"] += 1
            self._stats["last_signal"] = (
                event.payload if hasattr(event, "payload") else event
            )

        payload = event.payload if hasattr(event, "payload") else event
        if not isinstance(payload, dict):
            log.warning("Signal payload not dict: %s", payload)
            return

        symbol = payload.get("symbol", "BTC/USDT")
        final_signal = payload.get("final_signal")
        confidence = payload.get("confidence", 0.0)
        weighted_conf = payload.get("weighted_confidence", 0.0)

        # Filters
        if self.config.allowed_symbols and symbol not in self.config.allowed_symbols:
            log.debug("Symbol %s not in allowed list", symbol)
            return
        if self.config.blocked_symbols and symbol in self.config.blocked_symbols:
            log.debug("Symbol %s blocked", symbol)
            return
        if confidence < self.config.min_confidence:
            log.debug(
                "Confidence %.3f below min %.3f for %s",
                confidence,
                self.config.min_confidence,
                symbol,
            )
            with self._lock:
                self._stats["orders_blocked"] += 1
            return
        if weighted_conf < self.config.require_organism_weight:
            log.debug(
                "Weighted conf %.3f below required %.3f",
                weighted_conf,
                self.config.require_organism_weight,
            )
            with self._lock:
                self._stats["orders_blocked"] += 1
            return
        if not final_signal or final_signal.upper() not in ("BUY", "SELL"):
            log.debug("No tradable signal for %s: %s", symbol, final_signal)
            return

        # Execute
        try:
            result = self.okx_engine.place_order_from_signal(
                payload, max_quantity=self.config.max_quantity
            )
            if result:
                if result.success:
                    with self._lock:
                        self._stats["orders_placed"] += 1
                    log.info(
                        "OKX order placed from signal %s %s conf=%.2f qty=%.6f",
                        symbol,
                        final_signal,
                        confidence,
                        result.filled_quantity,
                    )
                else:
                    with self._lock:
                        self._stats["orders_blocked"] += 1
                    log.info("OKX order blocked for %s: %s", symbol, result.message)
        except Exception as exc:
            log.exception(
                "EventDrivenExecutor failed to place order for %s: %s", symbol, exc
            )
            with self._lock:
                self._stats["orders_blocked"] += 1

    def _on_kill_switch(self, event: Any):
        payload = event.payload if hasattr(event, "payload") else {}
        reason = (
            payload.get("reason", "unknown") if isinstance(payload, dict) else "unknown"
        )
        log.warning("Kill switch received in executor: %s", reason)
        # Could pause trading
        self.okx_engine.local_breaker.trip(f"Kill switch propagated: {reason}")

    # ---- direct API for non-event usage ----
    def execute_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Direct execution without bus."""
        result = self.okx_engine.place_order_from_signal(
            signal, max_quantity=self.config.max_quantity
        )
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
        base.update(self.okx_engine.get_status())
        return base
