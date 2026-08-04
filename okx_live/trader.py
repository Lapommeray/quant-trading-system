"""
OKX Live Trader - high-level trader using real engine + config, fail-closed.

This is the canonical trader referenced by autonomy/execution.py and main.py.
It wraps OKXLiveEngine + OKXLiveConfig + SafetyGuard.

No QuantConnect. No synthetic fallback.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .config import OKXLiveConfig, get_okx_config
from .engine import OKXLiveEngine, OKXOrderRequest, OKXOrderResult, OrderSide, OrderType
from .safety import OKXSafetyGuard

log = logging.getLogger(__name__)

try:
    from core.event_bus import get_event_bus  # type: ignore
    EVENTBUS_AVAILABLE = True
except ImportError:
    EVENTBUS_AVAILABLE = False
    get_event_bus = None  # type: ignore


class OKXLiveTrader:
    """High-level trader for real OKX trading - fail-closed."""

    def __init__(self, config: Optional[OKXLiveConfig] = None, event_bus: Optional[Any] = None):
        self.config = config or get_okx_config()
        self.event_bus = event_bus
        if self.event_bus is None and EVENTBUS_AVAILABLE and get_event_bus:
            try:
                self.event_bus = get_event_bus()
            except Exception:
                self.event_bus = None

        # Validate config for real trading - fails closed if missing
        if self.config.live_trading:
            self.config.validate_for_real_trading()

        self.safety_guard = OKXSafetyGuard(
            max_leverage=self.config.max_leverage,
            max_position_pct=self.config.max_position_pct,
            max_daily_loss_pct=self.config.max_daily_loss_pct,
            require_live_credentials=self.config.live_trading,
        )

        # Engine is real, fail-closed
        self.engine = OKXLiveEngine(
            paper_mode=not self.config.live_trading,
            event_bus=self.event_bus,
            max_leverage=self.config.max_leverage,
            max_position_pct=self.config.max_position_pct,
            max_daily_loss_pct=self.config.max_daily_loss_pct,
        )

        log.info("OKXLiveTrader initialized live=%s", self.config.live_trading)

    def connect(self) -> bool:
        """Connect to OKX - fails closed if credentials or ccxt missing."""
        return self.engine.connect()

    def is_connected(self) -> bool:
        return self.engine.is_connected()

    def place_order_from_signal(self, signal: Dict[str, Any], max_quantity: Optional[float] = None) -> Optional[OKXOrderResult]:
        """Place order from consensus signal - fails closed if no real data."""
        return self.engine.place_order_from_signal(signal, max_quantity=max_quantity)

    def place_market_order(self, symbol: str, side: str, quantity: float) -> OKXOrderResult:
        """Direct market order - fail-closed."""
        side_enum = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        order = OKXOrderRequest(symbol=symbol, side=side_enum, quantity=quantity, order_type=OrderType.MARKET, leverage=1.0)
        return self.engine.place_order(order)

    def get_balance(self) -> Dict[str, Any]:
        return self.engine.get_balance()

    def get_ticker(self, symbol: str) -> float:
        return self.engine.get_ticker(symbol)

    def get_status(self) -> Dict[str, Any]:
        return {
            "config": {
                "live_trading": self.config.live_trading,
                "max_leverage": self.config.max_leverage,
                "max_position_pct": self.config.max_position_pct,
            },
            "engine": self.engine.get_status(),
        }

    def activate_kill_switch(self, reason: str):
        self.engine.activate_kill_switch(reason)
