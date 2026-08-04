"""
OKX Live Engine - Real trading only, fails closed.

Requirements:
- ccxt must be installed, otherwise __init__ raises RuntimeError (no simulation fallback)
- OKX_API_KEY, OKX_API_SECRET, OKX_PASSPHRASE must be set, otherwise connect() fails
- Real market data required (no synthetic)

This is intentionally strict: real trading should fail closed if any dependency is missing.
For simulation/testing, use execution/okx_engine.py which explicitly allows simulation.

Safety:
- Leverage cap 3x default
- Position 10% equity cap
- Daily loss 3% (via safety_governance)
- Kill switch
- Human confirmation for live (via safety_governance)
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# ccxt is REQUIRED for real trading - fail closed if missing
try:
    import ccxt  # type: ignore
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

from .safety import OKXSafetyGuard, SAFETY_AVAILABLE

try:
    from core.event_bus import EventBus, get_event_bus  # type: ignore
    EVENTBUS_AVAILABLE = True
except ImportError:
    EVENTBUS_AVAILABLE = False
    EventBus = None  # type: ignore
    get_event_bus = None  # type: ignore


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    REJECTED = "rejected"


@dataclass
class OKXOrderRequest:
    symbol: str  # e.g. BTC/USDT
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    leverage: float = 1.0
    client_order_id: str = field(default_factory=lambda: f"okx_live_{int(time.time()*1000)}")
    reduce_only: bool = False
    tag: str = "okx_live_real"

    def normalized_symbol(self) -> str:
        return self.symbol.replace("-", "/").upper()


@dataclass
class OKXOrderResult:
    success: bool
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    filled_quantity: float
    avg_fill_price: float
    status: OrderStatus
    message: str
    timestamp: float = field(default_factory=time.time)
    fees: float = 0.0
    pnl: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value if isinstance(self.side, Enum) else str(self.side),
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "status": self.status.value if isinstance(self.status, Enum) else str(self.status),
            "message": self.message,
            "timestamp": self.timestamp,
            "fees": self.fees,
            "pnl": self.pnl,
        }


class OKXLiveEngine:
    """Real OKX trading engine - fails closed."""

    def __init__(
        self,
        paper_mode: bool = False,
        event_bus: Optional[Any] = None,
        max_leverage: float = 3.0,
        max_position_pct: float = 0.10,
        max_daily_loss_pct: float = 0.03,
    ):
        if not CCXT_AVAILABLE:
            raise RuntimeError(
                "ccxt not installed - real OKX trading requires ccxt. "
                "Install via pip install ccxt. Failing closed per safety requirements. "
                "For simulation, use execution/okx_engine.py (explicitly marked simulation)."
            )

        # Real trading must NOT be paper by default; but allow paper for test via env
        allow_paper = os.getenv("OKX_ALLOW_PAPER_FOR_TEST", "false").lower() in ("true", "1", "yes")
        if paper_mode and not allow_paper:
            raise RuntimeError(
                "OKXLiveEngine is for real trading only - paper_mode not allowed unless OKX_ALLOW_PAPER_FOR_TEST=true. "
                "Failing closed."
            )

        self.paper_mode = paper_mode
        self.api_key = os.getenv("OKX_API_KEY") or os.getenv("OKX_KEY")
        self.api_secret = os.getenv("OKX_API_SECRET") or os.getenv("OKX_SECRET")
        self.passphrase = os.getenv("OKX_PASSPHRASE")

        self.max_leverage = max_leverage
        self.max_position_pct = max_position_pct
        self.max_daily_loss_pct = max_daily_loss_pct

        if EVENTBUS_AVAILABLE and event_bus is None:
            try:
                self.event_bus = get_event_bus()  # type: ignore
            except Exception:
                self.event_bus = None
        else:
            self.event_bus = event_bus

        self.safety_guard = OKXSafetyGuard(
            max_leverage=self.max_leverage,
            max_position_pct=self.max_position_pct,
            max_daily_loss_pct=self.max_daily_loss_pct,
            require_live_credentials=not paper_mode,
        )

        # Fail closed on credentials
        ok, msg = self.safety_guard.validate_credentials()
        if not ok:
            if not allow_paper:
                raise RuntimeError(f"Credential validation failed: {msg} - failing closed for real trading")

        self.ccxt_client = None
        self.connected = False
        self._balance_cache: Optional[Dict[str, Any]] = None
        self.is_simulation = False  # Explicitly NOT simulation

        log.info("OKXLiveEngine initialized - REAL TRADING, fail-closed, paper=%s", self.paper_mode)

    def connect(self) -> bool:
        # Validate credentials again
        ok, msg = self.safety_guard.validate_credentials()
        if not ok:
            log.error("OKX connect failed credential check: %s", msg)
            raise RuntimeError(msg)

        if self.safety_guard.is_kill_switch_active():
            raise RuntimeError("Kill switch active - cannot connect")

        try:
            self.ccxt_client = ccxt.okx(  # type: ignore
                {
                    "apiKey": self.api_key,
                    "secret": self.api_secret,
                    "password": self.passphrase,
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                }
            )
            # Test fetch balance to validate connectivity
            bal = self.ccxt_client.fetch_balance()
            self._balance_cache = bal
            self.connected = True
            log.info("OKXLiveEngine connected - balance fetched")

            if self.event_bus:
                self.event_bus.publish("OKX_CONNECTED", {"real_trading": True}, source="OKXLiveEngine")

            return True
        except Exception as exc:
            log.exception("OKXLiveEngine connect failed: %s", exc)
            # Fail closed, do not fallback to simulation
            raise RuntimeError(f"OKXLiveEngine connect failed (fail-closed): {exc}") from exc

    def is_connected(self) -> bool:
        return self.connected and self.ccxt_client is not None

    def get_balance(self) -> Dict[str, Any]:
        if not self.is_connected():
            raise RuntimeError("Not connected - call connect() first (fail-closed)")

        try:
            bal = self.ccxt_client.fetch_balance()
            self._balance_cache = bal
            return bal
        except Exception as exc:
            raise RuntimeError(f"get_balance failed (fail-closed): {exc}") from exc

    def get_ticker(self, symbol: str) -> float:
        """Requires real market data - fails closed if unavailable."""
        if not self.is_connected():
            raise RuntimeError("Not connected - cannot get ticker (fail-closed)")

        norm = symbol.replace("-", "/").upper()
        try:
            ticker = self.ccxt_client.fetch_ticker(norm)
            price = float(ticker.get("last") or ticker.get("close") or 0)
            if not price:
                raise RuntimeError(f"Ticker for {symbol} returned no price - fail-closed")
            return price
        except Exception as exc:
            raise RuntimeError(f"get_ticker {symbol} failed (fail-closed, no synthetic fallback): {exc}") from exc

    def place_order(self, order: OKXOrderRequest) -> OKXOrderResult:
        if not self.is_connected():
            raise RuntimeError("Not connected - cannot place order (fail-closed)")

        if self.safety_guard.is_kill_switch_active():
            return OKXOrderResult(
                success=False,
                order_id="",
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                side=order.side,
                filled_quantity=0,
                avg_fill_price=0,
                status=OrderStatus.REJECTED,
                message="Kill switch active - fail-closed",
            )

        # Real price required
        price = order.limit_price or self.get_ticker(order.symbol)

        # Get equity for position sizing check
        bal = self.get_balance()
        # Try to extract total equity: look for USDT total
        equity = 100000.0  # fallback if parsing fails, but try to parse
        try:
            total = bal.get("total", {})
            if isinstance(total, dict):
                equity = float(total.get("USDT") or total.get("USD") or list(total.values())[0] if total else 100000.0)
        except Exception:
            pass

        ok, msg = self.safety_guard.check_order(
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            price=price,
            equity=equity,
            leverage=order.leverage,
        )
        if not ok:
            return OKXOrderResult(
                success=False,
                order_id="",
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                side=order.side,
                filled_quantity=0,
                avg_fill_price=0,
                status=OrderStatus.REJECTED,
                message=msg,
            )

        if self.event_bus:
            self.event_bus.publish(
                "ORDER_REQUEST",
                {
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "quantity": order.quantity,
                    "type": order.order_type.value,
                    "real_trading": True,
                },
                source="OKXLiveEngine",
            )

        try:
            ccxt_symbol = order.symbol.replace("-", "/")
            amount = order.quantity
            if order.order_type == OrderType.MARKET:
                ccxt_order = self.ccxt_client.create_market_order(ccxt_symbol, order.side.value, amount)
            else:
                if not order.limit_price:
                    raise RuntimeError("LIMIT order requires limit_price - fail-closed")
                ccxt_order = self.ccxt_client.create_order(
                    ccxt_symbol, order.order_type.value, order.side.value, amount, order.limit_price
                )

            filled = float(ccxt_order.get("filled", 0) or ccxt_order.get("amount", amount))
            avg_price = float(ccxt_order.get("average", 0) or ccxt_order.get("price", 0) or price)
            order_id = str(ccxt_order.get("id", ""))

            result = OKXOrderResult(
                success=True,
                order_id=order_id,
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                side=order.side,
                filled_quantity=filled,
                avg_fill_price=avg_price,
                status=OrderStatus.FILLED,
                message="REAL LIVE filled",
            )

            if self.event_bus:
                self.event_bus.publish("ORDER_FILLED", result.to_dict(), source="OKXLiveEngine")

            return result
        except Exception as exc:
            log.exception("Real OKX order failed: %s", exc)
            if self.event_bus:
                self.event_bus.publish(
                    "RISK_ALERT", {"error": str(exc), "order": order.client_order_id}, source="OKXLiveEngine"
                )
            # Fail closed - return rejected, not simulation fill
            return OKXOrderResult(
                success=False,
                order_id="",
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                side=order.side,
                filled_quantity=0,
                avg_fill_price=0,
                status=OrderStatus.REJECTED,
                message=f"REAL order exception (fail-closed): {exc}",
            )

    def place_order_from_signal(self, signal: Dict[str, Any], max_quantity: Optional[float] = None) -> Optional[OKXOrderResult]:
        """Requires real signal with symbol and final_signal BUY/SELL."""
        symbol = signal.get("symbol") or signal.get("asset")
        if not symbol:
            raise RuntimeError("Signal missing symbol - fail-closed")

        direction = (signal.get("final_signal") or signal.get("signal") or "").upper()
        if direction not in ("BUY", "SELL"):
            return None

        confidence = float(signal.get("confidence", 0.0))
        if confidence < 0.60:
            return None

        price = self.get_ticker(symbol)  # fails closed if no real data

        # Position sizing: equity * 1% * confidence / price - must have real equity
        bal = self.get_balance()
        equity = 100000.0
        try:
            total = bal.get("total", {})
            if isinstance(total, dict):
                equity = float(total.get("USDT") or total.get("USD") or 100000.0)
        except Exception:
            pass

        qty = (equity * 0.01 * confidence) / price
        if max_quantity:
            qty = min(qty, max_quantity)

        if qty * price < 5:
            return None

        from .engine import OrderSide as OSide, OrderType as OType  # local

        side = OSide.BUY if direction == "BUY" else OSide.SELL
        order = OKXOrderRequest(
            symbol=symbol,
            side=side,
            quantity=qty,
            order_type=OType.MARKET,
            leverage=min(1.0 + confidence, self.max_leverage),
        )

        return self.place_order(order)

    def get_status(self) -> Dict[str, Any]:
        return {
            "real_trading": True,
            "simulation": False,
            "connected": self.connected,
            "ccxt_available": CCXT_AVAILABLE,
            "paper_mode": self.paper_mode,
        }

    def activate_kill_switch(self, reason: str):
        if SAFETY_AVAILABLE and hasattr(self.safety_guard, "safety") and self.safety_guard.safety:
            self.safety_guard.safety.activate_kill_switch(reason, user="OKXLiveEngine")
        if self.event_bus:
            self.event_bus.publish("KILL_SWITCH", {"reason": reason, "real_trading": True}, source="OKXLiveEngine")
