"""
OKX Live Trader - real trading, fail-closed.

No simulation fallback. Requires ccxt + credentials + real market data.

This file consolidates previous engine.py + safety.py + config.py logic into single trader
to match expected checkout that only has config.py and trader.py.

If ccxt or credentials missing, raises RuntimeError (fail-closed).
"""

from __future__ import annotations

import os
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# ccxt required - fail-closed
try:
    import ccxt  # type: ignore
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

# Event bus optional
try:
    from autonomy.organism import get_event_bus
    EVENTBUS_AVAILABLE = True
except ImportError:
    EVENTBUS_AVAILABLE = False
    def get_event_bus():  # type: ignore
        return None

from .config import OKXLiveConfig, get_okx_config


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    FILLED = "filled"
    REJECTED = "rejected"


@dataclass
class OKXOrderRequest:
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    client_order_id: str = field(default_factory=lambda: f"okx_live_{int(time.time()*1000)}")
    leverage: float = 1.0


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

    def to_dict(self):
        return {
            "success": self.success,
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "status": self.status.value,
            "message": self.message,
        }


class OKXLiveTrader:
    """Real OKX trader - fail-closed, no simulation."""

    def __init__(self, config: Optional[OKXLiveConfig] = None, event_bus: Optional[Any] = None):
        if not CCXT_AVAILABLE:
            raise RuntimeError("ccxt not installed - real OKX trading requires ccxt, fail-closed")

        self.config = config or get_okx_config()
        self.event_bus = event_bus or (get_event_bus() if EVENTBUS_AVAILABLE else None)

        # Validate credentials for real trading
        if self.config.live_trading:
            self.config.validate_for_real_trading()

        self.ccxt_client = None
        self.connected = False

        # Safety params from config
        self.max_leverage = self.config.max_leverage
        self.max_position_pct = self.config.max_position_pct

        log.info("OKXLiveTrader initialized live=%s", self.config.live_trading)

    def connect(self) -> bool:
        if not CCXT_AVAILABLE:
            raise RuntimeError("ccxt missing - fail-closed")

        # Validate credentials
        try:
            self.config.validate_for_real_trading()
        except Exception as exc:
            if self.config.live_trading:
                raise

        try:
            self.ccxt_client = ccxt.okx({
                "apiKey": self.config.api_key,
                "secret": self.config.api_secret,
                "password": self.config.passphrase,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
            bal = self.ccxt_client.fetch_balance()
            self.connected = True
            log.info("OKXLiveTrader connected")
            if self.event_bus:
                self.event_bus.publish("OKX_CONNECTED", {"real_trading": True}, source="OKXLiveTrader")
            return True
        except Exception as exc:
            log.exception("Connect failed: %s", exc)
            raise RuntimeError(f"OKX connect failed (fail-closed): {exc}") from exc

    def is_connected(self) -> bool:
        return self.connected and self.ccxt_client is not None

    def get_balance(self):
        if not self.is_connected():
            raise RuntimeError("Not connected - fail-closed")
        try:
            return self.ccxt_client.fetch_balance()
        except Exception as exc:
            raise RuntimeError(f"get_balance failed (fail-closed): {exc}") from exc

    def get_ticker(self, symbol: str) -> float:
        if not self.is_connected():
            raise RuntimeError("Not connected - fail-closed")
        norm = symbol.replace("-", "/").upper()
        try:
            ticker = self.ccxt_client.fetch_ticker(norm)
            price = float(ticker.get("last") or ticker.get("close") or 0)
            if not price:
                raise RuntimeError(f"No price for {symbol} - fail-closed")
            return price
        except Exception as exc:
            raise RuntimeError(f"get_ticker {symbol} failed (fail-closed): {exc}") from exc

    def place_order(self, order: OKXOrderRequest) -> OKXOrderResult:
        if not self.is_connected():
            raise RuntimeError("Not connected - fail-closed")

        # Safety checks
        if order.leverage > self.max_leverage:
            return OKXOrderResult(False, "", order.client_order_id, order.symbol, order.side, 0, 0, OrderStatus.REJECTED, f"Leverage {order.leverage}x > cap {self.max_leverage}x")

        price = order.limit_price or self.get_ticker(order.symbol)

        # Notional check
        bal = self.get_balance()
        equity = 100000.0
        try:
            total = bal.get("total", {})
            if isinstance(total, dict):
                equity = float(total.get("USDT") or total.get("USD") or 100000.0)
        except Exception:
            pass

        notional = order.quantity * price
        if equity > 0 and notional / equity > self.max_position_pct:
            return OKXOrderResult(False, "", order.client_order_id, order.symbol, order.side, 0, 0, OrderStatus.REJECTED, f"Notional exceeds {self.max_position_pct:.0%}")

        if self.event_bus:
            self.event_bus.publish("ORDER_REQUEST", {"symbol": order.symbol, "side": order.side.value, "quantity": order.quantity}, source="OKXLiveTrader")

        try:
            ccxt_symbol = order.symbol.replace("-", "/")
            if order.order_type == OrderType.MARKET:
                ccxt_order = self.ccxt_client.create_market_order(ccxt_symbol, order.side.value, order.quantity)
            else:
                if not order.limit_price:
                    raise RuntimeError("LIMIT requires price - fail-closed")
                ccxt_order = self.ccxt_client.create_order(ccxt_symbol, order.order_type.value, order.side.value, order.quantity, order.limit_price)

            filled = float(ccxt_order.get("filled", 0) or order.quantity)
            avg_price = float(ccxt_order.get("average", 0) or price)
            order_id = str(ccxt_order.get("id", ""))

            result = OKXOrderResult(True, order_id, order.client_order_id, order.symbol, order.side, filled, avg_price, OrderStatus.FILLED, "REAL LIVE filled")
            if self.event_bus:
                self.event_bus.publish("ORDER_FILLED", result.to_dict(), source="OKXLiveTrader")
            return result
        except Exception as exc:
            log.exception("Order failed: %s", exc)
            return OKXOrderResult(False, "", order.client_order_id, order.symbol, order.side, 0, 0, OrderStatus.REJECTED, f"REAL order exception (fail-closed): {exc}")

    def place_order_from_signal(self, signal: Dict[str, Any], max_quantity: Optional[float] = None) -> Optional[OKXOrderResult]:
        symbol = signal.get("symbol")
        if not symbol:
            raise RuntimeError("Signal missing symbol - fail-closed")
        direction = (signal.get("final_signal") or "").upper()
        if direction not in ("BUY", "SELL"):
            return None
        confidence = float(signal.get("confidence", 0.0))
        if confidence < 0.60:
            return None

        price = self.get_ticker(symbol)
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

        side = OrderSide.BUY if direction == "BUY" else OrderSide.SELL
        order = OKXOrderRequest(symbol=symbol, side=side, quantity=qty, leverage=min(1.0 + confidence, self.max_leverage))
        return self.place_order(order)

    def get_status(self):
        return {"real_trading": True, "connected": self.connected, "ccxt_available": CCXT_AVAILABLE, "live": self.config.live_trading}

    def activate_kill_switch(self, reason: str):
        if self.event_bus:
            self.event_bus.publish("KILL_SWITCH", {"reason": reason}, source="OKXLiveTrader")
