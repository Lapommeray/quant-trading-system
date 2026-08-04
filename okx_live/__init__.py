"""
OKX Live Trading Package - Real trading, fail-closed.

Expected files per reviewer:
- okx_live/config.py
- okx_live/trader.py

No simulation fallback. Requires ccxt + credentials + real market data.
"""

from .config import OKXLiveConfig, get_okx_config
from .trader import OKXLiveTrader, OKXOrderRequest, OKXOrderResult, OrderSide, OrderType

__all__ = [
    "OKXLiveConfig",
    "get_okx_config",
    "OKXLiveTrader",
    "OKXOrderRequest",
    "OKXOrderResult",
    "OrderSide",
    "OrderType",
]
