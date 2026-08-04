"""
OKX Live Trading Package - Real trading, fails closed.

No simulation fallback. Requires:
- ccxt package
- OKX_API_KEY, OKX_API_SECRET, OKX_PASSPHRASE env vars
- Real market data (yfinance or OKX ticker) - no synthetic fallback for live runner

For paper/simulation testing, use execution/okx_engine.py (old path) which is explicitly marked as simulation.
This package is for real trading only.
"""

from .engine import OKXLiveEngine, OKXOrderRequest, OKXOrderResult, OrderSide, OrderType
from .safety import OKXSafetyGuard
from .config import OKXLiveConfig, get_okx_config
from .trader import OKXLiveTrader

__all__ = [
    "OKXLiveEngine",
    "OKXOrderRequest",
    "OKXOrderResult",
    "OrderSide",
    "OrderType",
    "OKXSafetyGuard",
    "OKXLiveConfig",
    "get_okx_config",
    "OKXLiveTrader",
]
