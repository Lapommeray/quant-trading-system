"""Public API for the top-level quant trading package."""

from .config import settings
from .data_feeds.yfinance_feed import get_price_history
from .logging import configure_root_logger

__all__ = [
    "settings",
    "get_price_history",
    "configure_root_logger",
]
