"""Market data adapter backed by yfinance.

Despite the legacy module name, this adapter now uses yfinance as the data source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Candle:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class YFinanceAdapter:
    """Fetch OHLCV data through yfinance."""

    def __init__(self) -> None:
        try:
            import yfinance as yf  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("yfinance is required for market data access") from exc
        self._yf = yf

    def get_ohlcv(self, symbol: str, period: str = "1mo", interval: str = "1d") -> List[Dict[str, Any]]:
        ticker = self._yf.Ticker(symbol)
        history = ticker.history(period=period, interval=interval)
        rows: List[Dict[str, Any]] = []
        for ts, row in history.iterrows():
            rows.append(
                {
                    "timestamp": ts.isoformat(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"]),
                }
            )
        return rows


# Backward-compatible alias for code still importing the old class name.
AlphaVantageAdapter = YFinanceAdapter
