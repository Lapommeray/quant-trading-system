"""
Data Pipeline - Unified Market Data Loader

Handles fetching, normalizing, and caching market data from multiple sources.
Provides a clean interface for both live and historical data access.

Supports:
- yfinance (live + historical)
- CSV files (backtesting)
- Multi-timeframe data (1h, 4h, daily)

No external AI. No sockets. File-based + HTTP only.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("DataPipeline")

_CACHE_TTL = 55
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

SYMBOL_MAP = {
    "XAUUSD": "GC=F",
    "XAGUSD": "SI=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "NZDUSD": "NZDUSD=X",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "SPX500": "^GSPC",
    "US30": "^DJI",
    "NASDAQ": "^IXIC",
    "US100": "^NDX",
}


class DataPipeline:
    """
    Unified data loader for the trading system.

    Caches fetched data to avoid redundant API calls within the same cycle.
    Normalizes all data into a standard OHLCV format.
    """

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, float] = {}

    def fetch(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Fetch multi-timeframe data for a symbol.

        Args:
            symbol: Trading symbol (e.g. "XAUUSD")
            timeframes: List of timeframes to fetch. Default: ["1h", "1d"]

        Returns:
            Dict with 'ohlcv' (primary), 'ohlcv_daily' (HTF), metadata
        """
        if timeframes is None:
            timeframes = ["1h", "1d"]

        cache_key = f"{symbol}_{'_'.join(timeframes)}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        result = self._fetch_yfinance(symbol, timeframes)

        if result is None:
            result = self._fetch_csv(symbol, timeframes)

        if result is None:
            logger.warning(f"No data available for {symbol}")
            return {"symbol": symbol, "ohlcv": [], "error": "no_data"}

        self._set_cache(cache_key, result)
        return result

    def fetch_historical(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data as a pandas DataFrame for backtesting.
        """
        try:
            import yfinance as yf
            yf_sym = SYMBOL_MAP.get(symbol.replace("/", ""), symbol)
            ticker = yf.Ticker(yf_sym)
            df = ticker.history(period=period, interval=interval)
            if not df.empty:
                return df
        except Exception as e:
            logger.debug(f"yfinance historical fetch failed for {symbol}: {e}")

        return self._load_csv_as_dataframe(symbol, interval)

    def _fetch_yfinance(
        self, symbol: str, timeframes: List[str]
    ) -> Optional[Dict[str, Any]]:
        try:
            import yfinance as yf
        except ImportError:
            return None

        clean = symbol.replace("/", "")
        yf_sym = SYMBOL_MAP.get(clean)
        if not yf_sym:
            return None

        ticker = yf.Ticker(yf_sym)
        result = {"symbol": symbol, "source": "yfinance"}

        tf_config = {
            "1h": ("5d", "1h", "ohlcv"),
            "4h": ("1mo", "1d", "ohlcv_4h"),
            "1d": ("3mo", "1d", "ohlcv_daily"),
        }

        primary_set = False
        for tf in timeframes:
            cfg = tf_config.get(tf)
            if cfg is None:
                continue

            period, interval, key = cfg
            try:
                hist = ticker.history(period=period, interval=interval)
                if hist.empty or len(hist) < 10:
                    continue

                records = hist.tail(80).to_dict("records")
                result[key] = records

                if not primary_set:
                    result["close"] = float(hist["Close"].iloc[-1])
                    result["volume"] = float(hist["Volume"].iloc[-1])
                    result["timestamp"] = hist.index[-1].isoformat()
                    result["bars"] = len(hist)
                    primary_set = True

            except Exception as e:
                logger.debug(f"Failed to fetch {tf} for {symbol}: {e}")

        if not primary_set:
            return None

        if "ohlcv" not in result and "ohlcv_daily" in result:
            result["ohlcv"] = result["ohlcv_daily"]

        return result

    def _fetch_csv(
        self, symbol: str, timeframes: List[str]
    ) -> Optional[Dict[str, Any]]:
        clean = symbol.replace("/", "")
        csv_dir = os.path.join(_DATA_DIR, "ticks")
        if not os.path.isdir(csv_dir):
            return None

        for fname in os.listdir(csv_dir):
            if clean.lower() in fname.lower() and fname.endswith(".csv"):
                path = os.path.join(csv_dir, fname)
                try:
                    df = pd.read_csv(path)
                    df = self._normalize_dataframe(df)
                    if df is not None and len(df) >= 10:
                        return {
                            "symbol": symbol,
                            "source": "csv",
                            "ohlcv": df.tail(80).to_dict("records"),
                            "close": float(df["Close"].iloc[-1]),
                            "volume": float(df.get("Volume", pd.Series([0])).iloc[-1]),
                            "timestamp": str(df.index[-1]),
                        }
                except Exception as e:
                    logger.debug(f"CSV load failed for {fname}: {e}")

        return None

    def _load_csv_as_dataframe(
        self, symbol: str, interval: str
    ) -> Optional[pd.DataFrame]:
        clean = symbol.replace("/", "")
        csv_dir = os.path.join(_DATA_DIR, "ticks")
        if not os.path.isdir(csv_dir):
            return None

        for fname in os.listdir(csv_dir):
            if clean.lower() in fname.lower() and fname.endswith(".csv"):
                path = os.path.join(csv_dir, fname)
                try:
                    df = pd.read_csv(path)
                    return self._normalize_dataframe(df)
                except Exception:
                    pass
        return None

    def _normalize_dataframe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        col_map = {}
        for col in df.columns:
            lc = col.lower().strip()
            if lc in ("open", "o"):
                col_map[col] = "Open"
            elif lc in ("high", "h"):
                col_map[col] = "High"
            elif lc in ("low", "l"):
                col_map[col] = "Low"
            elif lc in ("close", "c"):
                col_map[col] = "Close"
            elif lc in ("volume", "vol", "v"):
                col_map[col] = "Volume"

        if col_map:
            df = df.rename(columns=col_map)

        required = ["Open", "High", "Low", "Close"]
        if not all(c in df.columns for c in required):
            return None

        for c in required + ["Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if "Volume" not in df.columns:
            df["Volume"] = 0

        df = df.dropna(subset=required)
        return df.reset_index(drop=True)

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        ts = self._cache_timestamps.get(key, 0)
        if time.time() - ts < _CACHE_TTL and key in self._cache:
            return self._cache[key]
        return None

    def _set_cache(self, key: str, data: Dict[str, Any]):
        self._cache[key] = data
        self._cache_timestamps[key] = time.time()
