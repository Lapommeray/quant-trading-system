"""
Data Pipeline - Unified Market Data Loader

Handles fetching, normalizing, and caching market data from multiple sources.
Provides a clean interface for both live and historical data access.

Supports:
- yfinance (live + historical)
- CSV files (backtesting)
- Multi-timeframe data (1m, 5m, 15m, 1h, 4h, daily)
- Intraday tick aggregation with session context
- Rolling z-score normalization for regime sensitivity

No external AI. No sockets. File-based + HTTP only.
"""

import os
import time
import logging
import math
from collections import deque
from datetime import datetime, timedelta
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

    def fetch_intraday(
        self,
        symbol: str,
        interval: str = "5m",
        lookback_bars: int = 240,
    ) -> Optional[Dict[str, Any]]:
        cache_key = f"{symbol}_intraday_{interval}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            import yfinance as yf
        except ImportError:
            return None

        clean = symbol.replace("/", "")
        yf_sym = SYMBOL_MAP.get(clean)
        if not yf_sym:
            return None

        period_map = {
            "1m": "1d", "2m": "5d", "5m": "5d",
            "15m": "5d", "30m": "5d", "1h": "5d",
        }
        period = period_map.get(interval, "5d")

        try:
            ticker = yf.Ticker(yf_sym)
            hist = ticker.history(period=period, interval=interval)
            if hist.empty or len(hist) < 20:
                return None

            hist_daily = ticker.history(period="3mo", interval="1d")

            result = {
                "symbol": symbol,
                "source": "yfinance",
                "interval": interval,
                "ohlcv": hist.tail(lookback_bars).to_dict("records"),
                "close": float(hist["Close"].iloc[-1]),
                "volume": float(hist["Volume"].iloc[-1]),
                "timestamp": hist.index[-1].isoformat(),
                "bars": len(hist),
            }

            if not hist_daily.empty and len(hist_daily) >= 30:
                result["ohlcv_daily"] = hist_daily.tail(60).to_dict("records")

            self._set_cache(cache_key, result)
            return result
        except Exception as e:
            logger.debug(f"Intraday fetch failed for {symbol}: {e}")
            return None


class IntradayDataPipeline:
    """
    Real-time tick aggregation pipeline for intraday trading.

    Aggregates raw ticks into completed bars (1m / 5m),
    computes rolling z-score normalization, and tracks session boundaries.
    """

    def __init__(
        self,
        symbols: List[str],
        bar_interval: int = 60,
        lookback: int = 240,
        session_reset_hour: int = 9,
        zscore_window: int = 120,
    ):
        self.symbols = symbols
        self.bar_interval = bar_interval
        self.lookback = lookback
        self.session_reset_hour = session_reset_hour
        self.zscore_window = zscore_window

        self.tick_buffers: Dict[str, list] = {s: [] for s in symbols}
        self.bar_data: Dict[str, deque] = {
            s: deque(maxlen=lookback) for s in symbols
        }
        self.last_bar_time: Dict[str, float] = {s: 0.0 for s in symbols}
        self.last_session_reset: Dict[str, datetime] = {
            s: datetime.now() for s in symbols
        }
        self.session_stats: Dict[str, Dict[str, float]] = {
            s: {"high": 0, "low": float("inf"), "volume": 0, "trades": 0}
            for s in symbols
        }

        logger.info(f"IntradayDataPipeline initialized for {symbols}, bar={bar_interval}s")

    def process_tick(self, tick: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        symbol = tick.get("symbol", self.symbols[0] if self.symbols else "UNKNOWN")
        if symbol not in self.tick_buffers:
            self.tick_buffers[symbol] = []
            self.bar_data[symbol] = deque(maxlen=self.lookback)
            self.last_bar_time[symbol] = 0.0

        self.tick_buffers[symbol].append(tick)

        price = tick.get("price", 0)
        stats = self.session_stats.get(symbol, {"high": 0, "low": float("inf"), "volume": 0, "trades": 0})
        if price > stats["high"]:
            stats["high"] = price
        if price < stats["low"]:
            stats["low"] = price
        stats["volume"] += tick.get("volume", 0)
        stats["trades"] += 1
        self.session_stats[symbol] = stats

        now = time.time()
        elapsed = now - self.last_bar_time.get(symbol, 0)

        if elapsed >= self.bar_interval and len(self.tick_buffers[symbol]) >= 2:
            bar = self._aggregate_bar(symbol)
            self.bar_data[symbol].append(bar)
            self.tick_buffers[symbol] = []
            self.last_bar_time[symbol] = now

            dt = datetime.now()
            if (
                dt.hour == self.session_reset_hour
                and dt - self.last_session_reset.get(symbol, datetime.min) > timedelta(hours=23)
            ):
                self._reset_session(symbol)

            return bar

        return None

    def _aggregate_bar(self, symbol: str) -> Dict[str, Any]:
        ticks = self.tick_buffers[symbol]
        prices = [t.get("price", 0) for t in ticks]
        volumes = [t.get("volume", 0) for t in ticks]

        if not prices:
            return {"Open": 0, "High": 0, "Low": 0, "Close": 0, "Volume": 0}

        bar = {
            "Open": prices[0],
            "High": max(prices),
            "Low": min(prices),
            "Close": prices[-1],
            "Volume": sum(volumes),
            "timestamp": datetime.now().isoformat(),
            "tick_count": len(ticks),
        }

        bar["features"] = self._compute_bar_features(prices, volumes, symbol)
        return bar

    def _compute_bar_features(
        self, prices: List[float], volumes: List[float], symbol: str
    ) -> Dict[str, float]:
        p = np.array(prices, dtype=np.float64)
        v = np.array(volumes, dtype=np.float64)

        features: Dict[str, float] = {}

        if len(p) >= 2 and p[0] != 0:
            features["bar_momentum"] = float((p[-1] - p[0]) / p[0])
        else:
            features["bar_momentum"] = 0.0

        if len(p) >= 3:
            coeffs = np.polyfit(range(len(p)), p, 1)
            features["vol_slope"] = float(coeffs[0])
        else:
            features["vol_slope"] = 0.0

        mean_v = np.mean(v) if len(v) > 0 else 0
        features["volume_delta"] = float(v[-1] - mean_v) if len(v) > 0 else 0.0

        mean_p = np.mean(p)
        features["bar_volatility"] = float(np.std(p) / mean_p) if mean_p != 0 else 0.0

        buy_vol = sum(v[i] for i in range(1, len(p)) if p[i] > p[i - 1])
        sell_vol = sum(v[i] for i in range(1, len(p)) if p[i] < p[i - 1])
        total_vol = buy_vol + sell_vol
        features["depth_imbalance"] = float((buy_vol - sell_vol) / total_vol) if total_vol > 0 else 0.0

        bars = list(self.bar_data.get(symbol, []))
        if len(bars) >= 5:
            recent_closes = [b.get("Close", 0) for b in bars[-5:]]
            recent_vols = [b.get("Volume", 0) for b in bars[-5:]]
            avg_vol = np.mean(recent_vols) if recent_vols else 0
            features["relative_volume"] = float(sum(volumes) / avg_vol) if avg_vol > 0 else 1.0
            rc = np.array(recent_closes)
            if len(rc) >= 2:
                features["short_momentum"] = float((rc[-1] - rc[0]) / rc[0]) if rc[0] != 0 else 0.0
            else:
                features["short_momentum"] = 0.0
        else:
            features["relative_volume"] = 1.0
            features["short_momentum"] = 0.0

        return features

    def get_rolling_zscore(
        self, symbol: str, field: str = "Close"
    ) -> Optional[float]:
        bars = list(self.bar_data.get(symbol, []))
        if len(bars) < 10:
            return None

        window = min(self.zscore_window, len(bars))
        values = [b.get(field, 0) for b in bars[-window:]]
        arr = np.array(values, dtype=np.float64)
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0
        return float((arr[-1] - mean) / std)

    def get_session_context(self, symbol: str) -> Dict[str, float]:
        stats = self.session_stats.get(symbol, {})
        bars = list(self.bar_data.get(symbol, []))

        context: Dict[str, float] = {
            "session_high": stats.get("high", 0),
            "session_low": stats.get("low", 0),
            "session_volume": stats.get("volume", 0),
            "session_trades": stats.get("trades", 0),
            "bars_in_session": len(bars),
        }

        if stats.get("high", 0) > 0 and stats.get("low", float("inf")) < float("inf"):
            context["session_range"] = stats["high"] - stats["low"]
        else:
            context["session_range"] = 0

        if len(bars) >= 10:
            closes = [b.get("Close", 0) for b in bars[-10:]]
            context["session_volatility"] = float(np.std(closes) / np.mean(closes)) if np.mean(closes) != 0 else 0
        else:
            context["session_volatility"] = 0

        return context

    def get_latest_features(self, symbol: str) -> Optional[Dict[str, float]]:
        bars = list(self.bar_data.get(symbol, []))
        if not bars:
            return None
        return bars[-1].get("features", {})

    def get_bar_dataframe(self, symbol: str) -> Optional[pd.DataFrame]:
        bars = list(self.bar_data.get(symbol, []))
        if len(bars) < 5:
            return None
        df = pd.DataFrame(bars)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _reset_session(self, symbol: str):
        self.session_stats[symbol] = {
            "high": 0, "low": float("inf"), "volume": 0, "trades": 0
        }
        self.last_session_reset[symbol] = datetime.now()
        logger.info(f"Session reset for {symbol}")
