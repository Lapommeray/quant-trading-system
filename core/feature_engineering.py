"""
Feature Engineering - Unified Feature Extraction

Extracts all features needed by the Unified Intelligence Core from raw OHLCV data.
Provides both real-time feature extraction and historical stream generation for backtesting.

Features extracted:
- Trend: SMA spread, price acceleration
- Momentum: RSI, MACD histogram, rate of change
- Volatility: ATR, Bollinger width, realized vol
- Volume: depth imbalance, absorption, OBV trend
- Microstructure: fractal density, entropy shift, liquidity gaps
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple, Generator

import numpy as np
import pandas as pd

logger = logging.getLogger("FeatureEng")


class FeatureEngineer:
    """
    Extracts a standardized feature vector from OHLCV data.

    Used by:
    - UnifiedIntelligenceCore (live trading)
    - Backtest scripts (historical evaluation)
    - RL agent (state discretization)
    - Bayesian filter (evidence)
    """

    def extract(self, df: pd.DataFrame) -> Dict[str, float]:
        if df is None or len(df) < 20:
            return self._empty()

        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values
        volume = df["Volume"].values if "Volume" in df.columns else np.zeros(len(df))

        features = {}

        features["trend_spread"] = self._sma_spread(close, 10, 30)
        features["price_acceleration"] = self._price_acceleration(close)
        features["momentum"] = self._momentum(close, 14)

        features["rsi"] = self._rsi(close, 14)
        features["macd_histogram"] = self._macd_histogram(close)
        features["roc"] = self._rate_of_change(close, 10)

        features["atr"] = self._atr(high, low, close, 14)
        features["volatility"] = self._realized_vol(close, 14)
        features["bollinger_width"] = self._bollinger_width(close, 20)

        features["depth_imbalance"] = self._depth_imbalance(close, volume)
        features["obv_trend"] = self._obv_trend(close, volume)
        features["absorption"] = self._absorption(close, volume)

        features["fractal_density"] = self._fractal_density(close)
        features["entropy_shift"] = self._entropy_shift(close)

        features["adx"] = self._adx(high, low, close, 14)

        features["liquidity_gap"] = self._liquidity_gap(high, low, close)
        features["trend_exhaustion"] = self._trend_exhaustion(close)
        features["volatility_regime"] = self._volatility_regime(close)
        features["mean_reversion_score"] = self._mean_reversion_score(close)
        features["bar_range_ratio"] = self._bar_range_ratio(high, low, close)

        return features

    def load_features_stream(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None,
        window: int = 50,
    ) -> Generator[Tuple[Dict[str, float], str], None, None]:
        """
        Generate a stream of (features, true_direction) tuples for backtesting.

        Slides a window across the dataframe. For each position:
        - features are computed from the window
        - true_direction is determined by the NEXT bar's close vs current close

        Yields:
            (features_dict, "BUY" | "SELL" | "HOLD")
        """
        if df is None:
            df = self._load_default_data(symbol)
            if df is None:
                return

        if len(df) < window + 5:
            logger.warning(f"Not enough data for {symbol}: {len(df)} bars, need {window + 5}")
            return

        for i in range(window, len(df) - 1):
            window_df = df.iloc[i - window:i].copy().reset_index(drop=True)
            features = self.extract(window_df)

            current_close = float(df["Close"].iloc[i])
            next_close = float(df["Close"].iloc[i + 1])

            if current_close == 0:
                continue

            change = (next_close - current_close) / current_close
            if change > 0.0002:
                true_dir = "BUY"
            elif change < -0.0002:
                true_dir = "SELL"
            else:
                true_dir = "HOLD"

            yield features, true_dir

    def _load_default_data(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "data_pipeline",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_pipeline.py")
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            pipeline = mod.DataPipeline()
            return pipeline.fetch_historical(symbol, period="1y", interval="1d")
        except Exception as e:
            logger.warning(f"Could not load default data for {symbol}: {e}")
            return None

    def _empty(self) -> Dict[str, float]:
        return {
            "trend_spread": 0.0, "price_acceleration": 0.0,
            "momentum": 0.0, "rsi": 50.0, "macd_histogram": 0.0,
            "roc": 0.0, "atr": 0.0, "volatility": 0.01,
            "bollinger_width": 0.0, "depth_imbalance": 0.0,
            "obv_trend": 0.0, "absorption": 0.0,
            "fractal_density": 0.0, "entropy_shift": 0.0, "adx": 20.0,
            "liquidity_gap": 0.0, "trend_exhaustion": 0.0,
            "volatility_regime": 0.5, "mean_reversion_score": 0.0,
            "bar_range_ratio": 1.0,
        }

    def _sma_spread(self, close: np.ndarray, fast: int, slow: int) -> float:
        if len(close) < slow:
            return 0.0
        sma_f = np.mean(close[-fast:])
        sma_s = np.mean(close[-slow:])
        if sma_s == 0:
            return 0.0
        return float((sma_f - sma_s) / sma_s)

    def _price_acceleration(self, close: np.ndarray) -> float:
        if len(close) < 10:
            return 0.0
        half = len(close) // 2
        vel_1 = (close[half] - close[0]) / max(half, 1)
        vel_2 = (close[-1] - close[half]) / max(len(close) - half, 1)
        if close[0] == 0:
            return 0.0
        return float((vel_2 - vel_1) / close[0] * 100)

    def _momentum(self, close: np.ndarray, period: int) -> float:
        if len(close) < period + 1:
            return 0.0
        if close[-period - 1] == 0:
            return 0.0
        return float((close[-1] - close[-period - 1]) / close[-period - 1])

    def _rsi(self, close: np.ndarray, period: int = 14) -> float:
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    def _macd_histogram(self, close: np.ndarray) -> float:
        if len(close) < 26:
            return 0.0
        s = pd.Series(close)
        ema12 = s.ewm(span=12, adjust=False).mean()
        ema26 = s.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return float((macd.iloc[-1] - signal.iloc[-1]))

    def _rate_of_change(self, close: np.ndarray, period: int) -> float:
        if len(close) < period + 1 or close[-period - 1] == 0:
            return 0.0
        return float((close[-1] - close[-period - 1]) / close[-period - 1] * 100)

    def _atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
        if len(high) < period + 1:
            return 0.0
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        return float(np.mean(tr[-period:]))

    def _realized_vol(self, close: np.ndarray, period: int) -> float:
        if len(close) < period + 1:
            return 0.01
        returns = np.diff(close[-period - 1:]) / close[-period - 1:-1]
        returns = returns[~np.isnan(returns)]
        if len(returns) == 0:
            return 0.01
        return float(np.std(returns))

    def _bollinger_width(self, close: np.ndarray, period: int) -> float:
        if len(close) < period:
            return 0.0
        sma = np.mean(close[-period:])
        std = np.std(close[-period:])
        if sma == 0:
            return 0.0
        return float(2 * std / sma)

    def _depth_imbalance(self, close: np.ndarray, volume: np.ndarray) -> float:
        if volume.sum() == 0 or len(close) < 2:
            return 0.0
        buy_v, sell_v = 0.0, 0.0
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                buy_v += volume[i]
            elif close[i] < close[i - 1]:
                sell_v += volume[i]
        total = buy_v + sell_v
        if total == 0:
            return 0.0
        return float((buy_v - sell_v) / total)

    def _obv_trend(self, close: np.ndarray, volume: np.ndarray) -> float:
        if volume.sum() == 0 or len(close) < 10:
            return 0.0
        obv = np.zeros(len(close))
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]
        obv_sma = np.mean(obv[-10:])
        if obv_sma == 0:
            return 0.0
        return float((obv[-1] - obv_sma) / (abs(obv_sma) + 1e-10))

    def _absorption(self, close: np.ndarray, volume: np.ndarray) -> float:
        if len(close) < 5 or volume.sum() == 0:
            return 0.0
        recent_vol = np.mean(volume[-5:])
        recent_range = np.mean(np.abs(np.diff(close[-5:])))
        avg_vol = np.mean(volume)
        avg_range = np.mean(np.abs(np.diff(close)))
        if avg_range == 0 or avg_vol == 0 or recent_range == 0:
            return 0.0
        vol_ratio = recent_vol / avg_vol
        range_ratio = recent_range / avg_range
        return float(min(1.0, vol_ratio / (range_ratio + 1e-10) / 3))

    def _fractal_density(self, close: np.ndarray) -> float:
        if len(close) < 10:
            return 0.0
        reversals = 0
        for i in range(2, len(close)):
            d1 = close[i - 1] - close[i - 2]
            d2 = close[i] - close[i - 1]
            if d1 * d2 < 0:
                reversals += 1
        return float(reversals / (len(close) - 2))

    def _entropy_shift(self, close: np.ndarray) -> float:
        if len(close) < 20:
            return 0.0
        returns = np.diff(close) / close[:-1]
        returns = returns[~np.isnan(returns)]
        if len(returns) < 10:
            return 0.0
        half = len(returns) // 2
        e1 = self._entropy(returns[:half])
        e2 = self._entropy(returns[half:])
        return float(e2 - e1)

    def _entropy(self, data: np.ndarray, bins: int = 10) -> float:
        if len(data) < 2:
            return 0.0
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        probs = hist / hist.sum()
        return float(-np.sum(probs * np.log2(probs + 1e-10)))

    def _adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
        if len(high) < period + 2:
            return 20.0

        plus_dm = np.diff(high)
        minus_dm = -np.diff(low)

        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )

        atr = np.mean(tr[-period:])
        if atr == 0:
            return 20.0

        plus_di = 100 * np.mean(plus_dm[-period:]) / atr
        minus_di = 100 * np.mean(minus_dm[-period:]) / atr

        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 20.0

        dx = 100 * abs(plus_di - minus_di) / di_sum
        return float(dx)

    def _liquidity_gap(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        if len(high) < 10:
            return 0.0
        gaps = 0
        for i in range(1, len(high)):
            if low[i] > high[i - 1]:
                gaps += 1
            elif high[i] < low[i - 1]:
                gaps += 1
        return float(gaps / (len(high) - 1))

    def _trend_exhaustion(self, close: np.ndarray) -> float:
        if len(close) < 20:
            return 0.0
        returns = np.diff(close[-20:]) / close[-20:-1]
        returns = returns[~np.isnan(returns)]
        if len(returns) < 10:
            return 0.0
        recent = returns[-5:]
        earlier = returns[:-5]
        recent_abs = np.mean(np.abs(recent))
        earlier_abs = np.mean(np.abs(earlier))
        if earlier_abs == 0:
            return 0.0
        ratio = recent_abs / earlier_abs
        if ratio < 0.5:
            return float(1.0 - ratio)
        return 0.0

    def _volatility_regime(self, close: np.ndarray) -> float:
        if len(close) < 30:
            return 0.5
        returns = np.diff(close) / close[:-1]
        returns = returns[~np.isnan(returns)]
        if len(returns) < 20:
            return 0.5
        short_vol = np.std(returns[-10:])
        long_vol = np.std(returns[-30:])
        if long_vol == 0:
            return 0.5
        return float(min(2.0, short_vol / long_vol))

    def _mean_reversion_score(self, close: np.ndarray) -> float:
        if len(close) < 20:
            return 0.0
        sma = np.mean(close[-20:])
        std = np.std(close[-20:])
        if std == 0:
            return 0.0
        zscore = (close[-1] - sma) / std
        if abs(zscore) > 2.0:
            return float(-np.sign(zscore) * min(1.0, (abs(zscore) - 2.0) / 2.0))
        return 0.0

    def _bar_range_ratio(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        if len(high) < 5:
            return 1.0
        ranges = high[-5:] - low[-5:]
        avg_range = np.mean(ranges)
        if avg_range == 0:
            return 1.0
        current_range = high[-1] - low[-1]
        return float(current_range / avg_range)


def compute_features(bar: Dict[str, Any]) -> Dict[str, float]:
    """
    Convenience function: extract features from a single bar dict.
    Used by the intraday pipeline when feeding tick-aggregated bars.
    """
    features: Dict[str, float] = {}
    features["momentum"] = bar.get("features", {}).get("bar_momentum", 0.0)
    features["vol_slope"] = bar.get("features", {}).get("vol_slope", 0.0)
    features["volume_delta"] = bar.get("features", {}).get("volume_delta", 0.0)
    features["volatility"] = bar.get("features", {}).get("bar_volatility", 0.01)
    features["depth_imbalance"] = bar.get("features", {}).get("depth_imbalance", 0.0)
    features["relative_volume"] = bar.get("features", {}).get("relative_volume", 1.0)
    features["short_momentum"] = bar.get("features", {}).get("short_momentum", 0.0)

    o = bar.get("Open", 0)
    h = bar.get("High", 0)
    l = bar.get("Low", 0)
    c = bar.get("Close", 0)
    if o != 0:
        features["bar_direction"] = float((c - o) / o)
    else:
        features["bar_direction"] = 0.0
    if h != l:
        features["wick_ratio"] = float((h - max(o, c)) + (min(o, c) - l)) / (h - l)
    else:
        features["wick_ratio"] = 0.0

    return features
