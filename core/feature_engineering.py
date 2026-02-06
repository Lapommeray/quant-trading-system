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
