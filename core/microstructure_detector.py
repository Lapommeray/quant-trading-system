"""
Microstructure Detector

Extracts hidden market features from price action that typical indicators miss.
Detects institutional footprints, liquidity traps, and regime shifts.

No external AI. Pure mathematical analysis of price/volume microstructure.

Features detected:
- Depth imbalance (buy vs sell pressure from volume analysis)
- Momentum divergence (price vs volume disagreement)
- Fractal density (self-similar price patterns)
- Entropy shift (disorder/uncertainty in price distribution)
- Liquidity gaps (sudden price jumps indicating thin liquidity)
- Absorption detection (large volume with small price change)
- Trend exhaustion (diminishing returns on trend continuation)
"""

import logging
import math
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("Microstructure")


class MicrostructureDetector:
    """
    Extracts microstructure features from OHLCV data.

    These features capture market dynamics invisible to standard indicators:
    - Where is hidden buying/selling pressure?
    - Is the current move exhausting?
    - Are institutions accumulating or distributing?
    - Is the market about to shift regime?
    """

    def extract_features(self, df: pd.DataFrame) -> Dict[str, float]:
        if df is None or len(df) < 10:
            return self._empty_features()

        features = {}

        features["depth_imbalance"] = self._calc_depth_imbalance(df)
        features["momentum_divergence"] = self._calc_momentum_divergence(df)
        features["fractal_density"] = self._calc_fractal_density(df)
        features["entropy_shift"] = self._calc_entropy_shift(df)
        features["liquidity_gaps"] = self._calc_liquidity_gaps(df)
        features["absorption_ratio"] = self._calc_absorption(df)
        features["trend_exhaustion"] = self._calc_trend_exhaustion(df)
        features["volatility"] = self._calc_volatility(df)
        features["price_acceleration"] = self._calc_price_acceleration(df)

        return features

    def get_signal_bias(self, features: Dict[str, float]) -> Tuple[Optional[str], float]:
        if not features:
            return None, 0.0

        bull_score = 0.0
        bear_score = 0.0

        di = features.get("depth_imbalance", 0)
        if di > 0.2:
            bull_score += 0.3
        elif di < -0.2:
            bear_score += 0.3

        md = features.get("momentum_divergence", 0)
        if md > 0.3:
            bear_score += 0.25
        elif md < -0.3:
            bull_score += 0.25

        te = features.get("trend_exhaustion", 0)
        if te > 0.7:
            bear_score += 0.2 if features.get("depth_imbalance", 0) > 0 else 0
            bull_score += 0.2 if features.get("depth_imbalance", 0) < 0 else 0

        ar = features.get("absorption_ratio", 0)
        if ar > 0.8:
            if features.get("depth_imbalance", 0) > 0:
                bull_score += 0.15
            else:
                bear_score += 0.15

        lg = features.get("liquidity_gaps", 0)
        if lg > 2:
            bear_score += 0.1
            bull_score += 0.1

        if bull_score > bear_score + 0.15:
            return "BUY", min(0.9, 0.5 + bull_score)
        elif bear_score > bull_score + 0.15:
            return "SELL", min(0.9, 0.5 + bear_score)
        return "HOLD", 0.4

    def _empty_features(self) -> Dict[str, float]:
        return {
            "depth_imbalance": 0.0,
            "momentum_divergence": 0.0,
            "fractal_density": 0.0,
            "entropy_shift": 0.0,
            "liquidity_gaps": 0.0,
            "absorption_ratio": 0.0,
            "trend_exhaustion": 0.0,
            "volatility": 0.0,
            "price_acceleration": 0.0,
        }

    def _calc_depth_imbalance(self, df: pd.DataFrame) -> float:
        close = df["Close"].values
        volume = df["Volume"].values

        if volume.sum() == 0:
            return 0.0

        buy_vol = 0.0
        sell_vol = 0.0
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                buy_vol += volume[i]
            elif close[i] < close[i - 1]:
                sell_vol += volume[i]

        total = buy_vol + sell_vol
        if total == 0:
            return 0.0
        return float((buy_vol - sell_vol) / total)

    def _calc_momentum_divergence(self, df: pd.DataFrame) -> float:
        close = df["Close"].values
        volume = df["Volume"].values

        if len(close) < 10 or volume.sum() == 0:
            return 0.0

        half = len(close) // 2
        price_trend_1 = (close[half] - close[0]) / close[0] if close[0] != 0 else 0
        price_trend_2 = (close[-1] - close[half]) / close[half] if close[half] != 0 else 0

        vol_1 = volume[:half].mean()
        vol_2 = volume[half:].mean()
        vol_trend = (vol_2 - vol_1) / vol_1 if vol_1 != 0 else 0

        price_accel = price_trend_2 - price_trend_1

        if abs(price_accel) < 1e-6:
            return 0.0

        if price_accel > 0 and vol_trend < 0:
            return float(abs(price_accel) * 10)
        elif price_accel < 0 and vol_trend > 0:
            return float(-abs(price_accel) * 10)

        return 0.0

    def _calc_fractal_density(self, df: pd.DataFrame) -> float:
        close = df["Close"].values
        if len(close) < 10:
            return 0.0

        reversals = 0
        for i in range(2, len(close)):
            d1 = close[i - 1] - close[i - 2]
            d2 = close[i] - close[i - 1]
            if d1 * d2 < 0:
                reversals += 1

        return float(reversals / (len(close) - 2))

    def _calc_entropy_shift(self, df: pd.DataFrame) -> float:
        close = df["Close"].values
        if len(close) < 20:
            return 0.0

        returns = np.diff(close) / close[:-1]
        returns = returns[~np.isnan(returns)]
        if len(returns) < 10:
            return 0.0

        half = len(returns) // 2
        ent_1 = self._shannon_entropy(returns[:half])
        ent_2 = self._shannon_entropy(returns[half:])

        return float(ent_2 - ent_1)

    def _shannon_entropy(self, data: np.ndarray, bins: int = 10) -> float:
        if len(data) < 2:
            return 0.0
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        probs = hist / hist.sum()
        return float(-np.sum(probs * np.log2(probs + 1e-10)))

    def _calc_liquidity_gaps(self, df: pd.DataFrame) -> float:
        high = df["High"].values
        low = df["Low"].values
        close = df["Close"].values

        if len(close) < 5:
            return 0.0

        atr = np.mean(high[-14:] - low[-14:]) if len(high) >= 14 else np.mean(high - low)
        if atr == 0:
            return 0.0

        gaps = 0
        for i in range(1, len(close)):
            gap = abs(close[i] - close[i - 1])
            if gap > atr * 1.5:
                gaps += 1

        return float(gaps)

    def _calc_absorption(self, df: pd.DataFrame) -> float:
        close = df["Close"].values
        volume = df["Volume"].values

        if len(close) < 5 or volume.sum() == 0:
            return 0.0

        recent_vol = volume[-5:].mean()
        recent_range = np.mean(np.abs(np.diff(close[-5:])))

        if recent_range == 0:
            return 1.0

        avg_vol = volume.mean()
        avg_range = np.mean(np.abs(np.diff(close)))

        if avg_range == 0 or avg_vol == 0:
            return 0.0

        vol_ratio = recent_vol / avg_vol
        range_ratio = recent_range / avg_range

        if range_ratio == 0:
            return 1.0

        return float(min(1.0, vol_ratio / range_ratio / 3))

    def _calc_trend_exhaustion(self, df: pd.DataFrame) -> float:
        close = df["Close"].values
        if len(close) < 15:
            return 0.0

        third = len(close) // 3
        seg1 = (close[third] - close[0]) / close[0] if close[0] != 0 else 0
        seg2 = (close[2 * third] - close[third]) / close[third] if close[third] != 0 else 0
        seg3 = (close[-1] - close[2 * third]) / close[2 * third] if close[2 * third] != 0 else 0

        if abs(seg1) < 1e-8:
            return 0.0

        if seg1 > 0 and seg2 > 0 and seg3 > 0:
            if seg3 < seg2 < seg1:
                return float(min(1.0, 1.0 - seg3 / seg1))
        elif seg1 < 0 and seg2 < 0 and seg3 < 0:
            if seg3 > seg2 > seg1:
                return float(min(1.0, 1.0 - abs(seg3) / abs(seg1)))

        return 0.0

    def _calc_volatility(self, df: pd.DataFrame) -> float:
        close = df["Close"].values
        if len(close) < 5:
            return 0.0
        returns = np.diff(close) / close[:-1]
        returns = returns[~np.isnan(returns)]
        if len(returns) == 0:
            return 0.0
        return float(np.std(returns))

    def _calc_price_acceleration(self, df: pd.DataFrame) -> float:
        close = df["Close"].values
        if len(close) < 10:
            return 0.0

        half = len(close) // 2
        vel_1 = (close[half] - close[0]) / half if half > 0 else 0
        vel_2 = (close[-1] - close[half]) / (len(close) - half) if (len(close) - half) > 0 else 0

        if close[0] == 0:
            return 0.0

        return float((vel_2 - vel_1) / close[0] * 100)
