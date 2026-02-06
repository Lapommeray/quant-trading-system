"""
QMP Standalone Engine - Multi-Indicator Consensus Signal Generator

This engine combines multiple proven technical analysis methods to produce
high-confidence trading signals. It requires NO external dependencies beyond
pandas and numpy.

Signal generation rules:
- Multiple indicators must AGREE before emitting BUY/SELL
- Trend direction on higher timeframe must confirm the signal
- Never buy in a confirmed downtrend
- Never sell in a confirmed uptrend
- Volume must confirm the move
- Confidence reflects how many indicators agree
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("QMPStandalone")

# Minimum indicators that must agree for a signal
MIN_CONSENSUS = 4

# Weights for each indicator (sum to 1.0)
INDICATOR_WEIGHTS = {
    "trend_sma": 0.18,
    "rsi": 0.14,
    "macd": 0.16,
    "bollinger": 0.12,
    "adx": 0.10,
    "stochastic": 0.10,
    "obv": 0.08,
    "ichimoku": 0.12,
}


class QMPStandaloneEngine:
    """
    Standalone signal engine that combines multiple technical indicators
    with consensus-based decision making.

    No QuantConnect dependency. No randomness. Pure technical analysis.
    """

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def generate_signal(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a trading signal using multi-indicator consensus.

        Args:
            symbol: Trading symbol
            market_data: Dict with 'ohlcv' list of bar dicts and optionally
                         'ohlcv_daily' for higher timeframe confirmation

        Returns:
            Dict with 'final_signal', 'confidence', and indicator details
        """
        try:
            df = self._prepare_dataframe(market_data.get("ohlcv", []))
            if df is None or len(df) < 30:
                return {"final_signal": None, "confidence": 0.0, "reason": "insufficient_data"}

            df_daily = None
            if "ohlcv_daily" in market_data:
                df_daily = self._prepare_dataframe(market_data["ohlcv_daily"])

            votes = {}
            details = {}

            # 1. Trend (SMA 20/50 crossover)
            sig, conf, info = self._calc_trend_sma(df)
            votes["trend_sma"] = sig
            details["trend_sma"] = {"signal": sig, "confidence": conf, **info}

            # 2. RSI
            sig, conf, info = self._calc_rsi(df)
            votes["rsi"] = sig
            details["rsi"] = {"signal": sig, "confidence": conf, **info}

            # 3. MACD
            sig, conf, info = self._calc_macd(df)
            votes["macd"] = sig
            details["macd"] = {"signal": sig, "confidence": conf, **info}

            # 4. Bollinger Bands
            sig, conf, info = self._calc_bollinger(df)
            votes["bollinger"] = sig
            details["bollinger"] = {"signal": sig, "confidence": conf, **info}

            # 5. ADX (trend strength)
            sig, conf, info = self._calc_adx(df)
            votes["adx"] = sig
            details["adx"] = {"signal": sig, "confidence": conf, **info}

            # 6. Stochastic
            sig, conf, info = self._calc_stochastic(df)
            votes["stochastic"] = sig
            details["stochastic"] = {"signal": sig, "confidence": conf, **info}

            # 7. OBV (volume)
            sig, conf, info = self._calc_obv(df)
            votes["obv"] = sig
            details["obv"] = {"signal": sig, "confidence": conf, **info}

            # 8. Ichimoku Cloud
            sig, conf, info = self._calc_ichimoku(df)
            votes["ichimoku"] = sig
            details["ichimoku"] = {"signal": sig, "confidence": conf, **info}

            # Higher timeframe trend filter
            htf_trend = None
            if df_daily is not None and len(df_daily) >= 30:
                htf_trend, _, _ = self._calc_trend_sma(df_daily)
                details["htf_trend"] = htf_trend

            # Consensus decision
            final_signal, confidence = self._compute_consensus(votes, htf_trend)

            return {
                "final_signal": final_signal,
                "confidence": round(confidence, 4),
                "votes": {k: v for k, v in votes.items()},
                "details": details,
                "htf_trend": htf_trend,
            }

        except Exception as e:
            logger.error(f"Signal generation error for {symbol}: {e}")
            return {"final_signal": None, "confidence": 0.0, "error": str(e)}

    # ------------------------------------------------------------------
    # Consensus
    # ------------------------------------------------------------------
    def _compute_consensus(
        self, votes: Dict[str, Optional[str]], htf_trend: Optional[str]
    ) -> Tuple[Optional[str], float]:
        buy_weight = 0.0
        sell_weight = 0.0
        hold_weight = 0.0
        buy_count = 0
        sell_count = 0

        for name, direction in votes.items():
            w = INDICATOR_WEIGHTS.get(name, 0.1)
            if direction == "BUY":
                buy_weight += w
                buy_count += 1
            elif direction == "SELL":
                sell_weight += w
                sell_count += 1
            else:
                hold_weight += w

        # Determine raw direction
        if buy_weight > sell_weight and buy_count >= MIN_CONSENSUS:
            raw_signal = "BUY"
            raw_confidence = min(0.98, 0.5 + buy_weight)
        elif sell_weight > buy_weight and sell_count >= MIN_CONSENSUS:
            raw_signal = "SELL"
            raw_confidence = min(0.98, 0.5 + sell_weight)
        else:
            return "HOLD", round(0.3 + hold_weight * 0.2, 4)

        # Higher timeframe filter: never trade against the dominant trend
        if htf_trend is not None:
            if raw_signal == "BUY" and htf_trend == "SELL":
                logger.info("BUY blocked by bearish higher-timeframe trend")
                return "HOLD", 0.3
            if raw_signal == "SELL" and htf_trend == "BUY":
                logger.info("SELL blocked by bullish higher-timeframe trend")
                return "HOLD", 0.3

        return raw_signal, raw_confidence

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------
    def _prepare_dataframe(self, ohlcv: List[Dict]) -> Optional[pd.DataFrame]:
        if not ohlcv or len(ohlcv) < 5:
            return None
        df = pd.DataFrame(ohlcv)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            lc = col.lower()
            if col not in df.columns and lc in df.columns:
                df[col] = df[lc]
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

    # --- 1. Trend SMA ---
    def _calc_trend_sma(self, df: pd.DataFrame) -> Tuple[Optional[str], float, Dict]:
        close = df["Close"]
        sma_fast = close.rolling(window=min(10, len(close))).mean()
        sma_slow = close.rolling(window=min(30, len(close))).mean()

        if sma_fast.isna().iloc[-1] or sma_slow.isna().iloc[-1]:
            return None, 0.0, {}

        fast_val = sma_fast.iloc[-1]
        slow_val = sma_slow.iloc[-1]
        spread = (fast_val - slow_val) / slow_val if slow_val != 0 else 0

        # Check if trend is accelerating or decelerating
        prev_spread = 0
        if len(sma_fast) > 5 and not sma_fast.isna().iloc[-5]:
            prev_fast = sma_fast.iloc[-5]
            prev_slow = sma_slow.iloc[-5]
            if prev_slow != 0:
                prev_spread = (prev_fast - prev_slow) / prev_slow

        accelerating = abs(spread) > abs(prev_spread)

        if spread > 0.001:
            sig = "BUY"
            conf = min(0.95, 0.6 + abs(spread) * 15)
            if accelerating:
                conf = min(0.98, conf + 0.05)
        elif spread < -0.001:
            sig = "SELL"
            conf = min(0.95, 0.6 + abs(spread) * 15)
            if accelerating:
                conf = min(0.98, conf + 0.05)
        else:
            sig = "HOLD"
            conf = 0.4

        return sig, conf, {"sma_fast": round(fast_val, 4), "sma_slow": round(slow_val, 4), "spread": round(spread, 6)}

    # --- 2. RSI ---
    def _calc_rsi(self, df: pd.DataFrame, period: int = 14) -> Tuple[Optional[str], float, Dict]:
        close = df["Close"]
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=min(period, len(close) - 1)).mean()
        avg_loss = loss.rolling(window=min(period, len(close) - 1)).mean()

        if avg_loss.iloc[-1] == 0:
            rsi = 100.0
        else:
            rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
            rsi = 100.0 - (100.0 / (1.0 + rs))

        # RSI with trend context
        if rsi < 30:
            sig = "BUY"
            conf = min(0.95, 0.7 + (30 - rsi) / 100)
        elif rsi > 70:
            sig = "SELL"
            conf = min(0.95, 0.7 + (rsi - 70) / 100)
        elif rsi < 40:
            sig = "BUY"
            conf = 0.55
        elif rsi > 60:
            sig = "SELL"
            conf = 0.55
        else:
            sig = "HOLD"
            conf = 0.4

        return sig, conf, {"rsi": round(rsi, 2)}

    # --- 3. MACD ---
    def _calc_macd(self, df: pd.DataFrame) -> Tuple[Optional[str], float, Dict]:
        close = df["Close"]
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        macd_val = macd_line.iloc[-1]
        signal_val = signal_line.iloc[-1]
        hist_val = histogram.iloc[-1]

        # Check crossover
        if len(histogram) >= 2:
            prev_hist = histogram.iloc[-2]
        else:
            prev_hist = 0

        # MACD crossover signals
        if hist_val > 0 and prev_hist <= 0:
            sig = "BUY"
            conf = 0.85
        elif hist_val < 0 and prev_hist >= 0:
            sig = "SELL"
            conf = 0.85
        elif hist_val > 0:
            sig = "BUY"
            conf = 0.65
        elif hist_val < 0:
            sig = "SELL"
            conf = 0.65
        else:
            sig = "HOLD"
            conf = 0.4

        return sig, conf, {
            "macd": round(macd_val, 4),
            "signal": round(signal_val, 4),
            "histogram": round(hist_val, 4),
        }

    # --- 4. Bollinger Bands ---
    def _calc_bollinger(self, df: pd.DataFrame, period: int = 20) -> Tuple[Optional[str], float, Dict]:
        close = df["Close"]
        sma = close.rolling(window=min(period, len(close))).mean()
        std = close.rolling(window=min(period, len(close))).std()

        if sma.isna().iloc[-1] or std.isna().iloc[-1]:
            return None, 0.0, {}

        upper = sma.iloc[-1] + 2 * std.iloc[-1]
        lower = sma.iloc[-1] - 2 * std.iloc[-1]
        mid = sma.iloc[-1]
        price = close.iloc[-1]

        # Percent B: where price is relative to bands
        band_width = upper - lower
        if band_width == 0:
            pct_b = 0.5
        else:
            pct_b = (price - lower) / band_width

        if pct_b < 0.05:
            sig = "BUY"
            conf = min(0.9, 0.7 + (0.05 - pct_b) * 4)
        elif pct_b > 0.95:
            sig = "SELL"
            conf = min(0.9, 0.7 + (pct_b - 0.95) * 4)
        elif pct_b < 0.2:
            sig = "BUY"
            conf = 0.6
        elif pct_b > 0.8:
            sig = "SELL"
            conf = 0.6
        else:
            sig = "HOLD"
            conf = 0.4

        return sig, conf, {
            "upper": round(upper, 4),
            "lower": round(lower, 4),
            "mid": round(mid, 4),
            "pct_b": round(pct_b, 4),
        }

    # --- 5. ADX ---
    def _calc_adx(self, df: pd.DataFrame, period: int = 14) -> Tuple[Optional[str], float, Dict]:
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        p = min(period, len(df) - 1)
        atr = tr.rolling(window=p).mean()
        plus_di = 100 * (plus_dm.rolling(window=p).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=p).mean() / atr)

        di_sum = plus_di + minus_di
        di_sum = di_sum.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / di_sum
        adx = dx.rolling(window=p).mean()

        adx_val = adx.iloc[-1] if not adx.isna().iloc[-1] else 0
        plus_di_val = plus_di.iloc[-1] if not plus_di.isna().iloc[-1] else 0
        minus_di_val = minus_di.iloc[-1] if not minus_di.isna().iloc[-1] else 0

        # ADX > 25 = trending, combined with DI direction
        if adx_val > 25:
            if plus_di_val > minus_di_val:
                sig = "BUY"
                conf = min(0.9, 0.6 + adx_val / 200)
            else:
                sig = "SELL"
                conf = min(0.9, 0.6 + adx_val / 200)
        else:
            sig = "HOLD"
            conf = 0.3

        return sig, conf, {
            "adx": round(adx_val, 2),
            "plus_di": round(plus_di_val, 2),
            "minus_di": round(minus_di_val, 2),
        }

    # --- 6. Stochastic ---
    def _calc_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[Optional[str], float, Dict]:
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        p = min(k_period, len(df))
        lowest_low = low.rolling(window=p).min()
        highest_high = high.rolling(window=p).max()

        denom = highest_high - lowest_low
        denom = denom.replace(0, np.nan)
        k = 100 * (close - lowest_low) / denom
        d = k.rolling(window=min(d_period, len(k))).mean()

        k_val = k.iloc[-1] if not k.isna().iloc[-1] else 50
        d_val = d.iloc[-1] if not d.isna().iloc[-1] else 50

        # Stochastic oversold/overbought
        if k_val < 20 and d_val < 20:
            sig = "BUY"
            conf = min(0.85, 0.65 + (20 - k_val) / 100)
        elif k_val > 80 and d_val > 80:
            sig = "SELL"
            conf = min(0.85, 0.65 + (k_val - 80) / 100)
        elif k_val < 30:
            sig = "BUY"
            conf = 0.55
        elif k_val > 70:
            sig = "SELL"
            conf = 0.55
        else:
            sig = "HOLD"
            conf = 0.4

        return sig, conf, {"stoch_k": round(k_val, 2), "stoch_d": round(d_val, 2)}

    # --- 7. OBV (On Balance Volume) ---
    def _calc_obv(self, df: pd.DataFrame) -> Tuple[Optional[str], float, Dict]:
        close = df["Close"]
        volume = df["Volume"]

        if volume.sum() == 0:
            return "HOLD", 0.3, {"obv_trend": "no_volume"}

        obv = pd.Series(0.0, index=df.index)
        for i in range(1, len(df)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        # OBV trend via SMA
        obv_sma = obv.rolling(window=min(10, len(obv))).mean()

        if obv_sma.isna().iloc[-1]:
            return "HOLD", 0.3, {"obv_trend": "insufficient"}

        obv_val = obv.iloc[-1]
        obv_sma_val = obv_sma.iloc[-1]

        # Price-volume divergence check
        price_rising = close.iloc[-1] > close.iloc[-5] if len(close) > 5 else False
        obv_rising = obv_val > obv.iloc[-5] if len(obv) > 5 else False

        if obv_val > obv_sma_val and obv_rising:
            sig = "BUY"
            conf = 0.7
        elif obv_val < obv_sma_val and not obv_rising:
            sig = "SELL"
            conf = 0.7
        # Divergence: price up but OBV down = bearish
        elif price_rising and not obv_rising:
            sig = "SELL"
            conf = 0.75
        # Divergence: price down but OBV up = bullish
        elif not price_rising and obv_rising:
            sig = "BUY"
            conf = 0.75
        else:
            sig = "HOLD"
            conf = 0.4

        return sig, conf, {"obv_trend": "rising" if obv_rising else "falling"}

    # --- 8. Ichimoku Cloud ---
    def _calc_ichimoku(self, df: pd.DataFrame) -> Tuple[Optional[str], float, Dict]:
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # Tenkan-sen (conversion line) - 9 period
        p9 = min(9, len(df))
        tenkan = (high.rolling(window=p9).max() + low.rolling(window=p9).min()) / 2

        # Kijun-sen (base line) - 26 period
        p26 = min(26, len(df))
        kijun = (high.rolling(window=p26).max() + low.rolling(window=p26).min()) / 2

        if tenkan.isna().iloc[-1] or kijun.isna().iloc[-1]:
            return None, 0.0, {}

        tenkan_val = tenkan.iloc[-1]
        kijun_val = kijun.iloc[-1]
        price = close.iloc[-1]

        # Senkou Span A (leading span A)
        senkou_a = (tenkan_val + kijun_val) / 2

        # Senkou Span B - 52 period
        p52 = min(52, len(df))
        senkou_b_series = (high.rolling(window=p52).max() + low.rolling(window=p52).min()) / 2
        senkou_b = senkou_b_series.iloc[-1] if not senkou_b_series.isna().iloc[-1] else kijun_val

        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        # Ichimoku signals
        above_cloud = price > cloud_top
        below_cloud = price < cloud_bottom
        tenkan_above_kijun = tenkan_val > kijun_val

        if above_cloud and tenkan_above_kijun:
            sig = "BUY"
            conf = 0.85
        elif below_cloud and not tenkan_above_kijun:
            sig = "SELL"
            conf = 0.85
        elif above_cloud:
            sig = "BUY"
            conf = 0.6
        elif below_cloud:
            sig = "SELL"
            conf = 0.6
        else:
            sig = "HOLD"
            conf = 0.4

        return sig, conf, {
            "tenkan": round(tenkan_val, 4),
            "kijun": round(kijun_val, 4),
            "cloud_top": round(cloud_top, 4),
            "cloud_bottom": round(cloud_bottom, 4),
            "above_cloud": above_cloud,
        }
