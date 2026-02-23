"""
Invisible Data Miner Module

This module extracts hidden patterns from legitimate market data sources without
using unauthorized data scraping or illegal methods. It identifies subtle correlations
and patterns that are not immediately visible in standard market analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import math
from encryption.xmss_encryption import XMSSEncryption
import traceback

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    from types import SimpleNamespace

    def _pearsonr(x, y):
        r = float(np.corrcoef(x, y)[0, 1])
        return r, 0.0

    def _linregress(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        var_x = np.var(x, ddof=1)
        slope = float(np.cov(x, y, ddof=1)[0, 1] / var_x) if var_x != 0 else 0.0
        intercept = float(np.mean(y) - slope * np.mean(x))
        r = float(np.corrcoef(x, y)[0, 1])
        return SimpleNamespace(slope=slope, intercept=intercept, rvalue=r, pvalue=0.0, stderr=0.0)

    def _zscore(a):
        a = np.asarray(a, dtype=float)
        std = a.std()
        return (a - a.mean()) / std if std != 0 else np.zeros_like(a)

    stats = SimpleNamespace(zscore=_zscore, pearsonr=_pearsonr, linregress=_linregress)


class _FallbackAlgorithm:
    """Minimal stand-in for tests outside QuantConnect runtime."""

    def __init__(self):
        self.Time = datetime.utcnow()

    def Debug(self, _message: str):
        return None

class InvisibleDataMiner:
    """
    Extracts hidden patterns from legitimate market data sources.
    
    This module identifies subtle correlations and patterns that are not immediately
    visible in standard market analysis, using only legitimate data sources and
    compliant methods.
    """
    
    def __init__(self, algorithm=None, tree_height: int = 10):
        """
        Initialize the Invisible Data Miner module.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        - tree_height: Security parameter (2^height signatures possible)
        """
        self.algorithm = algorithm or _FallbackAlgorithm()
        self.logger = logging.getLogger("InvisibleDataMiner")
        self.logger.setLevel(logging.INFO)
        
        self.pattern_memory = {}
        self.correlation_matrix = {}
        self.hidden_cycles = {}
        self.fibonacci_levels = {}
        self.volume_profile = {}
        
        self.lookback_periods = {
            "short": 14,
            "medium": 50,
            "long": 200
        }
        
        self.cycle_detection_params = {
            "min_cycle_length": 3,
            "max_cycle_length": 89,
            "significance_threshold": 0.75
        }
        
        self.correlation_threshold = 0.65
        self.pattern_match_threshold = 0.80
        
        self.encryption_engine = XMSSEncryption(tree_height=tree_height)
        self._init_failover()
        
        self.algorithm.Debug("Invisible Data Miner module initialized")
    
    def _init_failover(self):
        """Emergency fallback configuration"""
        self.failover = b"STEALTH_FAILOVER_XMSS"
        self.max_retries = 3
        self.failover_engaged = False
        self.failover_count = 0
    
    def mine(self, symbol, history_data=None) -> bool:
        """
        Mine invisible patterns from market data.
        
        Parameters:
        - symbol: Trading symbol
        - history_data: Dictionary of DataFrames for different timeframes
        
        Returns:
        - Dictionary containing mining results
        """
        symbol_str = str(symbol)
        if history_data is None and isinstance(symbol, dict):
            return self._mine_from_scores(symbol)
        
        if "1m" not in history_data or history_data["1m"].empty:
            return {"patterns_found": False, "confidence": 0.0, "direction": "NEUTRAL"}
        
        patterns = {}
        
        for timeframe, df in history_data.items():
            if df.empty or len(df) < self.lookback_periods["medium"]:
                continue
                
            df_copy = df.copy()
            
            timeframe_patterns = self._extract_patterns(df_copy, timeframe, symbol_str)
            patterns[timeframe] = timeframe_patterns
        
        integrated_result = self._integrate_patterns(patterns, symbol_str)
        
        self._update_pattern_memory(symbol_str, integrated_result)
        
        success = True
        for tag, score in integrated_result.items():
            for attempt in range(self.max_retries):
                try:
                    if not isinstance(score, (float, int)):
                        raise TypeError(f"Invalid score type: {type(score)}")
                    if not 0 <= score <= 1:
                        raise ValueError(f"Score {score} out of bounds [0,1]")
                    blob = f"{tag}:{score:.4f}".encode('utf-8')
                    encrypted = self.encryption_engine.encrypt(blob)
                    self.encrypted_blobs[tag] = encrypted
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        self._handle_failure(tag, score, e)
                        success = False
                        self.failover_count += 1
        return success
    
    def _handle_failure(self, tag: str, score: float, error: Exception):
        self.encrypted_blobs[tag] = self.failover
        self.logger.error(
            "Mining Failure\n"
            f"Signal: {tag}\n"
            f"Score: {score}\n"
            f"Error: {str(error)}\n"
            f"Traceback:\n{traceback.format_exc()}",
            extra={'tag': tag, 'score': score}
        )
        if not self.failover_engaged:
            self.logger.critical("ACTIVATING DARK FAILOVER PROTOCOL")
            self.failover_engaged = True 
    
    
    def _mine_from_scores(self, scores):
        success = True
        for tag, score in scores.items():
            for attempt in range(self.max_retries):
                try:
                    if not isinstance(score, (float, int)):
                        raise TypeError(f"Invalid score type: {type(score)}")
                    if not 0 <= score <= 1:
                        raise ValueError(f"Score {score} out of bounds [0,1]")
                    blob = f"{tag}:{float(score):.4f}".encode("utf-8")
                    self.encrypted_blobs[tag] = self.encryption_engine.encrypt(blob)
                    break
                except Exception as exc:
                    if attempt == self.max_retries - 1:
                        self._handle_failure(tag, score, exc)
                        self.failover_count += 1
                        success = False
        return success

    def _extract_patterns(self, df, timeframe, symbol_str):
        """
        Extract hidden patterns from a single timeframe.
        
        Parameters:
        - df: DataFrame with OHLCV data
        - timeframe: String indicating the timeframe
        - symbol_str: String representation of the symbol
        
        Returns:
        - Dictionary containing extracted patterns
        """
        patterns = {}
        
        patterns["fibonacci"] = self._detect_fibonacci_relationships(df)
        
        patterns["volume"] = self._detect_volume_anomalies(df)
        
        patterns["cycles"] = self._detect_hidden_cycles(df)
        
        patterns["fractals"] = self._detect_fractal_patterns(df)
        
        patterns["divergences"] = self._detect_statistical_divergences(df)
        
        pattern_strengths = [p["strength"] for p in patterns.values()]
        overall_strength = sum(pattern_strengths) / len(pattern_strengths) if pattern_strengths else 0.0
        
        directions = [p["direction"] for p in patterns.values() if p["direction"] != "NEUTRAL"]
        if not directions:
            direction = "NEUTRAL"
        else:
            buy_count = directions.count("BUY")
            sell_count = directions.count("SELL")
            direction = "BUY" if buy_count > sell_count else "SELL" if sell_count > buy_count else "NEUTRAL"
        
        return {
            "timeframe": timeframe,
            "patterns": patterns,
            "overall_strength": overall_strength,
            "direction": direction
        }
    
    def _detect_fibonacci_relationships(self, df):
        """
        Detect hidden Fibonacci relationships in price movements.
        
        Parameters:
        - df: DataFrame with OHLCV data
        
        Returns:
        - Dictionary containing Fibonacci pattern information
        """
        if len(df) < 50:
            return {"detected": False, "strength": 0.0, "direction": "NEUTRAL"}
            
        recent_high = df["High"].iloc[-50:].max()
        recent_low = df["Low"].iloc[-50:].min()
        current_price = df["Close"].iloc[-1]
        
        range_size = recent_high - recent_low
        fib_levels = {
            0.236: recent_low + 0.236 * range_size,
            0.382: recent_low + 0.382 * range_size,
            0.5: recent_low + 0.5 * range_size,
            0.618: recent_low + 0.618 * range_size,
            0.786: recent_low + 0.786 * range_size
        }
        
        closest_level = None
        min_distance = float('inf')
        
        for level, price in fib_levels.items():
            distance = abs(current_price - price)
            if distance < min_distance:
                min_distance = distance
                closest_level = level
        
        proximity = 1.0 - (min_distance / range_size)
        
        if proximity > 0.95:
            detected = True
            strength = proximity
            
            if df["Close"].iloc[-1] > df["Close"].iloc[-2] and df["Close"].iloc[-2] > df["Close"].iloc[-3]:
                direction = "BUY"
            elif df["Close"].iloc[-1] < df["Close"].iloc[-2] and df["Close"].iloc[-2] < df["Close"].iloc[-3]:
                direction = "SELL"
            else:
                direction = "NEUTRAL"
        else:
            detected = False
            strength = proximity * 0.5  # Reduced strength for non-significant levels
            direction = "NEUTRAL"
        
        return {
            "detected": detected,
            "closest_level": closest_level,
            "proximity": proximity,
            "strength": strength,
            "direction": direction,
            "levels": fib_levels
        }
    
    def _detect_volume_anomalies(self, df):
        """
        Detect hidden volume anomalies and patterns.
        
        Parameters:
        - df: DataFrame with OHLCV data
        
        Returns:
        - Dictionary containing volume pattern information
        """
        if "Volume" not in df.columns or len(df) < 20:
            return {"detected": False, "strength": 0.0, "direction": "NEUTRAL"}
        
        df["volume_ma"] = df["Volume"].rolling(window=20).mean()
        df["volume_std"] = df["Volume"].rolling(window=20).std()
        df["volume_z"] = (df["Volume"] - df["volume_ma"]) / df["volume_std"]
        
        recent_volume_z = df["volume_z"].iloc[-5:].values
        has_anomaly = any(abs(z) > 2.0 for z in recent_volume_z)
        
        if not has_anomaly:
            return {"detected": False, "strength": 0.0, "direction": "NEUTRAL"}
        
        df["price_change"] = df["Close"].pct_change()
        
        df["abs_price_change"] = df["price_change"].abs()
        recent_corr = df["Volume"].iloc[-20:].corr(df["abs_price_change"].iloc[-20:])
        
        recent_volume = df["Volume"].iloc[-5:].values
        recent_prices = df["Close"].iloc[-5:].values
        
        volume_increasing = recent_volume[-1] > recent_volume[0]
        price_direction = "BUY" if recent_prices[-1] > recent_prices[0] else "SELL"
        
        if max(recent_volume_z) > 2.5:
            climax_index = list(recent_volume_z).index(max(recent_volume_z))
            
            if climax_index < len(recent_prices) - 1:
                pre_climax = recent_prices[climax_index - 1] if climax_index > 0 else recent_prices[0]
                climax_price = recent_prices[climax_index]
                post_climax = recent_prices[climax_index + 1]
                
                if (climax_price > pre_climax and post_climax < climax_price) or \
                   (climax_price < pre_climax and post_climax > climax_price):
                    direction = "SELL" if climax_price > pre_climax else "BUY"
                    strength = min(1.0, abs(max(recent_volume_z)) / 3.0)
                    
                    return {
                        "detected": True,
                        "type": "volume_climax_reversal",
                        "strength": strength,
                        "direction": direction,
                        "volume_z": max(recent_volume_z)
                    }
        
        if volume_increasing and abs(recent_corr) > 0.7:
            strength = min(1.0, abs(recent_corr))
            
            return {
                "detected": True,
                "type": "volume_confirmation",
                "strength": strength,
                "direction": price_direction,
                "correlation": recent_corr
            }
        
        if not volume_increasing and abs(recent_corr) < 0.3:
            strength = 0.5
            reverse_direction = "SELL" if price_direction == "BUY" else "BUY"
            
            return {
                "detected": True,
                "type": "volume_divergence",
                "strength": strength,
                "direction": reverse_direction,
                "correlation": recent_corr
            }
        
        return {
            "detected": has_anomaly,
            "type": "general_anomaly",
            "strength": 0.3,
            "direction": "NEUTRAL",
            "volume_z": max(abs(z) for z in recent_volume_z)
        }
    
    def _detect_hidden_cycles(self, df):
        """
        Detect hidden cyclical patterns in price data.
        
        Parameters:
        - df: DataFrame with OHLCV data
        
        Returns:
        - Dictionary containing cycle pattern information
        """
        if len(df) < 50:
            return {"detected": False, "strength": 0.0, "direction": "NEUTRAL"}
        
        prices = df["Close"].values
        
        min_length = self.cycle_detection_params["min_cycle_length"]
        max_length = min(self.cycle_detection_params["max_cycle_length"], len(prices) // 3)
        
        best_cycle = None
        best_strength = 0
        
        for cycle_length in range(min_length, max_length + 1):
            if len(prices) <= cycle_length:
                continue
                
            autocorr = np.correlate(prices, prices, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            if len(autocorr) <= cycle_length:
                continue
                
            cycle_strength = autocorr[cycle_length] / autocorr[0]
            
            if cycle_strength > best_strength:
                best_strength = cycle_strength
                best_cycle = cycle_length
        
        if best_cycle is None or best_strength < self.cycle_detection_params["significance_threshold"]:
            return {"detected": False, "strength": 0.0, "direction": "NEUTRAL"}
        
        cycle_position = len(prices) % best_cycle
        
        historical_positions = []
        
        for i in range(len(prices) - best_cycle, 0, -best_cycle):
            if i + cycle_position < len(prices):
                historical_positions.append(prices[i + cycle_position] - prices[i])
        
        if not historical_positions:
            return {"detected": True, "cycle_length": best_cycle, "strength": best_strength, "direction": "NEUTRAL"}
        
        avg_move = sum(historical_positions) / len(historical_positions)
        direction = "BUY" if avg_move > 0 else "SELL" if avg_move < 0 else "NEUTRAL"
        
        return {
            "detected": True,
            "cycle_length": best_cycle,
            "cycle_position": cycle_position,
            "strength": best_strength,
            "direction": direction,
            "avg_historical_move": avg_move
        }
    
    def _detect_fractal_patterns(self, df):
        """
        Detect fractal patterns in price data.
        
        Parameters:
        - df: DataFrame with OHLCV data
        
        Returns:
        - Dictionary containing fractal pattern information
        """
        if len(df) < 30:
            return {"detected": False, "strength": 0.0, "direction": "NEUTRAL"}
        
        potential_fractals = []
        
        for i in range(2, len(df) - 2):
            if (df["Low"].iloc[i-2] > df["Low"].iloc[i] and
                df["Low"].iloc[i-1] > df["Low"].iloc[i] and
                df["Low"].iloc[i] < df["Low"].iloc[i+1] and
                df["Low"].iloc[i] < df["Low"].iloc[i+2]):
                
                potential_fractals.append({
                    "type": "bullish",
                    "index": i,
                    "price": df["Low"].iloc[i],
                    "time": df.index[i]
                })
        
        for i in range(2, len(df) - 2):
            if (df["High"].iloc[i-2] < df["High"].iloc[i] and
                df["High"].iloc[i-1] < df["High"].iloc[i] and
                df["High"].iloc[i] > df["High"].iloc[i+1] and
                df["High"].iloc[i] > df["High"].iloc[i+2]):
                
                potential_fractals.append({
                    "type": "bearish",
                    "index": i,
                    "price": df["High"].iloc[i],
                    "time": df.index[i]
                })
        
        if not potential_fractals:
            return {"detected": False, "strength": 0.0, "direction": "NEUTRAL"}
        
        recent_fractals = [f for f in potential_fractals if f["index"] > len(df) - 10]
        
        if not recent_fractals:
            return {"detected": False, "strength": 0.0, "direction": "NEUTRAL"}
        
        most_recent = max(recent_fractals, key=lambda x: x["index"])
        
        current_price = df["Close"].iloc[-1]
        
        if most_recent["type"] == "bullish":
            post_fractal_high = df["High"].iloc[most_recent["index"]+1:].max()
            confirmed = current_price > post_fractal_high
            direction = "BUY" if confirmed else "NEUTRAL"
        else:
            post_fractal_low = df["Low"].iloc[most_recent["index"]+1:].min()
            confirmed = current_price < post_fractal_low
            direction = "SELL" if confirmed else "NEUTRAL"
        
        if most_recent["type"] == "bullish":
            pre_fractal_low = df["Low"].iloc[most_recent["index"]-2:most_recent["index"]].min()
            post_fractal_low = df["Low"].iloc[most_recent["index"]+1:most_recent["index"]+3].min()
            strength = min(1.0, (pre_fractal_low - most_recent["price"] + post_fractal_low - most_recent["price"]) / (2 * most_recent["price"]) * 100)
        else:
            pre_fractal_high = df["High"].iloc[most_recent["index"]-2:most_recent["index"]].max()
            post_fractal_high = df["High"].iloc[most_recent["index"]+1:most_recent["index"]+3].max()
            strength = min(1.0, (most_recent["price"] - pre_fractal_high + most_recent["price"] - post_fractal_high) / (2 * most_recent["price"]) * 100)
        
        return {
            "detected": True,
            "type": most_recent["type"],
            "time": most_recent["time"],
            "price": most_recent["price"],
            "confirmed": confirmed,
            "strength": strength,
            "direction": direction,
            "total_fractals": len(potential_fractals),
            "recent_fractals": len(recent_fractals)
        }
    
    def _detect_statistical_divergences(self, df):
        """
        Detect statistical divergences between price and indicators.
        
        Parameters:
        - df: DataFrame with OHLCV data
        
        Returns:
        - Dictionary containing divergence information
        """
        if len(df) < 30:
            return {"detected": False, "strength": 0.0, "direction": "NEUTRAL"}
        
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = df["EMA12"] - df["EMA26"]
        df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        
        window = min(30, len(df) - 1)
        
        price_data = df["Close"].iloc[-window:].values
        price_highs = []
        price_lows = []
        
        for i in range(1, window - 1):
            if price_data[i] > price_data[i-1] and price_data[i] > price_data[i+1]:
                price_highs.append(i)
            if price_data[i] < price_data[i-1] and price_data[i] < price_data[i+1]:
                price_lows.append(i)
        
        rsi_data = df["RSI"].iloc[-window:].values
        rsi_highs = []
        rsi_lows = []
        
        for i in range(1, window - 1):
            if rsi_data[i] > rsi_data[i-1] and rsi_data[i] > rsi_data[i+1]:
                rsi_highs.append(i)
            if rsi_data[i] < rsi_data[i-1] and rsi_data[i] < rsi_data[i+1]:
                rsi_lows.append(i)
        
        bearish_div = False
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            if price_data[price_highs[-1]] > price_data[price_highs[-2]] and rsi_data[rsi_highs[-1]] < rsi_data[rsi_highs[-2]]:
                bearish_div = True
        
        bullish_div = False
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            if price_data[price_lows[-1]] < price_data[price_lows[-2]] and rsi_data[rsi_lows[-1]] > rsi_data[rsi_lows[-2]]:
                bullish_div = True
        
        if not bearish_div and not bullish_div:
            return {"detected": False, "strength": 0.0, "direction": "NEUTRAL"}
        
        if bearish_div:
            price_change = (price_data[price_highs[-1]] - price_data[price_highs[-2]]) / price_data[price_highs[-2]]
            rsi_change = (rsi_data[rsi_highs[-1]] - rsi_data[rsi_highs[-2]]) / rsi_data[rsi_highs[-2]]
            strength = min(1.0, abs(price_change - rsi_change))
            direction = "SELL"
        else:
            price_change = (price_data[price_lows[-1]] - price_data[price_lows[-2]]) / price_data[price_lows[-2]]
            rsi_change = (rsi_data[rsi_lows[-1]] - rsi_data[rsi_lows[-2]]) / rsi_data[rsi_lows[-2]]
            strength = min(1.0, abs(price_change - rsi_change))
            direction = "BUY"
        
        return {
            "detected": True,
            "type": "bearish" if bearish_div else "bullish",
            "indicator": "RSI",
            "strength": strength,
            "direction": direction
        }
    
    def _integrate_patterns(self, patterns, symbol_str):
        """
        Integrate patterns across timeframes.
        
        Parameters:
        - patterns: Dictionary of patterns by timeframe
        - symbol_str: String representation of the symbol
        
        Returns:
        - Dictionary containing integrated pattern results
        """
        if not patterns:
            return {"patterns_found": False, "confidence": 0.0, "direction": "NEUTRAL"}
        
        timeframe_weights = {
            "1m": 0.1,
            "5m": 0.15,
            "10m": 0.15,
            "15m": 0.2,
            "20m": 0.2,
            "25m": 0.2
        }
        
        weighted_strengths = []
        directions = []
        
        for timeframe, result in patterns.items():
            weight = timeframe_weights.get(timeframe, 0.1)
            weighted_strengths.append(result["overall_strength"] * weight)
            
            if result["direction"] != "NEUTRAL":
                directions.append(result["direction"])
        
        confidence = sum(weighted_strengths) / sum([sum(timeframe_weights.values()) for timeframe in patterns.keys()])
        
        if not directions:
            direction = "NEUTRAL"
        else:
            buy_count = directions.count("BUY")
            sell_count = directions.count("SELL")
            direction = "BUY" if buy_count > sell_count else "SELL" if sell_count > buy_count else "NEUTRAL"
        
        aligned_timeframes = []
        for timeframe, result in patterns.items():
            if result["direction"] == direction and result["direction"] != "NEUTRAL":
                aligned_timeframes.append(timeframe)
        
        alignment_factor = len(aligned_timeframes) / len(patterns) if patterns else 0
        
        adjusted_confidence = confidence * (0.5 + 0.5 * alignment_factor)
        
        return {
            "patterns_found": adjusted_confidence > 0.3,
            "confidence": adjusted_confidence,
            "direction": direction,
            "aligned_timeframes": aligned_timeframes,
            "alignment_factor": alignment_factor,
            "pattern_details": patterns
        }
    
    def _update_pattern_memory(self, symbol_str, result):
        """
        Update pattern memory for future reference.
        
        Parameters:
        - symbol_str: String representation of the symbol
        - result: Dictionary containing pattern results
        """
        if symbol_str not in self.pattern_memory:
            self.pattern_memory[symbol_str] = []
        
        if result["patterns_found"] and result["confidence"] > 0.4:
            result["timestamp"] = self.algorithm.Time
            
            if len(self.pattern_memory[symbol_str]) > 100:
                self.pattern_memory[symbol_str].pop(0)
            
            self.pattern_memory[symbol_str].append(result)
    
    def get_pattern_memory(self, symbol=None):
        """
        Get stored pattern memory.
        
        Parameters:
        - symbol: Optional symbol to filter memory
        
        Returns:
        - Dictionary containing pattern memory
        """
        if symbol:
            symbol_str = str(symbol)
            return self.pattern_memory.get(symbol_str, [])
        
        return self.pattern_memory
