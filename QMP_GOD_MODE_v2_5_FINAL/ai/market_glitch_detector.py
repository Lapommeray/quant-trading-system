"""
Market Glitch Detector
Advanced AI module for detecting market glitches, errors, and anomalies
that are imperceptible to humans but can be exploited for profit.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import os
import math
from scipy import stats
from collections import defaultdict

class MarketGlitchDetector:
    """
    Detects market glitches, errors, and anomalies that are imperceptible to humans
    but can be exploited for profit using advanced AI techniques.
    """
    
    def __init__(self):
        """Initialize the Market Glitch Detector with super high confidence requirements"""
        self.logger = self._setup_logger()
        self.logger.info("Market Glitch Detector initialized with super high confidence requirements")
        
        self.confidence_threshold = 0.95  # Super high confidence threshold
        self.min_data_quality = 0.95      # Minimum data quality score
        self.detection_history = []
        self.detection_stats = {
            "total_checks": 0,
            "glitches_detected": 0,
            "false_positives": 0,
            "detection_rate": 0.0,
            "average_confidence": 0.0,
            "exploitation_success_rate": 0.0
        }
        
        self.glitch_types = {
            "price_discontinuity": {"weight": 0.9, "min_confidence": 0.92},
            "order_book_imbalance": {"weight": 0.85, "min_confidence": 0.90},
            "temporal_arbitrage": {"weight": 0.95, "min_confidence": 0.94},
            "liquidity_vacuum": {"weight": 0.88, "min_confidence": 0.91},
            "flash_pattern": {"weight": 0.92, "min_confidence": 0.93},
            "quantum_probability_collapse": {"weight": 0.97, "min_confidence": 0.95},
            "market_maker_error": {"weight": 0.94, "min_confidence": 0.92},
            "fibonacci_violation": {"weight": 0.86, "min_confidence": 0.90},
            "golden_ratio_anomaly": {"weight": 0.89, "min_confidence": 0.91},
            "mathematical_constant_deviation": {"weight": 0.93, "min_confidence": 0.94}
        }
    
    def _setup_logger(self):
        """Set up logger for the Market Glitch Detector"""
        logger = logging.getLogger("MarketGlitchDetector")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            try:
                os.makedirs("logs", exist_ok=True)
                file_handler = logging.FileHandler("logs/market_glitch_detector.log")
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Could not create log file: {e}")
        
        return logger
    
    def detect_glitches(self, market_data):
        """
        Detect market glitches, errors, and anomalies in the provided market data.
        
        Parameters:
        - market_data: Dictionary containing market data (OHLCV, order book, etc.)
        
        Returns:
        - Dictionary with detection results
        """
        self.detection_stats["total_checks"] += 1
        
        quality_result = self._verify_data_quality(market_data)
        if not quality_result.get("quality_verified", False):
            return {
                "glitches_detected": False,
                "reason": quality_result.get("reason", "insufficient_data_quality"),
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "high_confidence": False
            }
        
        detected_glitches = []
        
        if "ohlcv" in market_data and len(market_data["ohlcv"]) > 10:
            price_disc = self._detect_price_discontinuity(market_data)
            if price_disc["detected"]:
                detected_glitches.append(price_disc)
        
        if "order_book" in market_data:
            order_imbalance = self._detect_order_book_imbalance(market_data)
            if order_imbalance["detected"]:
                detected_glitches.append(order_imbalance)
        
        if "ohlcv" in market_data and "timestamp" in market_data:
            temporal_arb = self._detect_temporal_arbitrage(market_data)
            if temporal_arb["detected"]:
                detected_glitches.append(temporal_arb)
        
        if "volume" in market_data or ("ohlcv" in market_data and len(market_data["ohlcv"]) > 0):
            liquidity_vac = self._detect_liquidity_vacuum(market_data)
            if liquidity_vac["detected"]:
                detected_glitches.append(liquidity_vac)
        
        if "ohlcv" in market_data and len(market_data["ohlcv"]) > 20:
            flash_pattern = self._detect_flash_pattern(market_data)
            if flash_pattern["detected"]:
                detected_glitches.append(flash_pattern)
        
        math_deviation = self._detect_mathematical_constant_deviation(market_data)
        if math_deviation["detected"]:
            detected_glitches.append(math_deviation)
        
        if not detected_glitches:
            return {
                "glitches_detected": False,
                "reason": "no_glitches_found",
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "high_confidence": False
            }
        
        overall_confidence = self._calculate_overall_confidence(detected_glitches)
        signal_info = self._generate_signal_from_glitches(detected_glitches)
        
        self.detection_stats["glitches_detected"] += 1
        self.detection_stats["average_confidence"] = (
            (self.detection_stats["average_confidence"] * (self.detection_stats["glitches_detected"] - 1) + 
             overall_confidence) / self.detection_stats["glitches_detected"]
        )
        self.detection_stats["detection_rate"] = (
            self.detection_stats["glitches_detected"] / self.detection_stats["total_checks"]
        )
        
        self.detection_history.append({
            "timestamp": datetime.now().isoformat(),
            "glitches": detected_glitches,
            "confidence": overall_confidence,
            "signal": signal_info["signal"]
        })
        
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-1000:]
        
        return {
            "glitches_detected": True,
            "glitches": detected_glitches,
            "confidence": overall_confidence,
            "signal": signal_info["signal"],
            "high_confidence": overall_confidence >= self.confidence_threshold,
            "exploitation_window": signal_info.get("exploitation_window", None),
            "expected_profit": signal_info.get("expected_profit", None)
        }
    
    def _verify_data_quality(self, market_data):
        """
        Verify the quality of market data to ensure it meets super high standards.
        
        Parameters:
        - market_data: Dictionary containing market data
        
        Returns:
        - Dictionary with verification results
        """
        required_keys = ["ohlcv"]
        if not all(key in market_data for key in required_keys):
            return {
                "quality_verified": False,
                "reason": "missing_required_data",
                "quality_score": 0.0
            }
        
        if "ohlcv" in market_data and len(market_data["ohlcv"]) < 10:
            return {
                "quality_verified": False,
                "reason": "insufficient_data_points",
                "quality_score": 0.0
            }
        
        quality_score = self._calculate_data_quality_score(market_data)
        
        if quality_score < self.min_data_quality:
            return {
                "quality_verified": False,
                "reason": "low_quality_data",
                "quality_score": quality_score
            }
        
        return {
            "quality_verified": True,
            "quality_score": quality_score
        }
    
    def _calculate_data_quality_score(self, market_data):
        """
        Calculate a quality score for the market data.
        
        Parameters:
        - market_data: Dictionary containing market data
        
        Returns:
        - Quality score between 0 and 1
        """
        quality_factors = []
        
        if "ohlcv" in market_data and len(market_data["ohlcv"]) > 0:
            ohlcv_data = market_data["ohlcv"]
            missing_values = sum(1 for candle in ohlcv_data if None in candle or np.nan in candle)
            missing_ratio = missing_values / len(ohlcv_data)
            quality_factors.append(1 - missing_ratio)
        
        if "ohlcv" in market_data and len(market_data["ohlcv"]) > 1:
            ohlcv_data = market_data["ohlcv"]
            close_prices = [candle[4] for candle in ohlcv_data]
            price_jumps = [abs(close_prices[i] - close_prices[i-1]) / close_prices[i-1] 
                          for i in range(1, len(close_prices))]
            extreme_jumps = sum(1 for jump in price_jumps if jump > 0.1)  # 10% price jump
            extreme_ratio = extreme_jumps / len(price_jumps)
            quality_factors.append(1 - extreme_ratio)
        
        if "ohlcv" in market_data and len(market_data["ohlcv"]) > 1:
            ohlcv_data = market_data["ohlcv"]
            timestamps = [candle[0] for candle in ohlcv_data]
            time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            if len(set(time_diffs)) > 1:  # More than one unique time difference
                std_dev = np.std(time_diffs)
                mean_diff = np.mean(time_diffs)
                consistency = 1 - min(float(std_dev / mean_diff), 1.0)  # Normalize to [0,1]
                quality_factors.append(consistency)
        
        if "ohlcv" in market_data and len(market_data["ohlcv"]) > 1:
            ohlcv_data = market_data["ohlcv"]
            volumes = [candle[5] for candle in ohlcv_data]
            zero_volumes = sum(1 for vol in volumes if vol == 0)
            zero_ratio = zero_volumes / len(volumes)
            quality_factors.append(1 - zero_ratio)
        
        if not quality_factors:
            return 0.0
        
        return sum(quality_factors) / len(quality_factors)
    
    def _detect_price_discontinuity(self, market_data):
        """
        Detect price discontinuities (gaps) that indicate potential market glitches.
        
        Parameters:
        - market_data: Dictionary containing market data
        
        Returns:
        - Dictionary with detection results
        """
        ohlcv_data = market_data["ohlcv"]
        close_prices = [candle[4] for candle in ohlcv_data]
        
        price_changes = [abs(close_prices[i] - close_prices[i-1]) / close_prices[i-1] 
                        for i in range(1, len(close_prices))]
        
        mean_change = np.mean(price_changes)
        std_change = np.std(price_changes)
        
        threshold = mean_change + 3 * std_change
        discontinuities = [(i+1, price_changes[i]) for i in range(len(price_changes)) 
                          if price_changes[i] > threshold]
        
        if not discontinuities:
            return {"detected": False, "type": "price_discontinuity"}
        
        max_disc_idx, max_disc_val = max(discontinuities, key=lambda x: x[1])
        z_score = (max_disc_val - mean_change) / std_change
        confidence = min(0.5 + 0.5 * (z_score / 10), 0.99)  # Scale to [0.5, 0.99]
        
        exploitable = confidence >= self.glitch_types["price_discontinuity"]["min_confidence"]
        
        if max_disc_idx < len(close_prices) - 1:
            reversion = (close_prices[max_disc_idx+1] - close_prices[max_disc_idx]) / close_prices[max_disc_idx]
            signal = "SELL" if reversion < 0 else "BUY"
        else:
            signal = "BUY" if close_prices[max_disc_idx] > close_prices[max_disc_idx-1] else "SELL"
        
        return {
            "detected": True,
            "type": "price_discontinuity",
            "confidence": confidence,
            "exploitable": exploitable,
            "signal": signal,
            "location": max_disc_idx,
            "magnitude": max_disc_val,
            "z_score": z_score
        }
    
    def _detect_order_book_imbalance(self, market_data):
        """
        Detect order book imbalances that indicate potential market glitches.
        
        Parameters:
        - market_data: Dictionary containing market data
        
        Returns:
        - Dictionary with detection results
        """
        if "order_book" not in market_data:
            return {"detected": False, "type": "order_book_imbalance"}
        
        order_book = market_data["order_book"]
        
        if "bids" not in order_book or "asks" not in order_book:
            return {"detected": False, "type": "order_book_imbalance"}
        
        bids = order_book["bids"]
        asks = order_book["asks"]
        
        bid_volume = sum(bid[1] for bid in bids)
        ask_volume = sum(ask[1] for ask in asks)
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return {"detected": False, "type": "order_book_imbalance"}
        
        imbalance_ratio = abs(bid_volume - ask_volume) / total_volume
        
        if imbalance_ratio < 0.7:  # Less than 70% imbalance
            return {"detected": False, "type": "order_book_imbalance"}
        
        confidence = min(0.5 + 0.5 * imbalance_ratio, 0.99)  # Scale to [0.5, 0.99]
        
        exploitable = confidence >= self.glitch_types["order_book_imbalance"]["min_confidence"]
        
        signal = "BUY" if bid_volume > ask_volume else "SELL"
        
        return {
            "detected": True,
            "type": "order_book_imbalance",
            "confidence": confidence,
            "exploitable": exploitable,
            "signal": signal,
            "imbalance_ratio": imbalance_ratio,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume
        }
    
    def _detect_temporal_arbitrage(self, market_data):
        """
        Detect temporal arbitrage opportunities that indicate potential market glitches.
        
        Parameters:
        - market_data: Dictionary containing market data
        
        Returns:
        - Dictionary with detection results
        """
        if "ohlcv" not in market_data or "timestamp" not in market_data:
            return {"detected": False, "type": "temporal_arbitrage"}
        
        ohlcv_data = market_data["ohlcv"]
        current_timestamp = market_data["timestamp"]
        
        if len(ohlcv_data) < 3:
            return {"detected": False, "type": "temporal_arbitrage"}
        
        timestamps = [candle[0] for candle in ohlcv_data]
        close_prices = [candle[4] for candle in ohlcv_data]
        
        time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        mean_diff = np.mean(time_diffs)
        std_diff = np.std(time_diffs)
        
        anomalies = [(i+1, time_diffs[i]) for i in range(len(time_diffs)) 
                    if abs(time_diffs[i] - mean_diff) > 3 * std_diff]
        
        if not anomalies:
            return {"detected": False, "type": "temporal_arbitrage"}
        
        max_anom_idx, max_anom_val = max(anomalies, key=lambda x: abs(x[1] - mean_diff))
        
        if max_anom_idx < len(close_prices) - 1:
            price_change = (close_prices[max_anom_idx+1] - close_prices[max_anom_idx]) / close_prices[max_anom_idx]
            
            if abs(price_change) < 0.001:  # Less than 0.1% change
                return {"detected": False, "type": "temporal_arbitrage"}
            
            time_z_score = abs(max_anom_val - mean_diff) / std_diff
            confidence = min(0.5 + 0.25 * time_z_score + 0.25 * abs(price_change) * 100, 0.99)
            
            exploitable = confidence >= self.glitch_types["temporal_arbitrage"]["min_confidence"]
            
            signal = "BUY" if price_change > 0 else "SELL"
            
            exploitation_window = min(abs(max_anom_val - mean_diff) / 2, 60)  # In seconds, max 60 seconds
            
            return {
                "detected": True,
                "type": "temporal_arbitrage",
                "confidence": confidence,
                "exploitable": exploitable,
                "signal": signal,
                "location": max_anom_idx,
                "time_anomaly": max_anom_val - mean_diff,
                "price_change": price_change,
                "exploitation_window": exploitation_window
            }
        
        return {"detected": False, "type": "temporal_arbitrage"}
    
    def _detect_liquidity_vacuum(self, market_data):
        """
        Detect liquidity vacuums that indicate potential market glitches.
        
        Parameters:
        - market_data: Dictionary containing market data
        
        Returns:
        - Dictionary with detection results
        """
        if "volume" in market_data:
            volumes = market_data["volume"]
        elif "ohlcv" in market_data and len(market_data["ohlcv"]) > 0:
            volumes = [candle[5] for candle in market_data["ohlcv"]]
        else:
            return {"detected": False, "type": "liquidity_vacuum"}
        
        if len(volumes) < 5:
            return {"detected": False, "type": "liquidity_vacuum"}
        
        window_size = min(10, len(volumes) - 1)
        volume_ma = [np.mean(volumes[i-window_size:i]) for i in range(window_size, len(volumes))]
        
        volume_drops = []
        for i in range(1, len(volume_ma)):
            if volume_ma[i] < volume_ma[i-1] * 0.3:  # 70% or more drop in volume
                volume_drops.append((i + window_size, volume_ma[i] / volume_ma[i-1]))
        
        if not volume_drops:
            return {"detected": False, "type": "liquidity_vacuum"}
        
        max_drop_idx, max_drop_ratio = min(volume_drops, key=lambda x: x[1])
        
        confidence = min(0.5 + 0.5 * (1 - max_drop_ratio), 0.99)  # Scale to [0.5, 0.99]
        
        exploitable = confidence >= self.glitch_types["liquidity_vacuum"]["min_confidence"]
        
        if max_drop_idx < len(volumes) - 1:
            if "ohlcv" in market_data:
                ohlcv_data = market_data["ohlcv"]
                if max_drop_idx < len(ohlcv_data) - 1:
                    price_before = ohlcv_data[max_drop_idx-1][4]
                    price_after = ohlcv_data[max_drop_idx+1][4]
                    signal = "BUY" if price_after > price_before else "SELL"
                else:
                    trend = self._calculate_trend(market_data)
                    signal = "BUY" if trend > 0 else "SELL"
            else:
                trend = self._calculate_trend(market_data)
                signal = "BUY" if trend > 0 else "SELL"
        else:
            trend = self._calculate_trend(market_data)
            signal = "BUY" if trend > 0 else "SELL"
        
        return {
            "detected": True,
            "type": "liquidity_vacuum",
            "confidence": confidence,
            "exploitable": exploitable,
            "signal": signal,
            "location": max_drop_idx,
            "volume_drop_ratio": max_drop_ratio
        }
    
    def _detect_flash_pattern(self, market_data):
        """
        Detect flash patterns that indicate potential market glitches.
        
        Parameters:
        - market_data: Dictionary containing market data
        
        Returns:
        - Dictionary with detection results
        """
        if "ohlcv" not in market_data or len(market_data["ohlcv"]) < 20:
            return {"detected": False, "type": "flash_pattern"}
        
        ohlcv_data = market_data["ohlcv"]
        close_prices = [candle[4] for candle in ohlcv_data]
        
        returns = [close_prices[i] / close_prices[i-1] - 1 for i in range(1, len(close_prices))]
        
        flash_patterns = []
        for i in range(len(returns) - 2):
            if abs(returns[i]) > 0.005 and returns[i] * returns[i+1] < 0:  # 0.5% move followed by reversal
                magnitude = abs(returns[i])
                reversal_ratio = abs(returns[i+1] / returns[i])
                
                if reversal_ratio > 0.5:  # At least 50% reversal
                    flash_patterns.append((i+1, magnitude, reversal_ratio))
        
        if not flash_patterns:
            return {"detected": False, "type": "flash_pattern"}
        
        max_pattern = max(flash_patterns, key=lambda x: x[1] * x[2])
        max_idx, magnitude, reversal_ratio = max_pattern
        
        confidence = min(0.5 + 0.25 * magnitude * 100 + 0.25 * reversal_ratio, 0.99)  # Scale to [0.5, 0.99]
        
        exploitable = confidence >= self.glitch_types["flash_pattern"]["min_confidence"]
        
        signal = "BUY" if returns[max_idx] > 0 else "SELL"
        
        return {
            "detected": True,
            "type": "flash_pattern",
            "confidence": confidence,
            "exploitable": exploitable,
            "signal": signal,
            "location": max_idx,
            "magnitude": magnitude,
            "reversal_ratio": reversal_ratio
        }
    
    def _detect_mathematical_constant_deviation(self, market_data):
        """
        Detect deviations from mathematical constants that indicate potential market glitches.
        
        Parameters:
        - market_data: Dictionary containing market data
        
        Returns:
        - Dictionary with detection results
        """
        constants = {
            "pi": math.pi,
            "e": math.e,
            "phi": (1 + math.sqrt(5)) / 2,  # Golden ratio
            "sqrt2": math.sqrt(2),
            "sqrt3": math.sqrt(3),
            "ln2": math.log(2)
        }
        
        if "ohlcv" in market_data and len(market_data["ohlcv"]) > 0:
            ohlcv_data = market_data["ohlcv"]
            close_prices = [candle[4] for candle in ohlcv_data]
        else:
            return {"detected": False, "type": "mathematical_constant_deviation"}
        
        if len(close_prices) < 5:
            return {"detected": False, "type": "mathematical_constant_deviation"}
        
        price_ratios = [close_prices[i] / close_prices[i-1] for i in range(1, len(close_prices))]
        
        deviations = []
        for i, ratio in enumerate(price_ratios):
            for const_name, const_value in constants.items():
                deviation = abs(ratio - const_value) / const_value
                
                if deviation < 0.001:
                    deviations.append((i+1, const_name, deviation))
        
        if not deviations:
            return {"detected": False, "type": "mathematical_constant_deviation"}
        
        min_dev_idx, min_dev_const, min_dev_val = min(deviations, key=lambda x: x[2])
        
        confidence = min(0.9 + 0.1 * (1 - min_dev_val * 1000), 0.99)  # Scale to [0.9, 0.99]
        
        exploitable = confidence >= self.glitch_types["mathematical_constant_deviation"]["min_confidence"]
        
        if min_dev_idx < len(close_prices) - 1:
            subsequent_return = close_prices[min_dev_idx+1] / close_prices[min_dev_idx] - 1
            signal = "BUY" if subsequent_return > 0 else "SELL"
        else:
            trend = self._calculate_trend(market_data)
            signal = "BUY" if trend > 0 else "SELL"
        
        return {
            "detected": True,
            "type": "mathematical_constant_deviation",
            "confidence": confidence,
            "exploitable": exploitable,
            "signal": signal,
            "location": min_dev_idx,
            "constant": min_dev_const,
            "deviation": min_dev_val
        }
    
    def _calculate_trend(self, market_data):
        """
        Calculate the current market trend.
        
        Parameters:
        - market_data: Dictionary containing market data
        
        Returns:
        - Trend value (positive for uptrend, negative for downtrend)
        """
        if "ohlcv" not in market_data or len(market_data["ohlcv"]) < 5:
            return 0
        
        ohlcv_data = market_data["ohlcv"]
        close_prices = [candle[4] for candle in ohlcv_data]
        
        x = np.arange(len(close_prices))
        slope, _, _, _, _ = stats.linregress(x, close_prices)
        
        return slope
    
    def _calculate_overall_confidence(self, detected_glitches):
        """
        Calculate overall confidence based on detected glitches.
        
        Parameters:
        - detected_glitches: List of detected glitches
        
        Returns:
        - Overall confidence score
        """
        if not detected_glitches:
            return 0.0
        
        total_weight = 0
        weighted_confidence = 0
        
        for glitch in detected_glitches:
            glitch_type = glitch["type"]
            if glitch_type in self.glitch_types:
                weight = self.glitch_types[glitch_type]["weight"]
                confidence = glitch["confidence"]
                
                weighted_confidence += weight * confidence
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_confidence / total_weight
    
    def _generate_signal_from_glitches(self, detected_glitches):
        """
        Generate trading signal based on detected glitches.
        
        Parameters:
        - detected_glitches: List of detected glitches
        
        Returns:
        - Dictionary with signal information
        """
        if not detected_glitches:
            return {"signal": "NEUTRAL"}
        
        buy_signals = 0
        sell_signals = 0
        buy_confidence = 0
        sell_confidence = 0
        
        for glitch in detected_glitches:
            if not glitch.get("exploitable", False):
                continue
                
            confidence = glitch["confidence"]
            if glitch["signal"] == "BUY":
                buy_signals += 1
                buy_confidence += confidence
            elif glitch["signal"] == "SELL":
                sell_signals += 1
                sell_confidence += confidence
        
        if buy_signals > sell_signals:
            signal = "BUY"
            signal_confidence = buy_confidence / buy_signals if buy_signals > 0 else 0
        elif sell_signals > buy_signals:
            signal = "SELL"
            signal_confidence = sell_confidence / sell_signals if sell_signals > 0 else 0
        else:
            if buy_confidence > sell_confidence:
                signal = "BUY"
                signal_confidence = buy_confidence / buy_signals if buy_signals > 0 else 0
            elif sell_confidence > buy_confidence:
                signal = "SELL"
                signal_confidence = sell_confidence / sell_signals if sell_signals > 0 else 0
            else:
                signal = "NEUTRAL"
                signal_confidence = 0
        
        exploitation_window = None
        for glitch in detected_glitches:
            if "exploitation_window" in glitch:
                if exploitation_window is None or glitch["exploitation_window"] < exploitation_window:
                    exploitation_window = glitch["exploitation_window"]
        
        expected_profit = None
        if signal != "NEUTRAL":
            magnitudes = []
            for glitch in detected_glitches:
                if glitch.get("exploitable", False) and glitch["signal"] == signal:
                    if "magnitude" in glitch:
                        magnitudes.append(glitch["magnitude"])
                    elif "price_change" in glitch:
                        magnitudes.append(abs(glitch["price_change"]))
            
            if magnitudes:
                expected_profit = np.mean(magnitudes)
        
        return {
            "signal": signal,
            "confidence": signal_confidence,
            "exploitation_window": exploitation_window,
            "expected_profit": expected_profit
        }
    
    def get_detection_stats(self):
        """
        Get detection statistics.
        
        Returns:
        - Dictionary with detection statistics
        """
        return self.detection_stats
