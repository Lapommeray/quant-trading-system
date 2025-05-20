"""
Market Maker Analyzer

A realistic implementation of market manipulation detection using order book analysis
and volume profiling instead of fictional mind-reading technologies.
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore
import datetime
import json
import logging

class MarketMakerAnalyzer:
    """
    A realistic implementation of market manipulation detection.
    Replaces the fictional MarketMakerMindReader with practical techniques.
    """
    
    def __init__(self, sensitivity=0.75, lookback_periods=20, min_volume_threshold=100000):
        """
        Initialize the MarketMakerAnalyzer
        
        Parameters:
        - sensitivity: Detection sensitivity (0.0-1.0)
        - lookback_periods: Number of periods to analyze for patterns
        - min_volume_threshold: Minimum volume to consider significant
        """
        self.sensitivity = sensitivity
        self.lookback_periods = lookback_periods
        self.min_volume_threshold = min_volume_threshold
        
        self.detection_history = []
        self.last_detection = None
        self.dark_pool_data = {}
        self.order_book_snapshots = []
        self.volume_profile = {}
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("MarketMakerAnalyzer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def update_order_book(self, symbol, order_book_data):
        """
        Update order book data
        
        Parameters:
        - symbol: Symbol to update
        - order_book_data: Order book data (bids and asks)
        """
        order_book_data["timestamp"] = datetime.datetime.now()
        order_book_data["symbol"] = symbol
        
        self.order_book_snapshots.append(order_book_data)
        
        if len(self.order_book_snapshots) > self.lookback_periods:
            self.order_book_snapshots = self.order_book_snapshots[-self.lookback_periods:]
    
    def update_volume_profile(self, symbol, price, volume):
        """
        Update volume profile
        
        Parameters:
        - symbol: Symbol to update
        - price: Price level
        - volume: Volume at price level
        """
        if symbol not in self.volume_profile:
            self.volume_profile[symbol] = {}
        
        price_key = str(round(price, 2))
        
        if price_key not in self.volume_profile[symbol]:
            self.volume_profile[symbol][price_key] = 0
        
        self.volume_profile[symbol][price_key] += volume
    
    def update_dark_pool_data(self, symbol, dark_pool_data):
        """
        Update dark pool data
        
        Parameters:
        - symbol: Symbol to update
        - dark_pool_data: Dark pool transaction data
        """
        if symbol not in self.dark_pool_data:
            self.dark_pool_data[symbol] = []
        
        dark_pool_data["timestamp"] = datetime.datetime.now()
        
        self.dark_pool_data[symbol].append(dark_pool_data)
        
        if len(self.dark_pool_data[symbol]) > self.lookback_periods:
            self.dark_pool_data[symbol] = self.dark_pool_data[symbol][-self.lookback_periods:]
    
    def detect_spoofing(self, symbol):
        """
        Detect spoofing patterns in order book
        
        Parameters:
        - symbol: Symbol to analyze
        
        Returns:
        - Dictionary with spoofing detection results
        """
        if not self.order_book_snapshots:
            return {
                "spoofing_detected": False,
                "confidence": 0.0,
                "details": "No order book data available"
            }
        
        symbol_snapshots = [s for s in self.order_book_snapshots if s["symbol"] == symbol]
        
        if not symbol_snapshots:
            return {
                "spoofing_detected": False,
                "confidence": 0.0,
                "details": f"No order book data for {symbol}"
            }
        
        spoofing_indicators = []
        
        for i in range(1, len(symbol_snapshots)):
            prev_snapshot = symbol_snapshots[i-1]
            curr_snapshot = symbol_snapshots[i]
            
            for price in prev_snapshot.get("bids", {}).keys():
                if price in curr_snapshot.get("bids", {}):
                    prev_size = prev_snapshot["bids"][price]
                    curr_size = curr_snapshot["bids"][price]
                    
                    if prev_size > self.min_volume_threshold and curr_size < prev_size * 0.5:
                        spoofing_indicators.append({
                            "type": "bid_disappeared",
                            "price": float(price),
                            "prev_size": prev_size,
                            "curr_size": curr_size,
                            "timestamp": curr_snapshot["timestamp"]
                        })
            
            for price in prev_snapshot.get("asks", {}).keys():
                if price in curr_snapshot.get("asks", {}):
                    prev_size = prev_snapshot["asks"][price]
                    curr_size = curr_snapshot["asks"][price]
                    
                    if prev_size > self.min_volume_threshold and curr_size < prev_size * 0.5:
                        spoofing_indicators.append({
                            "type": "ask_disappeared",
                            "price": float(price),
                            "prev_size": prev_size,
                            "curr_size": curr_size,
                            "timestamp": curr_snapshot["timestamp"]
                        })
        
        confidence = min(0.95, len(spoofing_indicators) / 10 * self.sensitivity)
        spoofing_detected = confidence > 0.5
        
        result = {
            "spoofing_detected": spoofing_detected,
            "confidence": confidence,
            "indicators": spoofing_indicators,
            "details": f"Found {len(spoofing_indicators)} potential spoofing patterns"
        }
        
        return result
    
    def detect_iceberg_orders(self, symbol, trades_data):
        """
        Detect iceberg orders (hidden liquidity)
        
        Parameters:
        - symbol: Symbol to analyze
        - trades_data: Recent trades data
        
        Returns:
        - Dictionary with iceberg detection results
        """
        if not trades_data or len(trades_data) < 10:
            return {
                "iceberg_detected": False,
                "confidence": 0.0,
                "details": "Insufficient trades data"
            }
        
        df = pd.DataFrame(trades_data)
        
        if "timestamp" in df.columns:
            df["time_window"] = df["timestamp"].dt.floor("1min")
        else:
            return {
                "iceberg_detected": False,
                "confidence": 0.0,
                "details": "Timestamp data missing"
            }
        
        grouped = df.groupby(["price", "time_window"]).agg({"volume": "sum", "timestamp": "count"})
        grouped = grouped.reset_index()
        
        iceberg_indicators = []
        
        for price, group in grouped.groupby("price"):
            if len(group) >= 3:  # At least 3 time windows with activity at this price
                if group["timestamp"].sum() >= 10:  # At least 10 trades
                    if group["volume"].sum() > self.min_volume_threshold:  # Significant volume
                        iceberg_indicators.append({
                            "price": price,
                            "total_volume": group["volume"].sum(),
                            "trade_count": group["timestamp"].sum(),
                            "time_windows": len(group)
                        })
        
        confidence = min(0.95, len(iceberg_indicators) / 3 * self.sensitivity)
        iceberg_detected = confidence > 0.5
        
        result = {
            "iceberg_detected": iceberg_detected,
            "confidence": confidence,
            "indicators": iceberg_indicators,
            "details": f"Found {len(iceberg_indicators)} potential iceberg orders"
        }
        
        return result
    
    def detect_layering(self, symbol):
        """
        Detect layering patterns in order book
        
        Parameters:
        - symbol: Symbol to analyze
        
        Returns:
        - Dictionary with layering detection results
        """
        if not self.order_book_snapshots:
            return {
                "layering_detected": False,
                "confidence": 0.0,
                "details": "No order book data available"
            }
        
        symbol_snapshots = [s for s in self.order_book_snapshots if s["symbol"] == symbol]
        
        if not symbol_snapshots or len(symbol_snapshots) < 2:
            return {
                "layering_detected": False,
                "confidence": 0.0,
                "details": f"Insufficient order book data for {symbol}"
            }
        
        latest_snapshot = symbol_snapshots[-1]
        
        layering_indicators = []
        
        bid_prices = sorted([float(p) for p in latest_snapshot.get("bids", {}).keys()], reverse=True)
        
        if len(bid_prices) >= 5:  # Need at least 5 price levels
            bid_sizes = [latest_snapshot["bids"][str(p)] for p in bid_prices]
            
            if len(set(bid_sizes)) < len(bid_sizes) * 0.5:  # Many similar sizes
                layering_indicators.append({
                    "type": "bid_layering",
                    "price_levels": len(bid_prices),
                    "similar_sizes": len(set(bid_sizes)),
                    "timestamp": latest_snapshot["timestamp"]
                })
        
        ask_prices = sorted([float(p) for p in latest_snapshot.get("asks", {}).keys()])
        
        if len(ask_prices) >= 5:  # Need at least 5 price levels
            ask_sizes = [latest_snapshot["asks"][str(p)] for p in ask_prices]
            
            if len(set(ask_sizes)) < len(ask_sizes) * 0.5:  # Many similar sizes
                layering_indicators.append({
                    "type": "ask_layering",
                    "price_levels": len(ask_prices),
                    "similar_sizes": len(set(ask_sizes)),
                    "timestamp": latest_snapshot["timestamp"]
                })
        
        confidence = min(0.95, len(layering_indicators) / 2 * self.sensitivity)
        layering_detected = confidence > 0.5
        
        result = {
            "layering_detected": layering_detected,
            "confidence": confidence,
            "indicators": layering_indicators,
            "details": f"Found {len(layering_indicators)} potential layering patterns"
        }
        
        return result
    
    def analyze_dark_pool_activity(self, symbol):
        """
        Analyze dark pool activity for unusual patterns
        
        Parameters:
        - symbol: Symbol to analyze
        
        Returns:
        - Dictionary with dark pool analysis results
        """
        if symbol not in self.dark_pool_data or not self.dark_pool_data[symbol]:
            return {
                "unusual_activity": False,
                "confidence": 0.0,
                "details": "No dark pool data available"
            }
        
        total_volume = sum(d.get("volume", 0) for d in self.dark_pool_data[symbol])
        
        avg_trade_size = total_volume / len(self.dark_pool_data[symbol])
        
        large_trades = [d for d in self.dark_pool_data[symbol] if d.get("volume", 0) > avg_trade_size * 3]
        
        large_trade_volume = sum(d.get("volume", 0) for d in large_trades)
        large_trade_percentage = large_trade_volume / total_volume if total_volume > 0 else 0
        
        unusual_activity = large_trade_percentage > 0.3 and len(large_trades) >= 3
        
        confidence = min(0.95, large_trade_percentage * self.sensitivity)
        
        result = {
            "unusual_activity": unusual_activity,
            "confidence": confidence,
            "large_trades": len(large_trades),
            "large_trade_percentage": large_trade_percentage,
            "details": f"Found {len(large_trades)} large dark pool trades ({large_trade_percentage:.2%} of volume)"
        }
        
        return result
    
    def detect_manipulation(self, symbol):
        """
        Detect market manipulation patterns
        
        Parameters:
        - symbol: Symbol to analyze
        
        Returns:
        - True if manipulation is detected, False otherwise
        """
        spoofing_result = self.detect_spoofing(symbol)
        layering_result = self.detect_layering(symbol)
        dark_pool_result = self.analyze_dark_pool_activity(symbol)
        
        manipulation_detected = (
            spoofing_result["spoofing_detected"] or
            layering_result["layering_detected"] or
            dark_pool_result["unusual_activity"]
        )
        
        confidence = max(
            spoofing_result["confidence"],
            layering_result["confidence"],
            dark_pool_result["confidence"]
        )
        
        result = {
            "manipulation_detected": manipulation_detected,
            "confidence": confidence,
            "spoofing": spoofing_result,
            "layering": layering_result,
            "dark_pool": dark_pool_result,
            "timestamp": datetime.datetime.now()
        }
        
        if manipulation_detected:
            self.logger.warning(f"Manipulation detected for {symbol} with {confidence:.2%} confidence")
            self.logger.info(json.dumps(result, default=str))
        
        self.last_detection = result
        self.detection_history.append(result)
        
        if len(self.detection_history) > 100:
            self.detection_history = self.detection_history[-100:]
        
        return manipulation_detected
    
    def get_detection_stats(self, symbol=None, lookback_days=7):
        """
        Get manipulation detection statistics
        
        Parameters:
        - symbol: Symbol to analyze (optional)
        - lookback_days: Number of days to look back
        
        Returns:
        - Dictionary with detection statistics
        """
        if symbol:
            detections = [d for d in self.detection_history if d.get("symbol") == symbol]
        else:
            detections = self.detection_history
        
        cutoff_time = datetime.datetime.now() - datetime.timedelta(days=lookback_days)
        recent_detections = [d for d in detections if d.get("timestamp", datetime.datetime.now()) > cutoff_time]
        
        total_detections = len(recent_detections)
        spoofing_detections = sum(1 for d in recent_detections if d.get("spoofing", {}).get("spoofing_detected", False))
        layering_detections = sum(1 for d in recent_detections if d.get("layering", {}).get("layering_detected", False))
        dark_pool_detections = sum(1 for d in recent_detections if d.get("dark_pool", {}).get("unusual_activity", False))
        
        if recent_detections:
            avg_confidence = sum(d.get("confidence", 0) for d in recent_detections) / len(recent_detections)
        else:
            avg_confidence = 0.0
        
        return {
            "total_detections": total_detections,
            "spoofing_detections": spoofing_detections,
            "layering_detections": layering_detections,
            "dark_pool_detections": dark_pool_detections,
            "avg_confidence": avg_confidence,
            "lookback_days": lookback_days,
            "symbol": symbol
        }
