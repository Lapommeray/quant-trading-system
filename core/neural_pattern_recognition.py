"""
Neural Pattern Recognition
Detects market patterns using neural network approaches
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
import random
from collections import deque

class NeuralPatternRecognition:
    def __init__(self):
        """Initialize Neural Pattern Recognition"""
        self.patterns = {}
        self.last_update = {}
        self.update_frequency = timedelta(hours=4)
        self.memory_length = 100  # Number of price points to remember
        self.price_memory = deque(maxlen=self.memory_length)
        self.pattern_library = self._initialize_pattern_library()
        self.detected_patterns = {}
        
    def _initialize_pattern_library(self):
        """Initialize library of known patterns"""
        patterns = {
            "head_and_shoulders": {
                "points": 7,
                "template": [0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0],
                "confidence_threshold": 0.85,
                "signal": "SELL"
            },
            "inverse_head_and_shoulders": {
                "points": 7,
                "template": [1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0],
                "confidence_threshold": 0.85,
                "signal": "BUY"
            },
            "double_top": {
                "points": 5,
                "template": [0.0, 1.0, 0.5, 1.0, 0.0],
                "confidence_threshold": 0.8,
                "signal": "SELL"
            },
            "double_bottom": {
                "points": 5,
                "template": [1.0, 0.0, 0.5, 0.0, 1.0],
                "confidence_threshold": 0.8,
                "signal": "BUY"
            }
        }
        
        return patterns
    
    def update_price_memory(self, symbol, price, timestamp=None):
        """
        Update price memory with new price point
        
        Parameters:
        - symbol: Trading symbol
        - price: Current price
        - timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        self.price_memory.append({
            "symbol": symbol,
            "price": price,
            "timestamp": timestamp
        })
        
        self.last_update[symbol] = datetime.now()
        
        self._detect_patterns(symbol)
        
        return len(self.price_memory)
    
    def _normalize_prices(self, prices):
        """Normalize prices to 0-1 range for pattern matching"""
        if len(prices) < 2:
            return prices
            
        min_price = min(prices)
        max_price = max(prices)
        
        if max_price == min_price:
            return [0.5] * len(prices)
            
        return [(p - min_price) / (max_price - min_price) for p in prices]
    
    def _detect_patterns(self, symbol):
        """
        Detect patterns in price memory
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dictionary with detected patterns
        """
        if len(self.price_memory) < 5:
            return {}
            
        prices = [point["price"] for point in self.price_memory 
                 if point["symbol"] == symbol]
        
        if len(prices) < 5:
            return {}
            
        detected = {}
        
        for pattern_name, pattern_info in self.pattern_library.items():
            points = pattern_info["points"]
            template = pattern_info["template"]
            threshold = pattern_info["confidence_threshold"]
            
            if len(prices) < points:
                continue
                
            max_confidence = 0
            best_window = None
            
            for window_size in range(points * 2, min(len(prices), points * 10), points):
                window = prices[-window_size:]
                
                for i in range(len(window) - points + 1):
                    segment = window[i:i+points]
                    normalized = self._normalize_prices(segment)
                    
                    confidence = self._calculate_pattern_confidence(normalized, template)
                    
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_window = segment
            
            if max_confidence >= threshold:
                detected[pattern_name] = {
                    "confidence": max_confidence,
                    "signal": pattern_info["signal"],
                    "detected_at": datetime.now().isoformat(),
                    "window": best_window if best_window else []
                }
        
        self.detected_patterns[symbol] = detected
        
        return detected
    
    def _calculate_pattern_confidence(self, normalized_prices, template):
        """
        Calculate confidence of pattern match
        
        Parameters:
        - normalized_prices: Normalized price points
        - template: Pattern template
        
        Returns:
        - Confidence score (0-1)
        """
        if len(normalized_prices) != len(template):
            return 0
            
        mse = sum((normalized_prices[i] - template[i]) ** 2 for i in range(len(template))) / len(template)
        
        confidence = 1 - math.sqrt(mse)
        
        return max(0, min(1, confidence))
    
    def analyze_neural_patterns(self, symbol, current_price=None):
        """
        Analyze neural patterns for trading signals
        
        Parameters:
        - symbol: Trading symbol
        - current_price: Current market price (optional)
        
        Returns:
        - Dictionary with signal information
        """
        if current_price is not None:
            self.update_price_memory(symbol, current_price)
            
        patterns = self.detected_patterns.get(symbol, {})
        
        if not patterns:
            return {
                "direction": None,
                "confidence": 0,
                "message": "No neural patterns detected",
                "patterns": {}
            }
            
        buy_confidence = 0
        sell_confidence = 0
        buy_count = 0
        sell_count = 0
        
        for pattern_name, pattern_info in patterns.items():
            if pattern_info["signal"] == "BUY":
                buy_confidence += pattern_info["confidence"]
                buy_count += 1
            elif pattern_info["signal"] == "SELL":
                sell_confidence += pattern_info["confidence"]
                sell_count += 1
        
        if buy_count > 0 and sell_count > 0:
            if buy_confidence / buy_count > sell_confidence / sell_count:
                direction = "BUY"
                confidence = buy_confidence / buy_count
                message = f"Bullish neural patterns detected ({buy_count} patterns)"
            else:
                direction = "SELL"
                confidence = sell_confidence / sell_count
                message = f"Bearish neural patterns detected ({sell_count} patterns)"
        elif buy_count > 0:
            direction = "BUY"
            confidence = buy_confidence / buy_count
            message = f"Bullish neural patterns detected ({buy_count} patterns)"
        elif sell_count > 0:
            direction = "SELL"
            confidence = sell_confidence / sell_count
            message = f"Bearish neural patterns detected ({sell_count} patterns)"
        else:
            direction = None
            confidence = 0
            message = "No clear neural patterns detected"
        
        return {
            "direction": direction,
            "confidence": confidence,
            "message": message,
            "patterns": patterns
        }
    
    def get_neural_report(self, symbol):
        """
        Generate detailed neural pattern report for a symbol
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dictionary with detailed neural pattern metrics
        """
        patterns = self.detected_patterns.get(symbol, {})
        
        pattern_counts = {
            "bullish": 0,
            "bearish": 0,
            "neutral": 0
        }
        
        for pattern_name, pattern_info in patterns.items():
            if pattern_info["signal"] == "BUY":
                pattern_counts["bullish"] += 1
            elif pattern_info["signal"] == "SELL":
                pattern_counts["bearish"] += 1
            else:
                pattern_counts["neutral"] += 1
        
        report = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "pattern_counts": pattern_counts,
            "detected_patterns": patterns,
            "price_memory_length": len(self.price_memory),
            "analysis": self.analyze_neural_patterns(symbol)
        }
        
        return report
