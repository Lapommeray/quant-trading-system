"""
Market Regime Detection
Detects and classifies market regimes for adaptive trading
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
from collections import deque

class MarketRegimeDetection:
    def __init__(self):
        """Initialize Market Regime Detection"""
        self.regimes = {}
        self.last_update = {}
        self.update_frequency = timedelta(hours=1)
        self.price_memory = deque(maxlen=500)  # Store up to 500 price points
        self.volatility_window = 20  # Window for volatility calculation
        self.regime_types = [
            "normal",           # Normal trading conditions
            "trending_up",      # Strong uptrend
            "trending_down",    # Strong downtrend
            "high_volatility",  # High volatility but no clear direction
            "low_volatility",   # Low volatility, range-bound
            "pre_crisis",       # Early signs of market stress
            "crisis",           # Full market crisis
            "recovery"          # Post-crisis recovery
        ]
        
    def update_price_memory(self, symbol, price, high=None, low=None, timestamp=None):
        """
        Update price memory with new price point
        
        Parameters:
        - symbol: Trading symbol
        - price: Current price
        - high: High price (optional)
        - low: Low price (optional)
        - timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        if high is None:
            high = price * 1.001  # Estimated high
        if low is None:
            low = price * 0.999  # Estimated low
            
        self.price_memory.append({
            "symbol": symbol,
            "price": price,
            "high": high,
            "low": low,
            "timestamp": timestamp
        })
        
        self.last_update[symbol] = datetime.now()
        
        self._detect_regime(symbol)
        
        return len(self.price_memory)
    
    def _detect_regime(self, symbol):
        """
        Detect market regime for a symbol
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dictionary with regime information
        """
        if len(self.price_memory) < self.volatility_window:
            return {
                "regime": "unknown",
                "confidence": 0,
                "message": "Insufficient data for regime detection"
            }
            
        symbol_data = [point for point in self.price_memory if point["symbol"] == symbol]
        
        if len(symbol_data) < self.volatility_window:
            return {
                "regime": "unknown",
                "confidence": 0,
                "message": "Insufficient data for regime detection"
            }
            
        symbol_data = sorted(symbol_data, key=lambda x: x["timestamp"])
        
        prices = [point["price"] for point in symbol_data]
        highs = [point["high"] for point in symbol_data]
        lows = [point["low"] for point in symbol_data]
        
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        
        recent_returns = returns[-self.volatility_window:]
        volatility = np.std(recent_returns) * math.sqrt(252)  # Annualized
        
        true_ranges = []
        for i in range(1, len(symbol_data)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - prices[i-1])
            tr3 = abs(lows[i] - prices[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        atr = np.mean(true_ranges[-self.volatility_window:])
        atr_pct = atr / prices[-1]  # ATR as percentage of price
        
        price_change_pct = (prices[-1] - prices[0]) / prices[0]
        
        ma20 = np.mean(prices[-20:]) if len(prices) >= 20 else None
        ma50 = np.mean(prices[-50:]) if len(prices) >= 50 else None
        ma200 = np.mean(prices[-200:]) if len(prices) >= 200 else None
        
        regime_metrics = {
            "volatility": volatility,
            "atr": atr,
            "atr_pct": atr_pct,
            "price_change_pct": price_change_pct,
            "ma20": ma20,
            "ma50": ma50,
            "ma200": ma200,
            "current_price": prices[-1],
            "max_price": max(prices),
            "min_price": min(prices),
            "drawdown": (max(prices) - prices[-1]) / max(prices) if max(prices) > prices[-1] else 0
        }
        
        regime = self._classify_regime(regime_metrics)
        
        self.regimes[symbol] = regime
        
        return regime
    
    def _classify_regime(self, metrics):
        """
        Classify market regime based on metrics
        
        Parameters:
        - metrics: Dictionary with regime metrics
        
        Returns:
        - Dictionary with regime classification
        """
        regime = {
            "regime": "normal",
            "confidence": 0.5,
            "message": "Normal market conditions",
            "metrics": metrics
        }
        
        if metrics["volatility"] > 0.5 or metrics["drawdown"] > 0.2:
            regime["regime"] = "crisis"
            regime["confidence"] = min(0.95, max(0.7, metrics["volatility"] / 0.5))
            regime["message"] = "Market crisis detected"
            return regime
        
        if metrics["volatility"] > 0.3 or metrics["drawdown"] > 0.1:
            regime["regime"] = "pre_crisis"
            regime["confidence"] = min(0.9, max(0.6, metrics["volatility"] / 0.3))
            regime["message"] = "Pre-crisis conditions detected"
            return regime
        
        if metrics["volatility"] > 0.2:
            regime["regime"] = "high_volatility"
            regime["confidence"] = min(0.85, max(0.6, metrics["volatility"] / 0.2))
            regime["message"] = "High volatility regime detected"
            return regime
        
        if metrics["volatility"] < 0.1:
            regime["regime"] = "low_volatility"
            regime["confidence"] = min(0.8, max(0.6, 0.1 / metrics["volatility"]))
            regime["message"] = "Low volatility regime detected"
            return regime
        
        if metrics["price_change_pct"] > 0.05 and metrics.get("ma20") > metrics.get("ma50", 0):
            regime["regime"] = "trending_up"
            regime["confidence"] = min(0.85, max(0.6, metrics["price_change_pct"] / 0.05))
            regime["message"] = "Uptrend regime detected"
            return regime
        
        if metrics["price_change_pct"] < -0.05 and metrics.get("ma20") < metrics.get("ma50", float('inf')):
            regime["regime"] = "trending_down"
            regime["confidence"] = min(0.85, max(0.6, abs(metrics["price_change_pct"]) / 0.05))
            regime["message"] = "Downtrend regime detected"
            return regime
        
        if metrics["drawdown"] < 0.05 and metrics["price_change_pct"] > 0.05 and metrics.get("ma20", 0) > metrics.get("ma50", 0):
            regime["regime"] = "recovery"
            regime["confidence"] = min(0.8, max(0.6, metrics["price_change_pct"] / 0.05))
            regime["message"] = "Market recovery regime detected"
            return regime
        
        return regime
    
    def get_current_regime(self, symbol):
        """
        Get current market regime for a symbol
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dictionary with regime information
        """
        if symbol in self.regimes:
            return self.regimes[symbol]
        else:
            return {
                "regime": "unknown",
                "confidence": 0,
                "message": "No regime data available"
            }
    
    def should_trade(self, symbol, strategy_type="all"):
        """
        Determine if trading should be allowed in current regime
        
        Parameters:
        - symbol: Trading symbol
        - strategy_type: Strategy type ("trend", "mean_reversion", "volatility", "all")
        
        Returns:
        - Boolean indicating whether trading should be allowed
        - String with reason
        """
        regime = self.get_current_regime(symbol)
        
        if regime["regime"] == "unknown":
            return False, "Unknown market regime"
        
        if regime["regime"] == "crisis":
            return False, "Crisis regime - trading halted"
        
        if regime["regime"] == "pre_crisis":
            if strategy_type == "trend":
                return True, "Pre-crisis regime - trend strategies only"
            else:
                return False, "Pre-crisis regime - non-trend strategies halted"
        
        if strategy_type == "trend":
            if regime["regime"] in ["trending_up", "trending_down"]:
                return True, f"{regime['regime']} regime - ideal for trend strategies"
            elif regime["regime"] == "high_volatility":
                return False, "High volatility regime - trend strategies halted"
            else:
                return True, f"{regime['regime']} regime - trend strategies allowed"
                
        elif strategy_type == "mean_reversion":
            if regime["regime"] == "low_volatility":
                return True, "Low volatility regime - ideal for mean reversion"
            elif regime["regime"] in ["trending_up", "trending_down"]:
                return False, f"{regime['regime']} regime - mean reversion strategies halted"
            else:
                return True, f"{regime['regime']} regime - mean reversion allowed"
                
        elif strategy_type == "volatility":
            if regime["regime"] == "high_volatility":
                return True, "High volatility regime - ideal for volatility strategies"
            elif regime["regime"] == "low_volatility":
                return False, "Low volatility regime - volatility strategies halted"
            else:
                return True, f"{regime['regime']} regime - volatility strategies allowed"
        
        return True, f"{regime['regime']} regime - trading allowed"
    
    def get_regime_report(self, symbol):
        """
        Generate detailed regime report for a symbol
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dictionary with detailed regime metrics
        """
        regime = self.get_current_regime(symbol)
        
        trend_allowed, trend_reason = self.should_trade(symbol, "trend")
        mean_reversion_allowed, mean_reversion_reason = self.should_trade(symbol, "mean_reversion")
        volatility_allowed, volatility_reason = self.should_trade(symbol, "volatility")
        
        report = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "current_regime": regime,
            "strategy_recommendations": {
                "trend": {
                    "allowed": trend_allowed,
                    "reason": trend_reason
                },
                "mean_reversion": {
                    "allowed": mean_reversion_allowed,
                    "reason": mean_reversion_reason
                },
                "volatility": {
                    "allowed": volatility_allowed,
                    "reason": volatility_reason
                }
            },
            "price_memory_length": len(self.price_memory)
        }
        
        return report
