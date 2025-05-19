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
        self.circuit_breaker_active = False
        self.circuit_breaker_cooldown = 0  # Cooldown period after circuit breaker activation
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
        self.volatility_thresholds = {
            "default": {
                "crisis": 0.25,
                "pre_crisis": 0.15,
                "high_volatility": 0.12
            },
            "crypto": {  # Higher thresholds for crypto
                "crisis": 0.5,
                "pre_crisis": 0.3,
                "high_volatility": 0.25
            },
            "forex": {  # Lower thresholds for forex
                "crisis": 0.15,
                "pre_crisis": 0.1,
                "high_volatility": 0.08
            }
        }
        
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
        
        short_term_volatility = np.std(returns[-5:]) * math.sqrt(252) if len(returns) >= 5 else volatility
        
        volatility_ratio = short_term_volatility / volatility if volatility > 0 else 1.0
        
        true_ranges = []
        for i in range(1, len(symbol_data)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - prices[i-1])
            tr3 = abs(lows[i] - prices[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        atr = np.mean(true_ranges[-self.volatility_window:])
        atr_pct = atr / prices[-1]  # ATR as percentage of price
        
        short_term_atr = np.mean(true_ranges[-5:]) if len(true_ranges) >= 5 else atr
        short_term_atr_pct = short_term_atr / prices[-1]
        
        atr_ratio = short_term_atr / atr if atr > 0 else 1.0
        
        price_change_pct = (prices[-1] - prices[0]) / prices[0]
        
        ma20 = np.mean(prices[-20:]) if len(prices) >= 20 else None
        ma50 = np.mean(prices[-50:]) if len(prices) >= 50 else None
        ma200 = np.mean(prices[-200:]) if len(prices) >= 200 else None
        
        max_price = max(prices)
        drawdown = (max_price - prices[-1]) / max_price if max_price > prices[-1] else 0
        
        rolling_max = prices[0]
        max_drawdown = 0
        for price in prices[1:]:
            rolling_max = max(rolling_max, price)
            current_drawdown = (rolling_max - price) / rolling_max if rolling_max > 0 else 0
            max_drawdown = max(max_drawdown, current_drawdown)
        
        rapid_change = False
        if len(prices) >= 3:
            last_return = (prices[-1] - prices[-2]) / prices[-2]
            if abs(last_return) > 0.03:  # 3% move in a single period
                rapid_change = True
        
        # Determine asset class for appropriate thresholds
        asset_class = "default"
        if "BTC" in symbol or "ETH" in symbol:
            asset_class = "crypto"
        elif "USD" in symbol and len(symbol) <= 7:  # Forex pairs like EUR/USD
            asset_class = "forex"
        
        regime_metrics = {
            "volatility": volatility,
            "short_term_volatility": short_term_volatility,
            "volatility_ratio": volatility_ratio,
            "atr": atr,
            "atr_pct": atr_pct,
            "short_term_atr": short_term_atr,
            "short_term_atr_pct": short_term_atr_pct,
            "atr_ratio": atr_ratio,
            "price_change_pct": price_change_pct,
            "ma20": ma20,
            "ma50": ma50,
            "ma200": ma200,
            "current_price": prices[-1],
            "max_price": max_price,
            "min_price": min(prices),
            "drawdown": drawdown,
            "max_drawdown": max_drawdown,
            "rapid_change": rapid_change,
            "asset_class": asset_class
        }
        
        # Check for circuit breaker conditions
        if rapid_change and (volatility_ratio > 2.0 or atr_ratio > 2.0):
            self.circuit_breaker_active = True
            self.circuit_breaker_cooldown = 5  # Cooldown for 5 periods
            print(f"⚠️ CIRCUIT BREAKER ACTIVATED: Extreme volatility spike detected for {symbol}")
        elif self.circuit_breaker_cooldown > 0:
            self.circuit_breaker_cooldown -= 1
            if self.circuit_breaker_cooldown == 0:
                self.circuit_breaker_active = False
                print(f"✅ CIRCUIT BREAKER DEACTIVATED: Volatility has normalized for {symbol}")
        
        regime = self._classify_regime(regime_metrics)
        
        if self.circuit_breaker_active:
            regime["regime"] = "crisis"
            regime["confidence"] = 0.99
            regime["message"] = "Circuit breaker active - trading halted"
        
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
        
        if metrics["volatility"] > 0.25 or metrics["drawdown"] > 0.1 or metrics["atr_pct"] > 0.03:
            regime["regime"] = "crisis"
            regime["confidence"] = min(0.95, max(0.7, metrics["volatility"] / 0.25))
            regime["message"] = "Market crisis detected - trading halted"
            return regime
        
        # Earlier detection of pre-crisis conditions
        if metrics["volatility"] > 0.15 or metrics["drawdown"] > 0.05 or metrics["atr_pct"] > 0.02:
            regime["regime"] = "pre_crisis"
            regime["confidence"] = min(0.9, max(0.6, metrics["volatility"] / 0.15))
            regime["message"] = "Pre-crisis conditions detected - reduced position sizing"
            return regime
        
        recent_returns = [point["price"] for point in self.price_memory[-10:]]
        if len(recent_returns) >= 10:
            recent_volatility = np.std(np.diff(recent_returns) / recent_returns[:-1]) * math.sqrt(252)
            if recent_volatility > 0.2 and recent_volatility > metrics["volatility"] * 1.5:
                regime["regime"] = "pre_crisis"
                regime["confidence"] = min(0.9, max(0.6, recent_volatility / 0.2))
                regime["message"] = "Rapidly increasing volatility detected - caution advised"
                return regime
        
        if metrics["volatility"] > 0.12:
            regime["regime"] = "high_volatility"
            regime["confidence"] = min(0.85, max(0.6, metrics["volatility"] / 0.12))
            regime["message"] = "High volatility regime detected - reduce position sizes"
            return regime
        
        if metrics["volatility"] < 0.08:
            regime["regime"] = "low_volatility"
            regime["confidence"] = min(0.8, max(0.6, 0.08 / metrics["volatility"]))
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
        
        if metrics["drawdown"] < 0.03 and metrics["price_change_pct"] > 0.05 and metrics.get("ma20", 0) > metrics.get("ma50", 0):
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
        - Float position sizing multiplier (0.0-1.0)
        """
        regime = self.get_current_regime(symbol)
        
        position_sizing = 1.0
        
        if regime["regime"] == "unknown":
            return False, "Unknown market regime", 0.0
        
        if regime["regime"] == "crisis":
            return False, "Crisis regime - trading halted", 0.0
        
        if regime["regime"] == "pre_crisis":
            position_sizing = 0.25  # Only 25% of normal position size
            if strategy_type == "trend":
                return True, "Pre-crisis regime - trend strategies only with 25% position size", position_sizing
            else:
                return False, "Pre-crisis regime - non-trend strategies halted", 0.0
        
        if regime["regime"] == "high_volatility":
            position_sizing = 0.5  # Only 50% of normal position size
            if strategy_type == "volatility":
                return True, "High volatility regime - volatility strategies with 50% position size", position_sizing
            else:
                return False, "High volatility regime - non-volatility strategies halted", 0.0
        
        if strategy_type == "trend":
            if regime["regime"] in ["trending_up", "trending_down"]:
                return True, f"{regime['regime']} regime - ideal for trend strategies", 1.0
            elif regime["regime"] == "recovery":
                return True, f"{regime['regime']} regime - trend strategies with 75% position size", 0.75
            else:
                return True, f"{regime['regime']} regime - trend strategies allowed", 0.8
                
        elif strategy_type == "mean_reversion":
            if regime["regime"] == "low_volatility":
                return True, "Low volatility regime - ideal for mean reversion", 1.0
            elif regime["regime"] in ["trending_up", "trending_down"]:
                return False, f"{regime['regime']} regime - mean reversion strategies halted", 0.0
            else:
                return True, f"{regime['regime']} regime - mean reversion allowed", 0.8
                
        elif strategy_type == "volatility":
            if regime["regime"] == "high_volatility":
                return True, "High volatility regime - ideal for volatility strategies", 0.75
            elif regime["regime"] == "low_volatility":
                return False, "Low volatility regime - volatility strategies halted", 0.0
            else:
                return True, f"{regime['regime']} regime - volatility strategies allowed", 0.8
        
        if regime["regime"] in ["normal", "low_volatility"]:
            return True, f"{regime['regime']} regime - trading allowed", 1.0
        elif regime["regime"] in ["trending_up", "trending_down", "recovery"]:
            return True, f"{regime['regime']} regime - trading allowed with 80% position size", 0.8
        else:
            return True, f"{regime['regime']} regime - trading allowed with 60% position size", 0.6
    
    def get_regime_report(self, symbol):
        """
        Generate detailed regime report for a symbol
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dictionary with detailed regime metrics
        """
        regime = self.get_current_regime(symbol)
        
        trend_allowed, trend_reason, trend_sizing = self.should_trade(symbol, "trend")
        mean_reversion_allowed, mean_reversion_reason, mean_reversion_sizing = self.should_trade(symbol, "mean_reversion")
        volatility_allowed, volatility_reason, volatility_sizing = self.should_trade(symbol, "volatility")
        all_allowed, all_reason, all_sizing = self.should_trade(symbol, "all")
        
        risk_level = "Low"
        if regime["regime"] in ["pre_crisis", "high_volatility"]:
            risk_level = "High"
        elif regime["regime"] == "crisis":
            risk_level = "Extreme"
        elif regime["regime"] in ["trending_down"]:
            risk_level = "Medium-High"
        elif regime["regime"] in ["trending_up", "recovery"]:
            risk_level = "Medium"
        
        position_size_multiplier = all_sizing
        
        stop_loss_multiplier = 1.0
        if regime["regime"] == "crisis":
            stop_loss_multiplier = 0.5  # Much tighter stops
        elif regime["regime"] in ["pre_crisis", "high_volatility"]:
            stop_loss_multiplier = 0.7  # Tighter stops
        
        report = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "current_regime": regime,
            "risk_level": risk_level,
            "position_size_multiplier": position_size_multiplier,
            "stop_loss_multiplier": stop_loss_multiplier,
            "strategy_recommendations": {
                "trend": {
                    "allowed": trend_allowed,
                    "reason": trend_reason,
                    "position_sizing": trend_sizing
                },
                "mean_reversion": {
                    "allowed": mean_reversion_allowed,
                    "reason": mean_reversion_reason,
                    "position_sizing": mean_reversion_sizing
                },
                "volatility": {
                    "allowed": volatility_allowed,
                    "reason": volatility_reason,
                    "position_sizing": volatility_sizing
                },
                "all": {
                    "allowed": all_allowed,
                    "reason": all_reason,
                    "position_sizing": all_sizing
                }
            },
            "price_memory_length": len(self.price_memory),
            "circuit_breaker_active": regime["regime"] == "crisis"
        }
        
        return report
