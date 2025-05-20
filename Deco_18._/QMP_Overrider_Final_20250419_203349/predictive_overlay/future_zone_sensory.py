"""
Future Zone Sensory Line

Implements advanced future zone visualization based on gate and alignment consensus
for the QMP Overrider system, allowing traders to see high-probability price zones.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class FutureZoneSensory:
    """
    Generates future zone sensory lines based on gate consensus, alignment patterns,
    and transcendent intelligence signals.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the future zone sensory line generator.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.projection_horizon = 10  # Number of time steps to project
        self.zone_width_factor = 1.5  # Factor for zone width calculation
        self.last_zones = {}
    
    def generate_future_zones(self, symbol, history_data, gate_scores, transcendent_signal=None):
        """
        Generate future price zones based on gate consensus and alignment patterns.
        
        Parameters:
        - symbol: Trading symbol
        - history_data: Dictionary of DataFrames for different timeframes
        - gate_scores: Dictionary of gate scores from QMP engine
        - transcendent_signal: Signal from transcendent intelligence
        
        Returns:
        - Dictionary containing future zone data
        """
        latest_candle = None
        for timeframe in ["1m", "5m"]:
            if timeframe in history_data and not history_data[timeframe].empty:
                latest_candle = history_data[timeframe].iloc[-1]
                break
                
        if latest_candle is None:
            return {
                "success": False,
                "error": "No price data available"
            }
        
        volatility = self._calculate_volatility(history_data)
        
        gate_consensus = self._calculate_gate_consensus(gate_scores)
        
        consciousness_level = 0.5
        if transcendent_signal and "consciousness_level" in transcendent_signal:
            consciousness_level = transcendent_signal["consciousness_level"]
        
        direction_bias = 0.0
        
        if transcendent_signal and "type" in transcendent_signal:
            if transcendent_signal["type"] == "BUY":
                direction_bias = 0.0005 * transcendent_signal["strength"]
            elif transcendent_signal["type"] == "SELL":
                direction_bias = -0.0005 * transcendent_signal["strength"]
        
        if gate_consensus["direction"] == "bullish":
            direction_bias += 0.0003 * gate_consensus["strength"]
        elif gate_consensus["direction"] == "bearish":
            direction_bias -= 0.0003 * gate_consensus["strength"]
        
        future_zones = []
        
        current_time = self.algorithm.Time
        current_price = latest_candle["Close"]
        
        base_zone_width = current_price * volatility * self.zone_width_factor * (1.0 - (consciousness_level * 0.5))
        
        for i in range(self.projection_horizon):
            step_volatility = volatility * (1 + i * 0.1)
            
            price_change = current_price * (step_volatility * np.random.randn() + direction_bias)
            next_price = current_price + price_change
            
            zone_width = base_zone_width * (1 + i * 0.2)
            
            zone = {
                "time": current_time + timedelta(minutes=(i+1)*5),
                "center_price": next_price,
                "upper_bound": next_price + (zone_width / 2),
                "lower_bound": next_price - (zone_width / 2),
                "confidence": gate_consensus["strength"] * (1.0 - (i * 0.05)),  # Confidence decreases with time
                "direction": "bullish" if next_price > current_price else "bearish"
            }
            
            future_zones.append(zone)
            current_price = next_price
        
        self.last_zones[str(symbol)] = future_zones
        
        return {
            "success": True,
            "symbol": str(symbol),
            "timestamp": self.algorithm.Time,
            "latest_price": latest_candle["Close"],
            "future_zones": future_zones,
            "gate_consensus": gate_consensus,
            "consciousness_level": consciousness_level
        }
    
    def _calculate_volatility(self, history_data, timeframe="1m"):
        """
        Calculate recent price volatility.
        
        Parameters:
        - history_data: Dictionary of DataFrames for different timeframes
        - timeframe: Timeframe to use for volatility calculation
        
        Returns:
        - Volatility estimate
        """
        if timeframe not in history_data or history_data[timeframe].empty:
            return 0.001  # Default volatility if no data
            
        df = history_data[timeframe]
        
        returns = df["Close"].pct_change().dropna()
        
        if len(returns) < 10:
            return 0.001
            
        volatility = returns.std()
        
        return max(volatility, 0.0005)  # Ensure minimum volatility
    
    def _calculate_gate_consensus(self, gate_scores):
        """
        Calculate consensus from gate scores.
        
        Parameters:
        - gate_scores: Dictionary of gate scores from QMP engine
        
        Returns:
        - Gate consensus information
        """
        if not gate_scores:
            return {
                "direction": "neutral",
                "strength": 0.5,
                "consensus_level": 0.0
            }
        
        bullish_count = 0
        bearish_count = 0
        
        for gate, score in gate_scores.items():
            if "direction" in gate.lower() or "bias" in gate.lower():
                if score > 0.5:
                    bullish_count += 1
                else:
                    bearish_count += 1
        
        total_gates = len(gate_scores)
        avg_score = sum(gate_scores.values()) / total_gates if total_gates > 0 else 0.5
        
        if bullish_count > bearish_count:
            direction = "bullish"
            consensus_level = bullish_count / (bullish_count + bearish_count) if (bullish_count + bearish_count) > 0 else 0.5
        elif bearish_count > bullish_count:
            direction = "bearish"
            consensus_level = bearish_count / (bullish_count + bearish_count) if (bullish_count + bearish_count) > 0 else 0.5
        else:
            direction = "neutral"
            consensus_level = 0.5
        
        return {
            "direction": direction,
            "strength": avg_score,
            "consensus_level": consensus_level
        }
    
    def get_zone_accuracy(self, symbol, actual_price, time_index=0):
        """
        Calculate the accuracy of previous future zone predictions.
        
        Parameters:
        - symbol: Trading symbol
        - actual_price: The actual price that materialized
        - time_index: Index of the future zone to check
        
        Returns:
        - Accuracy score (0.0 to 1.0)
        """
        symbol_str = str(symbol)
        
        if symbol_str not in self.last_zones or not self.last_zones[symbol_str]:
            return 0.0
            
        if time_index >= len(self.last_zones[symbol_str]):
            return 0.0
            
        zone = self.last_zones[symbol_str][time_index]
        
        if actual_price >= zone["lower_bound"] and actual_price <= zone["upper_bound"]:
            zone_width = zone["upper_bound"] - zone["lower_bound"]
            distance_from_center = abs(actual_price - zone["center_price"])
            center_accuracy = 1.0 - (distance_from_center / (zone_width / 2))
            
            return max(0.5, center_accuracy)
        else:
            return 0.0
