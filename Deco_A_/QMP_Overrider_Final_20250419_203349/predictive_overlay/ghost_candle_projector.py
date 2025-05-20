"""
Ghost Candle Projector

Implements synthetic candle projection (Ghost Candles) for the QMP Overrider system,
allowing visualization of potential future price movements.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class GhostCandleProjector:
    """
    Projects synthetic future candles based on current market conditions,
    transcendent intelligence signals, and historical patterns.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the ghost candle projector.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.projection_horizon = 5  # Number of candles to project
        self.volatility_factor = 1.2  # Multiplier for volatility in projections
        self.last_projections = {}
    
    def calculate_volatility(self, history_data, timeframe="1m"):
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
    
    def project_ghost_candles(self, symbol, history_data, transcendent_signal=None):
        """
        Project synthetic future candles based on current market conditions.
        
        Parameters:
        - symbol: Trading symbol
        - history_data: Dictionary of DataFrames for different timeframes
        - transcendent_signal: Signal from transcendent intelligence
        
        Returns:
        - List of projected candles
        """
        latest_candle = None
        for timeframe in ["1m", "5m"]:
            if timeframe in history_data and not history_data[timeframe].empty:
                latest_candle = history_data[timeframe].iloc[-1]
                break
                
        if latest_candle is None:
            return []
            
        volatility = self.calculate_volatility(history_data) * self.volatility_factor
        
        bias = 0.0
        if transcendent_signal:
            if transcendent_signal["type"] == "BUY":
                bias = 0.0005 * transcendent_signal["strength"]
            elif transcendent_signal["type"] == "SELL":
                bias = -0.0005 * transcendent_signal["strength"]
        
        ghost_candles = []
        
        current_time = self.algorithm.Time
        current_close = latest_candle["Close"]
        
        for i in range(self.projection_horizon):
            price_change = current_close * (volatility * np.random.randn() + bias)
            
            ghost_open = current_close
            ghost_close = ghost_open + price_change
            
            ghost_high = max(ghost_open, ghost_close) + abs(price_change) * 0.5 * np.random.random()
            ghost_low = min(ghost_open, ghost_close) - abs(price_change) * 0.5 * np.random.random()
            
            ghost_volume = latest_candle.get("Volume", 1000) * (0.8 + 0.4 * np.random.random())
            
            ghost_time = current_time + timedelta(minutes=(i+1)*5)
            
            ghost_candle = {
                "Time": ghost_time,
                "Open": ghost_open,
                "High": ghost_high,
                "Low": ghost_low,
                "Close": ghost_close,
                "Volume": ghost_volume,
                "IsGhost": True,
                "Confidence": 1.0 - (i * 0.15)  # Confidence decreases with projection distance
            }
            
            ghost_candles.append(ghost_candle)
            
            current_close = ghost_close
        
        self.last_projections[str(symbol)] = ghost_candles
        
        return ghost_candles
    
    def calculate_projection_accuracy(self, symbol, actual_candle):
        """
        Calculate the accuracy of previous ghost candle projections.
        
        Parameters:
        - symbol: Trading symbol
        - actual_candle: The actual candle that materialized
        
        Returns:
        - Accuracy score (0.0 to 1.0)
        """
        symbol_str = str(symbol)
        
        if symbol_str not in self.last_projections or not self.last_projections[symbol_str]:
            return 0.0
            
        projected = self.last_projections[symbol_str][0]
        
        projected_close = projected["Close"]
        actual_close = actual_candle["Close"]
        
        price_diff_pct = abs(projected_close - actual_close) / actual_close
        
        accuracy = max(0.0, 1.0 - (price_diff_pct * 20))  # 5% difference = 0.0 accuracy
        
        return accuracy
