"""
Timeline Warp Plot

Implements visualization of alternative timeline paths for the QMP Overrider system,
allowing traders to see multiple possible future scenarios simultaneously.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class TimelineWarpPlot:
    """
    Generates alternative timeline visualizations based on quantum path analysis
    and transcendent intelligence signals.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the timeline warp plot generator.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.timeline_count = 5  # Number of alternative timelines to generate
        self.horizon_length = 10  # Number of candles in each timeline
        self.last_timelines = {}
    
    def generate_timelines(self, symbol, history_data, transcendent_signal=None):
        """
        Generate alternative timeline paths based on current market conditions.
        
        Parameters:
        - symbol: Trading symbol
        - history_data: Dictionary of DataFrames for different timeframes
        - transcendent_signal: Signal from transcendent intelligence
        
        Returns:
        - Dictionary containing timeline data
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
        
        consciousness_level = 0.5
        if transcendent_signal and "consciousness_level" in transcendent_signal:
            consciousness_level = transcendent_signal["consciousness_level"]
        
        timelines = []
        
        primary_timeline = self._generate_single_timeline(
            symbol, 
            latest_candle["Close"], 
            volatility,
            bias=0.0002 if transcendent_signal and transcendent_signal["type"] == "BUY" else -0.0002,
            confidence=0.8
        )
        
        timelines.append({
            "id": "primary",
            "name": "Primary Timeline",
            "data": primary_timeline,
            "confidence": 0.8,
            "color": "#4CAF50"  # Green
        })
        
        for i in range(self.timeline_count - 1):
            alt_bias = 0.0004 * (np.random.random() - 0.5)
            alt_volatility = volatility * (0.8 + 0.4 * np.random.random())
            
            alt_confidence = 0.3 + (0.3 * consciousness_level * np.random.random())
            
            alt_timeline = self._generate_single_timeline(
                symbol,
                latest_candle["Close"],
                alt_volatility,
                bias=alt_bias,
                confidence=alt_confidence
            )
            
            timelines.append({
                "id": f"alt_{i+1}",
                "name": f"Alternative {i+1}",
                "data": alt_timeline,
                "confidence": alt_confidence,
                "color": self._get_timeline_color(i)
            })
        
        self.last_timelines[str(symbol)] = timelines
        
        return {
            "success": True,
            "symbol": str(symbol),
            "timestamp": self.algorithm.Time,
            "latest_price": latest_candle["Close"],
            "timelines": timelines,
            "consciousness_level": consciousness_level
        }
    
    def _generate_single_timeline(self, symbol, start_price, volatility, bias=0.0, confidence=0.5):
        """
        Generate a single timeline path.
        
        Parameters:
        - symbol: Trading symbol
        - start_price: Starting price for the timeline
        - volatility: Price volatility
        - bias: Directional bias
        - confidence: Confidence in this timeline
        
        Returns:
        - List of price points for the timeline
        """
        timeline_data = []
        current_time = self.algorithm.Time
        current_price = start_price
        
        for i in range(self.horizon_length):
            step_volatility = volatility * (1 + i * 0.1)
            
            price_change = current_price * (step_volatility * np.random.randn() + bias)
            next_price = current_price + price_change
            
            price_point = {
                "time": current_time + timedelta(minutes=(i+1)*5),
                "price": next_price,
                "confidence": confidence * (1.0 - (i * 0.05))  # Confidence decreases with time
            }
            
            timeline_data.append(price_point)
            current_price = next_price
        
        return timeline_data
    
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
    
    def _get_timeline_color(self, index):
        """
        Get color for timeline visualization.
        
        Parameters:
        - index: Timeline index
        
        Returns:
        - Color hex code
        """
        colors = [
            "#2196F3",  # Blue
            "#FFC107",  # Amber
            "#9C27B0",  # Purple
            "#FF5722",  # Deep Orange
            "#607D8B",  # Blue Grey
            "#E91E63",  # Pink
            "#00BCD4",  # Cyan
            "#8BC34A",  # Light Green
            "#FF9800"   # Orange
        ]
        
        return colors[index % len(colors)]
    
    def get_convergence_analysis(self, symbol):
        """
        Analyze convergence of timeline paths to identify high-probability zones.
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Convergence analysis results
        """
        symbol_str = str(symbol)
        
        if symbol_str not in self.last_timelines:
            return None
            
        timelines = self.last_timelines[symbol_str]
        
        if not timelines:
            return None
            
        all_prices = []
        
        for timeline in timelines:
            timeline_prices = [point["price"] for point in timeline["data"]]
            all_prices.append(timeline_prices)
            
        price_array = np.array(all_prices)
        
        mean_prices = np.mean(price_array, axis=0)
        std_prices = np.std(price_array, axis=0)
        
        convergence_ratio = 1.0 / (1.0 + std_prices / mean_prices)
        
        high_convergence_indices = np.where(convergence_ratio > 0.8)[0]
        
        high_convergence_zones = []
        for idx in high_convergence_indices:
            zone = {
                "time_index": idx,
                "time": timelines[0]["data"][idx]["time"],
                "mean_price": mean_prices[idx],
                "convergence_ratio": convergence_ratio[idx],
                "direction": "bullish" if idx > 0 and mean_prices[idx] > mean_prices[idx-1] else "bearish"
            }
            high_convergence_zones.append(zone)
        
        return {
            "symbol": symbol_str,
            "convergence_ratios": convergence_ratio.tolist(),
            "mean_prices": mean_prices.tolist(),
            "std_prices": std_prices.tolist(),
            "high_convergence_zones": high_convergence_zones
        }
