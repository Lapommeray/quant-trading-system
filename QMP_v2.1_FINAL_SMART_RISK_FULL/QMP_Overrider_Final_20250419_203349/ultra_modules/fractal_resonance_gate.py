# Detects fractal harmonic time alignments across hidden cycles.

import numpy as np
from datetime import datetime, timedelta

class FractalResonanceGate:
    def __init__(self, algorithm):
        self.algo = algorithm
        self.lookback_period = 50  # Number of candles to analyze
        self.resonance_threshold = 0.65
        self.fibonacci_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618]
        
    def decode(self, symbol, history_window):
        """
        Analyzes price action for fractal patterns and time-based resonance.
        
        Parameters:
        - symbol: The trading symbol
        - history_window: List of TradeBars
        
        Returns:
        - True if fractal resonance is detected, False otherwise
        """
        if len(history_window) < self.lookback_period:
            self.algo.Debug(f"FractalResonance: Insufficient history for {symbol}")
            return False
            
        highs = np.array([bar.High for bar in history_window])
        lows = np.array([bar.Low for bar in history_window])
        closes = np.array([bar.Close for bar in history_window])
        timestamps = [bar.EndTime for bar in history_window]
        
        swing_highs = self._find_swing_points(highs, 'high')
        swing_lows = self._find_swing_points(lows, 'low')
        
        price_resonance = self._check_price_fibonacci(highs, lows, closes, swing_highs, swing_lows)
        
        time_resonance = self._check_time_resonance(timestamps, swing_highs, swing_lows)
        
        resonance_score = 0.6 * price_resonance + 0.4 * time_resonance
        is_resonant = resonance_score > self.resonance_threshold
        
        self.algo.Debug(f"FractalResonance: {symbol} - Score: {resonance_score:.2f}, Resonant: {is_resonant}")
        return is_resonant
    
    def _find_swing_points(self, data, point_type, window=5):
        """Identify swing highs or swing lows in the data"""
        result = []
        
        if point_type == 'high':
            for i in range(window, len(data) - window):
                if all(data[i] > data[i-j] for j in range(1, window+1)) and \
                   all(data[i] > data[i+j] for j in range(1, window+1)):
                    result.append((i, data[i]))
        else:
            for i in range(window, len(data) - window):
                if all(data[i] < data[i-j] for j in range(1, window+1)) and \
                   all(data[i] < data[i+j] for j in range(1, window+1)):
                    result.append((i, data[i]))
                    
        return result
    
    def _check_price_fibonacci(self, highs, lows, closes, swing_highs, swing_lows):
        """Check if price movements align with Fibonacci ratios"""
        if not swing_highs or not swing_lows:
            return 0.0
            
        recent_swings = sorted(swing_highs + swing_lows, key=lambda x: x[0])[-5:]
        if len(recent_swings) < 2:
            return 0.0
            
        movements = []
        for i in range(len(recent_swings) - 1):
            start_idx, start_price = recent_swings[i]
            end_idx, end_price = recent_swings[i + 1]
            movements.append(abs(end_price - start_price))
        
        fib_matches = 0
        for i in range(len(movements)):
            for j in range(i + 1, len(movements)):
                ratio = movements[i] / movements[j] if movements[j] != 0 else 0
                for fib in self.fibonacci_levels:
                    if abs(ratio - fib) < 0.05:  # 5% tolerance
                        fib_matches += 1
        
        max_possible_matches = len(movements) * (len(movements) - 1) / 2 * len(self.fibonacci_levels)
        return min(fib_matches / max_possible_matches if max_possible_matches > 0 else 0, 1.0)
    
    def _check_time_resonance(self, timestamps, swing_highs, swing_lows):
        """Check if time intervals between swing points align with Fibonacci ratios"""
        if not swing_highs or not swing_lows:
            return 0.0
            
        all_swings = [(idx, ts) for idx, _ in swing_highs + swing_lows for ts in [timestamps[idx]]]
        all_swings.sort(key=lambda x: x[0])
        
        if len(all_swings) < 3:
            return 0.0
            
        intervals = []
        for i in range(len(all_swings) - 1):
            _, start_time = all_swings[i]
            _, end_time = all_swings[i + 1]
            interval_minutes = (end_time - start_time).total_seconds() / 60
            intervals.append(interval_minutes)
        
        time_matches = 0
        for i in range(len(intervals)):
            for j in range(i + 1, len(intervals)):
                ratio = intervals[i] / intervals[j] if intervals[j] != 0 else 0
                for fib in self.fibonacci_levels:
                    if abs(ratio - fib) < 0.1:  # 10% tolerance for time
                        time_matches += 1
        
        max_possible_matches = len(intervals) * (len(intervals) - 1) / 2 * len(self.fibonacci_levels)
        return min(time_matches / max_possible_matches if max_possible_matches > 0 else 0, 1.0)
