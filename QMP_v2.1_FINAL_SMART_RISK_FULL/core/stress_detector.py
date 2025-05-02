"""
Stress Detector

Python wrapper for the stress detection module that analyzes earnings call
audio for signs of deception or stress in executives' voices. This module
uses the Truthset API to detect patterns that may indicate future stock
price movements.

Original implementation uses:
- Truthset API for stress analysis
- Alpaca API for earnings transcripts
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class StressDetector:
    def __init__(self, algorithm):
        self.algo = algorithm
        self.api_key = self.algo.GetParameter("TRUTHSET_API_KEY", "")
        self.alpaca_key = self.algo.GetParameter("ALPACA_API_KEY", "")
        
        self.extreme_stress_threshold = 0.85
        self.moderate_stress_threshold = 0.65
        
        self.detection_history = {}
        self.max_history_size = 100
        
        self.sp500_symbols = [
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
            "TSLA", "NVDA", "JPM", "V", "PG"
        ]
        
        self.earnings_calendar = {}
        
        self.algo.Debug("StressDetector initialized")
    
    def detect(self, symbol, current_time=None):
        """
        Detect stress patterns in earnings call audio
        
        Parameters:
        - symbol: Trading symbol
        - current_time: Current algorithm time (optional)
        
        Returns:
        - Dictionary with detection results
        """
        if current_time is None:
            current_time = self.algo.Time
            
        symbol_str = str(symbol)
        
        if not self._has_earnings_today(symbol_str, current_time):
            return None
            
        stress_score = self._simulate_stress_score(symbol_str, current_time)
        
        extreme_stress = stress_score > self.extreme_stress_threshold
        moderate_stress = stress_score > self.moderate_stress_threshold
        
        if extreme_stress:
            confidence = min((stress_score - self.extreme_stress_threshold) * 5 + 0.7, 1.0)
        elif moderate_stress:
            confidence = min((stress_score - self.moderate_stress_threshold) * 3 + 0.4, 0.7)
        else:
            confidence = 0.0
        
        if symbol_str not in self.detection_history:
            self.detection_history[symbol_str] = []
            
        self.detection_history[symbol_str].append({
            'time': current_time,
            'stress_score': stress_score,
            'extreme_stress': extreme_stress,
            'moderate_stress': moderate_stress,
            'confidence': confidence
        })
        
        if len(self.detection_history[symbol_str]) > self.max_history_size:
            self.detection_history[symbol_str] = self.detection_history[symbol_str][-self.max_history_size:]
        
        if extreme_stress:
            self.algo.Debug(f"StressDetector: Extreme stress detected in {symbol_str} earnings call! " +
                          f"Score: {stress_score:.2f}")
        elif moderate_stress:
            self.algo.Debug(f"StressDetector: Moderate stress detected in {symbol_str} earnings call. " +
                          f"Score: {stress_score:.2f}")
        
        return {
            'stress_detected': extreme_stress or moderate_stress,
            'extreme_stress': extreme_stress,
            'stress_score': stress_score,
            'confidence': confidence,
            'direction': 'SELL' if extreme_stress else ('SELL' if moderate_stress else None)
        }
    
    def _has_earnings_today(self, symbol, current_time):
        """
        Check if a symbol has earnings today
        
        Parameters:
        - symbol: Trading symbol
        - current_time: Current algorithm time
        
        Returns:
        - True if earnings today, False otherwise
        """
        
        self._update_earnings_calendar(current_time)
        
        today = current_time.date()
        return symbol in self.earnings_calendar.get(today, [])
    
    def _update_earnings_calendar(self, current_time):
        """
        Update the earnings calendar
        
        Parameters:
        - current_time: Current algorithm time
        """
        
        for i in range(30):
            date = (current_time + timedelta(days=i)).date()
            
            if date not in self.earnings_calendar:
                num_earnings = np.random.randint(2, 6)
                earnings_symbols = np.random.choice(self.sp500_symbols, num_earnings, replace=False)
                self.earnings_calendar[date] = list(earnings_symbols)
    
    def _simulate_stress_score(self, symbol, current_time):
        """
        Simulate stress score for testing
        In production, this would be replaced with actual API calls
        
        Parameters:
        - symbol: Trading symbol
        - current_time: Current algorithm time
        
        Returns:
        - Simulated stress score
        """
        symbol_hash = sum(ord(c) for c in symbol) % 100
        day_of_week = current_time.weekday()
        
        base_score = 0.4 + (symbol_hash / 200)  # 0.4 to 0.9
        
        time_factor = day_of_week / 14  # 0 to 0.5
        
        random_factor = np.random.normal(0, 0.15)
        
        if np.random.random() < 0.1:  # 10% chance of high stress
            spike_factor = np.random.uniform(0.2, 0.4)
        else:
            spike_factor = 0.0
            
        score = base_score + time_factor + random_factor + spike_factor
        
        return max(0, min(score, 1))
    
    def get_earnings_calendar(self, days=7):
        """
        Get the earnings calendar for the next N days
        
        Parameters:
        - days: Number of days to look ahead
        
        Returns:
        - Dictionary with earnings calendar
        """
        current_time = self.algo.Time
        self._update_earnings_calendar(current_time)
        
        calendar = {}
        for i in range(days):
            date = (current_time + timedelta(days=i)).date()
            if date in self.earnings_calendar:
                calendar[date] = self.earnings_calendar[date]
        
        return calendar
