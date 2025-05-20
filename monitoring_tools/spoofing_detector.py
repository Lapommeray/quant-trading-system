"""
Spoofing Detector

Python wrapper for the Pine Script module that detects high-frequency trading
spoofing patterns using FlowAlgo data. This module identifies potential market
manipulation that could affect trading decisions.

Original Pine Script implementation uses:
- FlowAlgo data via webhooks
- Technical analysis for volume pattern detection
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class SpoofingDetector:
    def __init__(self, algorithm):
        self.algo = algorithm
        self.last_check_time = None
        self.check_interval = timedelta(minutes=1)  # Check every minute
        self.api_key = self.algo.GetParameter("FLOWALGO_API_KEY", "")
        
        self.volume_ratio_threshold = 2.0
        self.cluster_threshold = 0.7
        
        self.detection_history = []
        self.max_history_size = 100
        
        self.algo.Debug("SpoofingDetector initialized")
    
    def detect(self, symbol, history_data):
        """
        Detect spoofing patterns in market data
        
        Parameters:
        - symbol: Trading symbol
        - history_data: Dictionary of DataFrames for different timeframes
        
        Returns:
        - Dictionary with detection results
        """
        current_time = self.algo.Time
        
        if self.last_check_time is not None and current_time - self.last_check_time < self.check_interval:
            return None
            
        self.last_check_time = current_time
        
        if '1m' not in history_data or history_data['1m'].empty:
            return None
            
        recent_data = history_data['1m'].tail(30)
        
        if len(recent_data) < 20:
            return None
            
        volumes = recent_data['Volume'].values
        avg_volume = np.mean(volumes[:-1])  # Average excluding current bar
        current_volume = volumes[-1]
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        flowalgo_score = self._simulate_flowalgo_data(symbol, current_time)
        
        spoofing_detected = (flowalgo_score > self.cluster_threshold and 
                            volume_ratio > self.volume_ratio_threshold)
        
        if spoofing_detected:
            confidence = min(flowalgo_score * volume_ratio / 4, 1.0)
        else:
            confidence = 0.0
        
        self.detection_history.append({
            'time': current_time,
            'symbol': str(symbol),
            'spoofing_detected': spoofing_detected,
            'volume_ratio': volume_ratio,
            'flowalgo_score': flowalgo_score,
            'confidence': confidence
        })
        
        if len(self.detection_history) > self.max_history_size:
            self.detection_history = self.detection_history[-self.max_history_size:]
        
        if spoofing_detected:
            self.algo.Debug(f"SpoofingDetector: Potential spoofing detected for {symbol}! " +
                          f"Volume ratio: {volume_ratio:.2f}, FlowAlgo score: {flowalgo_score:.2f}")
        
        return {
            'spoofing_detected': spoofing_detected,
            'volume_ratio': volume_ratio,
            'flowalgo_score': flowalgo_score,
            'confidence': confidence,
            'direction': 'SELL' if spoofing_detected else None  # Spoofing usually indicates a reversal
        }
    
    def _simulate_flowalgo_data(self, symbol, current_time):
        """
        Simulate FlowAlgo data for testing
        In production, this would be replaced with actual API calls
        
        Parameters:
        - symbol: Trading symbol
        - current_time: Current algorithm time
        
        Returns:
        - Simulated FlowAlgo score
        """
        hour = current_time.hour
        minute = current_time.minute
        second = current_time.second
        
        symbol_hash = sum(ord(c) for c in str(symbol)) % 100
        
        if 9 <= hour < 10 or 15 <= hour < 16:  # Market open/close hours
            base_score = 0.5  # Higher baseline during volatile hours
        else:
            base_score = 0.3
            
        time_factor = (minute % 15) / 15  # Cycles every 15 minutes
        
        random_factor = np.random.normal(0, 0.2)
        
        if np.random.random() < 0.05:  # 5% chance of spoofing
            spike_factor = np.random.uniform(0.3, 0.6)
        else:
            spike_factor = 0.0
            
        score = base_score + (symbol_hash / 200) + time_factor * 0.2 + random_factor + spike_factor
        
        return max(0, min(score, 1))
    
    def get_recent_patterns(self):
        """
        Analyze recent spoofing patterns
        
        Returns:
        - Dictionary with pattern analysis
        """
        if len(self.detection_history) < 10:
            return {
                'pattern_detected': False,
                'confidence': 0.0
            }
            
        recent = self.detection_history[-10:]
        
        spoofing_count = sum(1 for entry in recent if entry['spoofing_detected'])
        
        avg_confidence = np.mean([entry['confidence'] for entry in recent])
        
        frequent_spoofing = spoofing_count >= 3  # 30% or more of recent bars show spoofing
        
        return {
            'pattern_detected': frequent_spoofing,
            'spoofing_frequency': spoofing_count / len(recent),
            'avg_confidence': avg_confidence,
            'confidence': avg_confidence if frequent_spoofing else 0.0
        }
