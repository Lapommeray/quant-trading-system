"""
BTC Off-Chain Monitor

Python wrapper for the JavaScript module that tracks off-chain Bitcoin movements
using the Glassnode API. This module detects large Bitcoin transfers that may
indicate significant market movements.

Original JS implementation uses:
- glassnode-api for data access
- ccxt for exchange interactions
"""

from AlgorithmImports import *
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class BTCOffchainMonitor:
    def __init__(self, algorithm):
        self.algo = algorithm
        self.alert_threshold = 10000  # BTC
        self.last_check_time = None
        self.check_interval = timedelta(minutes=5)
        self.api_key = self.algo.GetParameter("GLASSNODE_API_KEY", "")
        
        self.transfer_history = []
        self.max_history_size = 24  # Keep 24 hours of data
        
        self.algo.Debug("BTCOffchainMonitor initialized")
    
    def check_transfers(self, current_time):
        """
        Check for large off-chain Bitcoin transfers
        
        Parameters:
        - current_time: Current algorithm time
        
        Returns:
        - Dictionary with detection results
        """
        if self.last_check_time is not None and current_time - self.last_check_time < self.check_interval:
            return None
            
        self.last_check_time = current_time
        
        transfer_amount = self._simulate_transfer_data(current_time)
        
        self.transfer_history.append({
            'time': current_time,
            'amount': transfer_amount
        })
        
        if len(self.transfer_history) > self.max_history_size:
            self.transfer_history = self.transfer_history[-self.max_history_size:]
        
        alert_triggered = transfer_amount > self.alert_threshold
        
        if alert_triggered:
            self.algo.Debug(f"BTCOffchainMonitor: Large transfer detected! {transfer_amount} BTC")
        
        return {
            'large_transfer_detected': alert_triggered,
            'transfer_amount': transfer_amount,
            'confidence': min(transfer_amount / (self.alert_threshold * 2), 1.0) if alert_triggered else 0.0,
            'direction': 'BUY' if alert_triggered else None
        }
    
    def _simulate_transfer_data(self, current_time):
        """
        Simulate off-chain transfer data for testing
        In production, this would be replaced with actual API calls
        
        Parameters:
        - current_time: Current algorithm time
        
        Returns:
        - Simulated transfer amount
        """
        hour = current_time.hour
        minute = current_time.minute
        day = current_time.day
        
        base_amount = 1000 + (hour * 100) + (minute * 10)
        
        random_factor = np.random.normal(1.0, 0.3)
        
        if np.random.random() < 0.05:  # 5% chance of large transfer
            spike_factor = np.random.uniform(10, 20)
        else:
            spike_factor = 1.0
            
        transfer_amount = base_amount * random_factor * spike_factor
        
        return transfer_amount
    
    def get_historical_pattern(self):
        """
        Analyze historical transfer patterns
        
        Returns:
        - Dictionary with pattern analysis
        """
        if len(self.transfer_history) < 6:
            return {
                'pattern_detected': False,
                'confidence': 0.0
            }
            
        amounts = [entry['amount'] for entry in self.transfer_history]
        
        trend = np.polyfit(range(len(amounts)), amounts, 1)[0]
        
        volatility = np.std(amounts) / np.mean(amounts) if np.mean(amounts) > 0 else 0
        
        increasing_pattern = trend > 500  # Significant upward trend
        high_volatility = volatility > 0.5
        
        return {
            'pattern_detected': increasing_pattern or high_volatility,
            'increasing_trend': increasing_pattern,
            'high_volatility': high_volatility,
            'confidence': min(abs(trend) / 1000 + volatility, 1.0)
        }
