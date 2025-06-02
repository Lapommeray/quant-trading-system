"""
Entropy Scanner Strategy for QuantConnect
Adapted from Quantum Liquidity Warper - removes qiskit dependency
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

class EntropyScanner:
    """QuantConnect-compatible Entropy Scanner (adapted from Quantum Liquidity Warper)"""
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.entropy_threshold = 0.5
        self.vix_threshold = 15.0
        self.min_data_points = 100
        
    def analyze(self, market_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Analyze market entropy and generate signals"""
        try:
            if not market_data or 'prices' not in market_data:
                return {'signal': 'NEUTRAL', 'confidence': 0.0, 'reason': 'No market data'}
                
            prices = market_data['prices']
            if len(prices) < self.min_data_points:
                return {'signal': 'NEUTRAL', 'confidence': 0.0, 'reason': 'Insufficient data'}
                
            returns = self._calculate_returns(prices)
            if len(returns) < 10:
                return {'signal': 'NEUTRAL', 'confidence': 0.0, 'reason': 'Insufficient returns'}
                
            entropy = self._calculate_shannon_entropy(returns)
            volatility = self._calculate_volatility(returns)
            vix_equivalent = volatility * 100
            
            if vix_equivalent >= self.vix_threshold:
                return {
                    'signal': 'NEUTRAL', 
                    'confidence': 0.0, 
                    'reason': f'VIX too high: {vix_equivalent:.2f}',
                    'entropy': entropy,
                    'volatility': volatility
                }
                
            if entropy < self.entropy_threshold:
                signal = 'BUY' if returns[-1] > 0 else 'SELL'
                confidence = min(0.9, (self.entropy_threshold - entropy) * 2)
                return {
                    'signal': signal, 
                    'confidence': confidence, 
                    'entropy': entropy,
                    'volatility': volatility,
                    'reason': f'Low entropy detected: {entropy:.3f}'
                }
            
            return {
                'signal': 'NEUTRAL', 
                'confidence': 0.4,
                'entropy': entropy,
                'volatility': volatility,
                'reason': f'Entropy within normal range: {entropy:.3f}'
            }
            
        except Exception as e:
            self.algorithm.Error(f"EntropyScanner error: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0.0, 'reason': f'Error: {e}'}
    
    def _calculate_returns(self, prices):
        """Calculate log returns from price series"""
        prices_array = np.array(prices)
        returns = np.log(prices_array[1:] / prices_array[:-1])
        return returns[~np.isnan(returns)]
    
    def _calculate_shannon_entropy(self, returns, bins=5):
        """Calculate Shannon entropy of returns distribution"""
        if len(returns) == 0:
            return 1.0
            
        hist, _ = np.histogram(returns, bins=bins)
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        
        if len(hist) == 0:
            return 1.0
            
        entropy = -np.sum(hist * np.log(hist))
        return entropy
    
    def _calculate_volatility(self, returns):
        """Calculate volatility using existing pattern"""
        if len(returns) == 0:
            return 0.001
        return max(float(np.std(returns)), 0.0005)
    
    def generate_signal(self, market_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Generate signal for AI consensus integration"""
        return self.analyze(market_data, symbol)
