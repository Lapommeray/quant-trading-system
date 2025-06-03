"""
QC-compatible entropy scanner strategy (replaces quantum_liquidity_warper)
"""
import numpy as np
import pandas as pd
from typing import Dict, Any

class EntropyScanner:
    """QuantConnect-compatible entropy-based market scanner."""
    
    def __init__(self, algorithm=None):
        self.algorithm = algorithm
        self.symbol = None
        self.entropy_threshold = 0.5
        self.volume_threshold = 10000
        
    def OnData(self, data):
        """Process market data for entropy-based signals."""
        if not self.symbol or self.symbol not in data:
            return
            
        try:
            closes = self.algorithm.History(self.symbol, 100, 1).close
            if len(closes) < 50:
                return
                
            returns = np.log(closes / closes.shift(1)).dropna()
            if len(returns) < 10:
                return
                
            entropy = self._calculate_entropy(returns)
            
            current_volume = data[self.symbol].Volume
            
            if entropy < self.entropy_threshold and current_volume > self.volume_threshold:
                signal_direction = 1 if returns.iloc[-1] > 0 else -1
                self.algorithm.MarketOrder(self.symbol, signal_direction)
                
        except Exception as e:
            self.algorithm.Log(f"EntropyScanner error: {e}")
    
    def _calculate_entropy(self, returns: pd.Series, bins: int = 5) -> float:
        """Calculate Shannon entropy of returns."""
        try:
            hist, _ = np.histogram(returns, bins=bins)
            hist = hist[hist > 0]  # Remove zero bins
            if len(hist) == 0:
                return 1.0
            probs = hist / hist.sum()
            return float(-np.sum(probs * np.log(probs + 1e-6)))
        except Exception:
            return 1.0
    
    def SetSymbol(self, symbol):
        """Set the trading symbol."""
        self.symbol = symbol
    
    def SetParameters(self, entropy_threshold: float = 0.5, volume_threshold: int = 10000):
        """Configure scanner parameters."""
        self.entropy_threshold = entropy_threshold
        self.volume_threshold = volume_threshold
