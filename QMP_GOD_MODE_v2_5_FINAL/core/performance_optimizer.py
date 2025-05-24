import numpy as np
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func):
        return func

if NUMBA_AVAILABLE:
    @njit
    def fast_volatility_calc(prices):
        """Numba-accelerated volatility calculation"""
        returns = np.empty(len(prices) - 1)
        for i in range(1, len(prices)):
            returns[i-1] = (prices[i] - prices[i-1]) / prices[i-1]
        return np.std(returns) * np.sqrt(252 * 1440)
        
    @njit  
    def fast_signal_processing(prices, volumes):
        """Numba-accelerated signal processing"""
        signals = np.empty(len(prices))
        for i in range(1, len(prices)):
            price_momentum = prices[i] > prices[i-1]
            volume_confirmation = volumes[i] > volumes[i-1] if i > 0 else True
            signals[i] = 1.0 if (price_momentum and volume_confirmation) else 0.0
        return signals
else:
    def fast_volatility_calc(prices):
        """Fallback volatility calculation"""
        returns = np.diff(prices) / prices[:-1] 
        return np.std(returns) * np.sqrt(252 * 1440)
        
    def fast_signal_processing(prices, volumes):
        """Fallback signal processing"""
        signals = np.zeros(len(prices))
        for i in range(1, len(prices)):
            price_momentum = prices[i] > prices[i-1]
            volume_confirmation = volumes[i] > volumes[i-1] if i > 0 else True
            signals[i] = 1.0 if (price_momentum and volume_confirmation) else 0.0
        return signals

class PerformanceOptimizer:
    def __init__(self):
        self.numba_enabled = NUMBA_AVAILABLE
        
    def optimize_data_processing(self, history_data):
        """Optimize data processing using numba where possible"""
        if '1m' not in history_data or history_data['1m'].empty:
            return None
            
        df = history_data['1m']
        prices = df['Close'].values
        volumes = df['Volume'].values
        
        volatility = fast_volatility_calc(prices)
        signals = fast_signal_processing(prices, volumes)
        
        return {
            'volatility': volatility,
            'signals': signals,
            'processing_optimized': self.numba_enabled
        }
