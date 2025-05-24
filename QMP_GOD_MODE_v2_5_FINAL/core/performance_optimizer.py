import numpy as np
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func
    prange = range

if NUMBA_AVAILABLE:
    @njit(fastmath=True, cache=True)
    def fast_volatility_calc(prices, out_array=None):
        """Memory-efficient Numba-accelerated volatility calculation"""
        n = len(prices) - 1
        if out_array is None or len(out_array) < n:
            returns = np.empty(n, dtype=np.float64)
        else:
            returns = out_array[:n]
            
        for i in range(1, len(prices)):
            returns[i-1] = (prices[i] - prices[i-1]) / prices[i-1]
            
        return np.std(returns) * np.sqrt(252 * 1440)
        
    @njit(fastmath=True, parallel=True, cache=True)
    def fast_signal_processing(prices, volumes, out_array=None):
        """Memory-efficient Numba-accelerated signal processing with parallelization"""
        n = len(prices)
        if out_array is None or len(out_array) != n:
            signals = np.zeros(n, dtype=np.float64)
        else:
            signals = out_array
            signals.fill(0.0)
            
        signals[0] = 0.0  # Initialize first element
        
        if n > 1000:
            for i in prange(1, n):
                price_momentum = prices[i] > prices[i-1]
                volume_confirmation = volumes[i] > volumes[i-1]
                signals[i] = 1.0 if (price_momentum and volume_confirmation) else 0.0
        else:
            for i in range(1, n):
                price_momentum = prices[i] > prices[i-1]
                volume_confirmation = volumes[i] > volumes[i-1]
                signals[i] = 1.0 if (price_momentum and volume_confirmation) else 0.0
                
        return signals
        
    dummy_prices = np.array([100.0, 101.0, 102.0, 101.5, 103.0], dtype=np.float64)
    dummy_volumes = np.array([1000.0, 1100.0, 900.0, 1200.0, 1300.0], dtype=np.float64)
    _ = fast_volatility_calc(dummy_prices)
    _ = fast_signal_processing(dummy_prices, dummy_volumes)
else:
    def fast_volatility_calc(prices, out_array=None):
        """Fallback volatility calculation"""
        returns = np.diff(prices) / prices[:-1] 
        return np.std(returns) * np.sqrt(252 * 1440)
        
    def fast_signal_processing(prices, volumes, out_array=None):
        """Fallback signal processing"""
        n = len(prices)
        if out_array is None or len(out_array) != n:
            signals = np.zeros(n, dtype=np.float64)
        else:
            signals = out_array
            signals.fill(0.0)
            
        for i in range(1, n):
            price_momentum = prices[i] > prices[i-1]
            volume_confirmation = volumes[i] > volumes[i-1]
            signals[i] = 1.0 if (price_momentum and volume_confirmation) else 0.0
            
        return signals

class PerformanceOptimizer:
    def __init__(self):
        self.numba_enabled = NUMBA_AVAILABLE
        self._returns_buffer = np.empty(10000, dtype=np.float64)  # Pre-allocate buffer
        self._signals_buffer = np.empty(10000, dtype=np.float64)  # Pre-allocate buffer
        
    def optimize_data_processing(self, history_data):
        """Optimize data processing using numba where possible"""
        if '1m' not in history_data or history_data['1m'].empty:
            return None
            
        df = history_data['1m']
        prices = df['Close'].to_numpy(copy=True)  # Zero-copy if possible
        volumes = df['Volume'].to_numpy(copy=True)
        
        if len(prices) > len(self._returns_buffer):
            self._returns_buffer = np.empty(len(prices), dtype=np.float64)
        if len(prices) > len(self._signals_buffer):
            self._signals_buffer = np.empty(len(prices), dtype=np.float64)
        
        volatility = fast_volatility_calc(prices, self._returns_buffer)
        signals = fast_signal_processing(prices, volumes, self._signals_buffer)
        
        return {
            'volatility': volatility,
            'signals': signals[:len(prices)],  # Return only the valid portion
            'processing_optimized': self.numba_enabled
        }
