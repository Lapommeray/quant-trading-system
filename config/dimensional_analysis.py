import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging

def analyze_dimensions(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates statistical dimensions of market data.
    
    Args:
        data: Market data DataFrame with OHLCV columns
        
    Returns:
        Dictionary containing dimensional analysis metrics
    """
    if data.empty:
        return {
            'volatility': 0.0,
            'kurtosis': 0.0,
            'sharpe': 0.0,
            'skewness': 0.0,
            'entropy': 0.0
        }
    
    close_prices = data.get('Close', data.iloc[:, 0] if len(data.columns) > 0 else pd.Series([]))
    
    if close_prices.empty or len(close_prices) < 2:
        return {
            'volatility': 0.0,
            'kurtosis': 0.0,
            'sharpe': 0.0,
            'skewness': 0.0,
            'entropy': 0.0
        }
    
    returns = close_prices.pct_change().dropna()
    
    if returns.empty or returns.std() == 0:
        return {
            'volatility': 0.0,
            'kurtosis': 0.0,
            'sharpe': 0.0,
            'skewness': 0.0,
            'entropy': 0.0
        }
    
    return {
        'volatility': returns.std() if not pd.isna(returns.std()) else 0.0,
        'kurtosis': returns.kurtosis() if not pd.isna(returns.kurtosis()) else 0.0,
        'sharpe': (returns.mean() / returns.std()) if returns.std() != 0 and not pd.isna(returns.std()) else 0.0,
        'skewness': returns.skew() if not pd.isna(returns.skew()) else 0.0,
        'entropy': calculate_entropy(returns)
    }

def calculate_entropy(series: pd.Series, bins: int = 10) -> float:
    """Calculate Shannon entropy of a time series."""
    try:
        hist, _ = np.histogram(series, bins=bins)
        hist = hist[hist > 0]  # Remove zero bins
        if len(hist) == 0:
            return 0.0
        probs = hist / hist.sum()
        return float(-np.sum(probs * np.log2(probs)))
    except Exception:
        return 0.0

def calculate_fractal_dimension(data: pd.Series, max_k: int = 10) -> float:
    """Calculate fractal dimension using box-counting method."""
    try:
        if len(data) < 4:
            return 1.0
        
        normalized = (data - data.min()) / (data.max() - data.min()) if data.max() != data.min() else data
        
        scales = []
        counts = []
        
        for k in range(1, min(max_k, len(data) // 4)):
            box_size = 1.0 / k
            count = 0
            
            for i in range(k):
                box_min = i * box_size
                box_max = (i + 1) * box_size
                if any((normalized >= box_min) & (normalized < box_max)):
                    count += 1
            
            if count > 0:
                scales.append(np.log(1.0 / box_size))
                counts.append(np.log(count))
        
        if len(scales) < 2:
            return 1.0
        
        coeffs = np.polyfit(scales, counts, 1)
        return float(abs(coeffs[0]))
    
    except Exception:
        return 1.0

class DimensionalAnalyzer:
    """Advanced dimensional analysis for quantum trading systems."""
    
    def __init__(self, algorithm=None):
        self.algorithm = algorithm
        self.logger = logging.getLogger("DimensionalAnalyzer")
        self.cache = {}
        
    def analyze_market_dimensions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive dimensional analysis of market data."""
        try:
            basic_dims = analyze_dimensions(data)
            
            if 'Close' in data.columns and len(data) > 10:
                fractal_dim = calculate_fractal_dimension(data['Close'])
                basic_dims['fractal_dimension'] = fractal_dim
            
            return basic_dims
            
        except Exception as e:
            self.logger.error(f"Dimensional analysis failed: {e}")
            return {
                'volatility': 0.0,
                'kurtosis': 0.0,
                'sharpe': 0.0,
                'skewness': 0.0,
                'entropy': 0.0,
                'fractal_dimension': 1.0
            }
    
    def quantum_dimension_analysis(self, data: pd.DataFrame) -> Dict[str, float]:
        """Quantum-enhanced dimensional analysis."""
        base_analysis = self.analyze_market_dimensions(data)
        
        if len(data) > 20:
            try:
                close_prices = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]
                returns = close_prices.pct_change().dropna()
                
                coherence = self._calculate_quantum_coherence(returns)
                base_analysis['quantum_coherence'] = coherence
                
                stability = self._calculate_dimensional_stability(returns)
                base_analysis['dimensional_stability'] = stability
                
            except Exception as e:
                self.logger.warning(f"Quantum analysis failed: {e}")
                base_analysis['quantum_coherence'] = 0.5
                base_analysis['dimensional_stability'] = 0.5
        
        return base_analysis
    
    def _calculate_quantum_coherence(self, returns: pd.Series) -> float:
        """Calculate quantum coherence metric."""
        try:
            if len(returns) < 10:
                return 0.5
            
            autocorr = returns.autocorr(lag=1)
            return float(abs(autocorr)) if not np.isnan(autocorr) else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_dimensional_stability(self, returns: pd.Series) -> float:
        """Calculate dimensional stability metric."""
        try:
            if len(returns) < 20:
                return 0.5
            
            rolling_vol = returns.rolling(window=10).std()
            vol_stability = 1.0 - rolling_vol.std() / rolling_vol.mean() if rolling_vol.mean() != 0 else 0.5
            
            return float(max(0.0, min(1.0, vol_stability)))
            
        except Exception:
            return 0.5
