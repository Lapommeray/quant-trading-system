"""
Advanced Cointegration Analysis for Institutional Trading

Provides sophisticated cointegration testing methods including Johansen tests
and kernel-based approaches for multi-asset arbitrage strategies.
"""

import numpy as np
import pandas as pd
import logging
import warnings
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

try:
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Johansen test will not be available.")

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Kernel cointegration will not be available.")

try:
    from statsmodels.tsa.stattools import adfuller
    ADFULLER_AVAILABLE = True
except ImportError:
    ADFULLER_AVAILABLE = False
    warnings.warn("statsmodels.tsa.stattools not available. ADF test will not be available.")

class AdvancedCointegration:
    """
    Advanced cointegration analysis with institutional-grade methods.
    """
    
    def __init__(self, lookback=252, confidence_level=0.95):
        self.lookback = lookback
        self.confidence_level = confidence_level
        self.logger = logging.getLogger('AdvancedCointegration')
        
    def johansen_test(self, prices_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform Johansen cointegration test for multiple time series.
        
        Parameters:
        - prices_matrix: DataFrame with price series as columns
        
        Returns:
        - Dictionary with test results and eigenvectors
        """
        if not STATSMODELS_AVAILABLE:
            self.logger.error("statsmodels not available. Cannot perform Johansen test.")
            return {'error': "statsmodels not available", 'cointegrated': False}
            
        if prices_matrix.shape[1] < 2:
            raise ValueError("Need at least 2 price series for cointegration test")
            
        recent_prices = prices_matrix.iloc[-self.lookback:] if len(prices_matrix) > self.lookback else prices_matrix
        
        try:
            result = coint_johansen(recent_prices.values, det_order=0, k_ar_diff=1)
            
            hedge_ratios = result.evec[:, 0]  # First eigenvector
            
            return {
                'hedge_ratios': hedge_ratios,
                'eigenvalues': result.eig,
                'trace_stats': result.lr1,
                'max_eigen_stats': result.lr2,
                'crit_values_trace': result.cvt,
                'crit_values_max_eigen': result.cvm,
                'cointegrated': result.lr1[0] > result.cvt[0, 1],  # 95% confidence
                'symbols': list(prices_matrix.columns),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Johansen test failed: {str(e)}")
            return {'error': str(e), 'cointegrated': False}
    
    def kernel_cointegration(self, x: pd.Series, y: pd.Series) -> Dict[str, Any]:
        """
        Kernel-based cointegration test using Gaussian Process.
        
        Parameters:
        - x, y: Price series to test
        
        Returns:
        - Dictionary with test results
        """
        if not SKLEARN_AVAILABLE or not ADFULLER_AVAILABLE:
            self.logger.error("scikit-learn or statsmodels not available. Cannot perform kernel cointegration.")
            return {'error': "Required dependencies not available", 'cointegrated': False}
            
        try:
            x_vals = x.values.reshape(-1, 1)
            y_vals = y.values
            
            gp = GaussianProcessRegressor()
            gp.fit(x_vals, y_vals)
            
            y_pred = gp.predict(x_vals)
            residuals = y_vals - y_pred
            
            adf_result = adfuller(residuals)
            
            return {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'cointegrated': adf_result[1] < (1 - self.confidence_level),
                'residuals': residuals,
                'gp_score': gp.score(x_vals, y_vals),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Kernel cointegration test failed: {str(e)}")
            return {'error': str(e), 'cointegrated': False}
    
    def calculate_hedge_ratio(self, x: pd.Series, y: pd.Series, method='ols') -> Dict[str, Any]:
        """
        Calculate hedge ratio between two price series.
        
        Parameters:
        - x, y: Price series
        - method: Method to use ('ols', 'robust', 'rolling')
        
        Returns:
        - Dictionary with hedge ratio and statistics
        """
        try:
            if method == 'ols':
                X = np.vstack([np.ones(len(x)), x.values]).T
                beta = np.linalg.lstsq(X, y.values, rcond=None)[0]
                
                residuals = y.values - (beta[0] + beta[1] * x.values)
                
                adf_result = None
                if ADFULLER_AVAILABLE:
                    adf_result = adfuller(residuals)
                
                return {
                    'hedge_ratio': beta[1],
                    'intercept': beta[0],
                    'residuals': residuals,
                    'adf_statistic': adf_result[0] if adf_result else None,
                    'p_value': adf_result[1] if adf_result else None,
                    'cointegrated': adf_result[1] < (1 - self.confidence_level) if adf_result else None,
                    'timestamp': datetime.now().isoformat()
                }
            elif method == 'robust':
                ratios = y.values / x.values
                hedge_ratio = np.median(ratios)
                
                residuals = y.values - hedge_ratio * x.values
                
                adf_result = None
                if ADFULLER_AVAILABLE:
                    adf_result = adfuller(residuals)
                
                return {
                    'hedge_ratio': hedge_ratio,
                    'intercept': 0,
                    'residuals': residuals,
                    'adf_statistic': adf_result[0] if adf_result else None,
                    'p_value': adf_result[1] if adf_result else None,
                    'cointegrated': adf_result[1] < (1 - self.confidence_level) if adf_result else None,
                    'timestamp': datetime.now().isoformat()
                }
            elif method == 'rolling':
                window = min(60, len(x) // 4)  # Use 60 days or 1/4 of data
                
                hedge_ratios = []
                for i in range(window, len(x)):
                    x_window = x.iloc[i-window:i]
                    y_window = y.iloc[i-window:i]
                    X = np.vstack([np.ones(len(x_window)), x_window.values]).T
                    beta = np.linalg.lstsq(X, y_window.values, rcond=None)[0]
                    hedge_ratios.append(beta[1])
                
                current_ratio = hedge_ratios[-1]
                
                residuals = y.values - (current_ratio * x.values)
                
                adf_result = None
                if ADFULLER_AVAILABLE:
                    adf_result = adfuller(residuals)
                
                return {
                    'hedge_ratio': current_ratio,
                    'hedge_ratio_history': hedge_ratios,
                    'residuals': residuals,
                    'adf_statistic': adf_result[0] if adf_result else None,
                    'p_value': adf_result[1] if adf_result else None,
                    'cointegrated': adf_result[1] < (1 - self.confidence_level) if adf_result else None,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                raise ValueError(f"Unknown method: {method}")
        except Exception as e:
            self.logger.error(f"Hedge ratio calculation failed: {str(e)}")
            return {'error': str(e)}
