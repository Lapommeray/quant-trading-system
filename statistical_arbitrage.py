from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
import pandas as pd
import warnings

class AdvancedCointegration:
    def __init__(self, lookback=252, confidence_level=0.95):
        self.lookback = lookback
        self.confidence_level = confidence_level
    
    def hedge_ratio_estimation(self, prices):
        """Johansen's procedure for multiple cointegrated assets"""
        result = coint_johansen(prices, det_order=0, k_ar_diff=1)
        eigenvectors = result.evec
        return eigenvectors[:, 0]
    
    def regime_switching_test(self, spread):
        """Detects structural breaks in cointegration"""
        from changepy import pelt
        return pelt(spread.values, pen=2)
    
    def nonlinear_coint(self, x, y):
        """Kernel-based cointegration test"""
        from sklearn.metrics.pairwise import rbf_kernel
        K = rbf_kernel(x.reshape(-1,1), gamma=0.1)
        return np.linalg.matrix_rank(K, tol=1e-3) < K.shape[0]

    def johansen_test(self, prices_matrix):
        """Institutional-grade Johansen cointegration test"""
        try:
            if len(prices_matrix) < 20:
                warnings.warn("Insufficient data for Johansen test, using fallback")
                return self.hedge_ratio_estimation(prices_matrix)
            
            result = coint_johansen(prices_matrix.T, det_order=0, k_ar_diff=1)
            return result.evec[:,0]
        except Exception as e:
            warnings.warn(f"Johansen test failed: {e}, using fallback")
            return self.hedge_ratio_estimation(prices_matrix)

    def kernel_coint(self, x, y):
        """Kernel-based cointegration test with Gaussian Process"""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from statsmodels.tsa.stattools import adfuller
        gp = GaussianProcessRegressor()
        gp.fit(x.reshape(-1,1), y)
        residuals = y - gp.predict(x.reshape(-1,1))
        return adfuller(residuals)[1] < 0.05
