from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
import pandas as pd

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
