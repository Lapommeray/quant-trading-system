import numpy as np
from scipy.stats import skew, kurtosis

def adjusted_var(returns, alpha=0.05):
    """
    Calculate Value at Risk with skewness and kurtosis adjustments
    """
    z_score = np.percentile(returns, alpha * 100)
    skewness = skew(returns)
    excess_kurtosis = kurtosis(returns) - 3
    
    z_cf = z_score + (skewness / 6) * (z_score**2 - 1) + \
           (excess_kurtosis / 24) * (z_score**3 - 3*z_score) - \
           (skewness**2 / 36) * (2*z_score**3 - 5*z_score)
    
    return -z_cf * np.std(returns)

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown from returns series"""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)

def risk_parity_weights(cov_matrix):
    """Calculate risk parity portfolio weights"""
    inv_vol = 1 / np.sqrt(np.diag(cov_matrix))
    weights = inv_vol / np.sum(inv_vol)
    return weights
