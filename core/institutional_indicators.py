"""
Institutional Trading Indicators

Advanced indicators for institutional trading including Heston volatility,
ML-enhanced RSI, order flow imbalance, and regime detection.
"""

import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional, Dict, Any

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. HestonVolatility will have limited functionality.")

try:
    from sklearn.ensemble import GradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. ML_RSI will have limited functionality.")

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    warnings.warn("hmmlearn not available. RegimeDetector will have limited functionality.")

class HestonVolatility:
    """
    Heston stochastic volatility model implementation.
    """
    
    def __init__(self, lookback: int = 30, risk_free: float = 0.01):
        self.lookback = lookback
        self.r = risk_free
        self.logger = logging.getLogger('HestonVolatility')
        
    def heston_objective(self, params: np.ndarray, returns: np.ndarray) -> float:
        """Objective function for Heston model calibration"""
        kappa, theta, xi, rho, v0 = params
        n = len(returns)
        v = np.zeros(n)
        v[0] = v0
        ll = 0
        
        for t in range(1, n):
            v[t] = np.abs(v[t-1] + kappa*(theta - v[t-1])/252 + 
                         xi*np.sqrt(v[t-1]/252)*returns[t-1])
            ll += -0.5*(np.log(2*np.pi) + np.log(v[t]/252) + returns[t]**2/(v[t]/252))
        
        return -ll
    
    def calculate(self, close_prices: pd.Series) -> pd.Series:
        """Calculate Heston volatility"""
        if not SCIPY_AVAILABLE:
            self.logger.warning("scipy not available, using simple volatility")
            return close_prices.pct_change().rolling(self.lookback).std() * np.sqrt(252)
            
        returns = np.log(close_prices/close_prices.shift(1)).dropna()
        
        init_params = [3.0, 0.04, 0.1, -0.7, 0.04]
        bounds = ((0.1, 10), (0.001, 0.5), (0.01, 0.5), (-0.99, 0.99), (0.001, 0.5))
        
        try:
            res = minimize(
                self.heston_objective, 
                init_params, 
                args=(returns[-self.lookback:],),
                bounds=bounds, 
                method='L-BFGS-B'
            )
            
            kappa, theta, xi, rho, v0 = res.x
            volatility = np.sqrt(v0) * np.sqrt(252)  # Annualized
            
            return pd.Series(volatility, index=close_prices.index[-len(close_prices):])
            
        except Exception as e:
            self.logger.warning(f"Heston calibration failed: {str(e)}")
            return returns.rolling(self.lookback).std() * np.sqrt(252)

class ML_RSI:
    """
    Machine Learning enhanced RSI indicator.
    """
    
    def __init__(self, window: int = 14, lookahead: int = 5):
        self.window = window
        self.lookahead = lookahead
        self.logger = logging.getLogger('ML_RSI')
        
        if SKLEARN_AVAILABLE:
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            self.model = None
            self.logger.warning("scikit-learn not available, ML_RSI will use simple predictions")
        
    def calculate(self, prices: pd.Series, rsi_values: pd.Series) -> pd.Series:
        """Calculate ML-enhanced RSI predictions"""
        if not SKLEARN_AVAILABLE or self.model is None:
            self.logger.warning("Using simple momentum-based prediction instead of ML")
            momentum = prices.pct_change(self.lookahead).shift(-self.lookahead)
            return momentum.fillna(0)
            
        X = []
        y = []
        
        for i in range(self.window, len(prices) - self.lookahead):
            price_momentum = prices.iloc[i] / prices.iloc[i-self.window] - 1
            price_normalized = ((prices.iloc[i] - prices.iloc[i-self.window:i].min()) / 
                              (prices.iloc[i-self.window:i].max() - prices.iloc[i-self.window:i].min()))
            rsi_normalized = ((rsi_values.iloc[i] - rsi_values.iloc[i-self.window:i].min()) / 
                            (rsi_values.iloc[i-self.window:i].max() - rsi_values.iloc[i-self.window:i].min()))
            
            features = [
                rsi_values.iloc[i],
                price_momentum,
                price_normalized,
                rsi_normalized
            ]
            
            X.append(features)
            y.append(prices.iloc[i+self.lookahead] / prices.iloc[i] - 1)
        
        if len(X) < self.lookahead:
            return pd.Series(index=prices.index)
        
        self.model.fit(X[:-self.lookahead], y[:-self.lookahead])
        
        predictions = self.model.predict(X)
        
        return pd.Series(
            predictions, 
            index=prices.index[self.window:-self.lookahead]
        )

class OrderFlowImbalance:
    """
    Order flow imbalance indicator for tick data.
    """
    
    def __init__(self, window: int = 100):
        self.window = window
        self.logger = logging.getLogger('OrderFlowImbalance')
        
    def calculate(self, trades: pd.DataFrame) -> pd.Series:
        """
        Calculate order flow imbalance from trade data.
        
        Parameters:
        - trades: DataFrame with columns ['price', 'quantity', 'side']
                 where side is 1 for buy, -1 for sell
        """
        if not isinstance(trades, pd.DataFrame):
            raise ValueError("Requires tick data DataFrame")
        
        required_cols = ['price', 'quantity', 'side']
        if not all(col in trades.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        trades['dollar_volume'] = trades['price'] * trades['quantity']
        
        buys = trades[trades['side'] == 1]
        sells = trades[trades['side'] == -1]
        
        buy_vol = buys['dollar_volume'].rolling(window=self.window).sum()
        sell_vol = sells['dollar_volume'].rolling(window=self.window).sum()
        
        imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol)
        
        return imbalance.fillna(0)

class RegimeDetector:
    """
    Market regime detection using Hidden Markov Models.
    """
    
    def __init__(self, n_regimes: int = 3, lookback: int = 252):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.logger = logging.getLogger('RegimeDetector')
        
    def calculate(self, *indicators: pd.Series) -> pd.Series:
        """
        Detect market regimes from multiple indicators.
        
        Parameters:
        - indicators: Variable number of indicator series
        """
        if not HMM_AVAILABLE:
            self.logger.warning("HMM not available, returning simple volatility-based regimes")
            return self._simple_regime_detection(indicators[0])
        
        data = np.column_stack([ind.values for ind in indicators])
        
        valid_mask = ~np.isnan(data).any(axis=1)
        clean_data = data[valid_mask]
        
        if len(clean_data) < self.lookback:
            return pd.Series(0, index=indicators[0].index)
        
        try:
            model = hmm.GaussianHMM(
                n_components=self.n_regimes, 
                covariance_type="diag",
                random_state=42
            )
            model.fit(clean_data[-self.lookback:])
            
            regimes = model.predict(clean_data)
            
            result = pd.Series(0, index=indicators[0].index)
            result.iloc[valid_mask] = regimes
            
            return result
            
        except Exception as e:
            self.logger.warning(f"HMM regime detection failed: {str(e)}")
            return self._simple_regime_detection(indicators[0])
    
    def _simple_regime_detection(self, indicator: pd.Series) -> pd.Series:
        """Fallback simple regime detection based on volatility"""
        volatility = indicator.rolling(20).std()
        
        low_vol = volatility.quantile(0.33)
        high_vol = volatility.quantile(0.67)
        
        regimes = pd.Series(1, index=indicator.index)  # Default: medium volatility
        regimes[volatility <= low_vol] = 0  # Low volatility
        regimes[volatility >= high_vol] = 2  # High volatility
        
        return regimes
