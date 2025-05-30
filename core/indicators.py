"""
Advanced Trading Indicators

This module implements advanced indicators that enhance the existing system
without replacing current functionality.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
from hmmlearn import hmm
import logging

class HestonVolatility:
    """
    Implements Heston stochastic volatility model on top of existing price data
    Requires: Close prices (as pandas Series)
    """
    def __init__(self, lookback=30, risk_free=0.01):
        self.lookback = lookback
        self.r = risk_free
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def heston_objective(self, params, returns):
        kappa, theta, xi, rho, v0 = params
        n = len(returns)
        v = np.zeros(n)
        v[0] = v0
        ll = 0
        
        for t in range(1, n):
            v[t] = np.abs(v[t-1] + kappa*(theta - v[t-1])/252 + xi*np.sqrt(v[t-1]/252)*returns[t-1])
            ll += -0.5*(np.log(2*np.pi) + np.log(v[t]/252) + returns[t]**2/(v[t]/252))
        return -ll
        
    def calculate(self, close_prices):
        """Calculate Heston volatility for given price series"""
        try:
            returns = np.log(close_prices/close_prices.shift(1)).dropna()
            init_params = [3.0, 0.04, 0.1, -0.7, 0.04]
            bounds = ((0.1, 10), (0.001, 0.5), (0.01, 0.5), (-0.99, 0.99), (0.001, 0.5))
            res = minimize(self.heston_objective, init_params, args=(returns[-self.lookback:],),
                          bounds=bounds, method='L-BFGS-B')
            kappa, theta, xi, rho, v0 = res.x
            return pd.Series(np.sqrt(v0)*np.sqrt(252), index=close_prices.index[-len(returns):])
        except Exception as e:
            self.logger.error(f"Error calculating Heston volatility: {e}")
            return pd.Series(0.2, index=close_prices.index)  # Fallback to 20% volatility

class ML_RSI:
    """
    Augments traditional RSI with machine learning
    Requires: Existing RSI values (from the base system)
    """
    def __init__(self, window=14, lookahead=5):
        self.window = window
        self.lookahead = lookahead
        self.model = GradientBoostingRegressor(n_estimators=100)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def calculate(self, prices, rsi_values):
        """Calculate ML-enhanced RSI predictions"""
        try:
            X = []
            y = []
            for i in range(self.window, len(prices)-self.lookahead):
                features = [
                    rsi_values[i],
                    prices[i]/prices[i-self.window] - 1,  # momentum
                    (prices[i] - prices[i-self.window:i].min()) / 
                    (prices[i-self.window:i].max() - prices[i-self.window:i].min()),  # price position
                    (rsi_values[i] - rsi_values[i-self.window:i].min()) / 
                    (rsi_values[i-self.window:i].max() - rsi_values[i-self.window:i].min())  # RSI position
                ]
                X.append(features)
                y.append(prices[i+self.lookahead]/prices[i] - 1)  # forward return
                
            if len(X) > self.lookahead:
                self.model.fit(X[:-self.lookahead], y[:-self.lookahead])
                predictions = self.model.predict(X)
                return pd.Series(predictions, index=prices.index[self.window:-self.lookahead])
            else:
                return pd.Series(0.0, index=prices.index[self.window:])
        except Exception as e:
            self.logger.error(f"Error calculating ML RSI: {e}")
            return pd.Series(0.0, index=prices.index[self.window:])

class OrderFlowImbalance:
    """
    Requires: Tick-level trade data (price, quantity, side)
    """
    def __init__(self, window=100):
        self.window = window
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def calculate(self, trades):
        """
        Calculate order flow imbalance
        trades: DataFrame with columns ['price', 'quantity', 'side'] 
                where side=1 for buy, -1 for sell
        """
        try:
            if not isinstance(trades, pd.DataFrame):
                raise ValueError("Requires tick data DataFrame")
                
            trades['dollar_volume'] = trades['price'] * trades['quantity']
            buys = trades[trades['side'] == 1]
            sells = trades[trades['side'] == -1]
            
            buy_vol = buys['dollar_volume'].rolling(self.window).sum()
            sell_vol = sells['dollar_volume'].rolling(self.window).sum()
            
            imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol)
            return imbalance.fillna(0)
        except Exception as e:
            self.logger.error(f"Error calculating order flow imbalance: {e}")
            return pd.Series(0.0, index=trades.index if hasattr(trades, 'index') else [])

class RegimeDetector:
    """
    Uses Hidden Markov Models on top of existing indicators
    """
    def __init__(self, n_regimes=3, lookback=252):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def calculate(self, *indicators):
        """
        Detect market regimes using multiple indicators
        indicators: Variable number of pd.Series of same length
        """
        try:
            data = np.column_stack(indicators)
            model = hmm.GaussianHMM(n_components=self.n_regimes, covariance_type="diag")
            model.fit(data[-self.lookback:])
            regimes = model.predict(data)
            return pd.Series(regimes, index=indicators[0].index)
        except Exception as e:
            self.logger.error(f"Error detecting regimes: {e}")
            return pd.Series(0, index=indicators[0].index)
