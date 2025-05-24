import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class RiskManager:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.max_portfolio_risk = 0.01  # 1% max risk per trade
        self.max_position_size = 0.25   # 25% max position size
        self.volatility_lookback = 30   # days
        
    def calculate_position_size(self, symbol, confidence, history_data):
        """Calculate position size using Kelly Criterion with volatility scaling"""
        if '1m' not in history_data or history_data['1m'].empty:
            return 0.0
            
        recent_data = history_data['1m'].tail(self.volatility_lookback * 1440)  # 30 days of 1m data
        if len(recent_data) < 100:
            return 0.0
            
        returns = recent_data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252 * 1440)  # Annualized volatility
        
        win_rate = confidence  # Use confidence as win rate proxy
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0.01
        avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0.01
        
        if avg_loss == 0:
            kelly_fraction = 0.01
        else:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(returns)
            var = np.percentile(returns, 5)  # 95% confidence
            es = returns[returns <= var].mean() if len(returns[returns <= var]) > 0 else avg_loss
            kelly_fraction = -es / returns.std()  # Fat-tail adjusted Kelly
            kelly_fraction = max(0.001, min(0.5, kelly_fraction))  # Cap between 0.1% and 50%
            
        vol_scaling = min(1.0, 0.15 / volatility)  # Scale down if vol > 15%
        
        position_size = kelly_fraction * vol_scaling * confidence
        
        position_size = max(0.0, min(position_size, self.max_position_size))
        
        portfolio_value = self.algorithm.Portfolio.TotalPortfolioValue
        max_risk_size = (self.max_portfolio_risk * portfolio_value) / (volatility * portfolio_value)
        position_size = min(position_size, max_risk_size)
        
        return position_size
