import numpy as np
import pandas as pd

class FatTailRiskManager:
    """
    Standalone risk manager for fat-tail risk calculations
    Used for testing and verification of risk management strategies
    """
    
    def __init__(self):
        """Initialize the fat-tail risk manager"""
        self.max_position_size = 0.25   # 25% max position size
        
    def kelly_criterion(self, returns):
        """
        Calculate Kelly Criterion using standard Gaussian assumptions
        
        Args:
            returns: Series of historical returns
            
        Returns:
            float: Kelly fraction (position size as percentage of portfolio)
        """
        if isinstance(returns, list):
            returns = pd.Series(returns)
            
        if len(returns) < 10:
            return 0.01
            
        mu = returns.mean()
        sigma = returns.std()
        
        if sigma == 0:
            return 0.01
            
        kelly_fraction = (mu / (sigma ** 2))
        
        kelly_fraction = max(0.001, min(0.2, kelly_fraction))
        
        return kelly_fraction
        
    def kelly_criterion_fat_tail(self, returns):
        """
        Calculate Kelly Criterion adjusted for fat-tails using Expected Shortfall
        
        Args:
            returns: Series of historical returns
            
        Returns:
            float: Fat-tail adjusted Kelly fraction
        """
        if isinstance(returns, list):
            returns = pd.Series(returns)
            
        if len(returns) < 10:
            return 0.01
            
        var_95 = np.percentile(returns, 5)  # 5% Value at Risk
        es_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        sigma = returns.std()
        
        if sigma == 0:
            return 0.01
            
        kelly_fraction = (-es_95 / sigma) * 0.33
        
        kurtosis = returns.kurtosis()
        is_fat_tailed = kurtosis > 3.0  # Normal distribution has kurtosis = 3
        
        if is_fat_tailed:
            kelly_fraction *= (3.0 / kurtosis)
            
        kelly_fraction = max(0.001, min(0.2, kelly_fraction))
        
        if returns.std() > 0.05:  # If daily vol > 5%
            kelly_fraction *= 0.5  # Further reduce position size
            
        return kelly_fraction
        
    def check_max_drawdown(self, portfolio_value, current_value, drawdown_limit=0.2):
        """
        Check if current drawdown exceeds maximum allowed drawdown
        
        Args:
            portfolio_value: Initial portfolio value
            current_value: Current portfolio value
            drawdown_limit: Maximum allowed drawdown (default 20%)
            
        Returns:
            bool: True if emergency stop should be triggered
        """
        if portfolio_value <= 0:
            return True
            
        drawdown = (current_value - portfolio_value) / portfolio_value
        
        return drawdown < -drawdown_limit
        
    def circuit_breaker_triggered(self, daily_return):
        """
        Check if circuit breaker should be triggered based on daily return
        
        Args:
            daily_return: Daily return (e.g. -0.07 for -7%)
            
        Returns:
            bool: True if circuit breaker should be triggered
        """
        level_1 = -0.07  # 7% decline
        level_2 = -0.13  # 13% decline
        level_3 = -0.20  # 20% decline
        
        if daily_return <= level_3:
            return True
        elif daily_return <= level_2:
            return True
        elif daily_return <= level_1:
            return True
            
        return False
