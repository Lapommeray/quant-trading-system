import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy.stats import skew, kurtosis

class EnhancedRiskManagement:
    """
    Enhanced risk management with non-Gaussian risk metrics
    """
    def __init__(self, confidence_level=0.95, max_position_size=0.2):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.confidence_level = confidence_level
        self.max_position_size = max_position_size
        self.history = []
        
    def calculate_var(self, returns, alpha=0.05):
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Array of returns
            alpha: Significance level (default: 0.05)
            
        Returns:
            VaR value
        """
        try:
            if len(returns) < 2:
                return 0.0
                
            return np.percentile(returns, alpha * 100)
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {str(e)}")
            return 0.0
            
    def calculate_cvar(self, returns, alpha=0.05):
        """
        Calculate Conditional Value at Risk (CVaR)
        
        Args:
            returns: Array of returns
            alpha: Significance level (default: 0.05)
            
        Returns:
            CVaR value
        """
        try:
            if len(returns) < 2:
                return 0.0
                
            var = self.calculate_var(returns, alpha)
            return np.mean(returns[returns <= var])
        except Exception as e:
            self.logger.error(f"Error calculating CVaR: {str(e)}")
            return 0.0
            
    def adjusted_var(self, returns, alpha=0.05):
        """
        Calculate adjusted VaR using Cornish-Fisher expansion
        
        Args:
            returns: Array of returns
            alpha: Significance level (default: 0.05)
            
        Returns:
            Adjusted VaR value
        """
        try:
            if len(returns) < 4:
                return self.calculate_var(returns, alpha)
                
            s = skew(returns)
            k = kurtosis(returns)
            
            z = np.percentile(np.random.normal(0, 1, 10000), alpha * 100)
            
            z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * (k - 3) / 24 - (2*z**3 - 5*z) * s**2 / 36
            
            mu = np.mean(returns)
            sigma = np.std(returns)
            
            return mu + sigma * z_cf
        except Exception as e:
            self.logger.error(f"Error calculating adjusted VaR: {str(e)}")
            return self.calculate_var(returns, alpha)
            
    def calculate_max_drawdown(self, equity_curve):
        """
        Calculate maximum drawdown
        
        Args:
            equity_curve: Array of equity values
            
        Returns:
            Maximum drawdown value
        """
        try:
            if len(equity_curve) < 2:
                return 0.0
                
            running_max = np.maximum.accumulate(equity_curve)
            
            drawdown = (equity_curve - running_max) / running_max
            
            return np.min(drawdown)
        except Exception as e:
            self.logger.error(f"Error calculating maximum drawdown: {str(e)}")
            return 0.0
            
    def calculate_portfolio_metrics(self, trades):
        """
        Calculate portfolio-level risk metrics
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary with portfolio metrics
        """
        try:
            if not trades:
                return {
                    'var': 0.0,
                    'cvar': 0.0,
                    'adjusted_var': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'sortino_ratio': 0.0
                }
                
            returns = np.array([trade['return'] for trade in trades if 'return' in trade])
            
            if len(returns) < 2:
                return {
                    'var': 0.0,
                    'cvar': 0.0,
                    'adjusted_var': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'sortino_ratio': 0.0
                }
                
            equity_curve = np.cumprod(1 + returns)
            
            var = self.calculate_var(returns)
            cvar = self.calculate_cvar(returns)
            adjusted_var = self.adjusted_var(returns)
            max_drawdown = self.calculate_max_drawdown(equity_curve)
            
            risk_free_rate = 0.0
            excess_returns = returns - risk_free_rate
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0.0
            
            downside_returns = excess_returns[excess_returns < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0.0001
            sortino_ratio = np.mean(excess_returns) * np.sqrt(252) / downside_deviation if downside_deviation > 0 else 0.0
            
            metrics = {
                'var': float(var),
                'cvar': float(cvar),
                'adjusted_var': float(adjusted_var),
                'max_drawdown': float(max_drawdown),
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'timestamp': datetime.now().isoformat()
            }
            
            self.history.append(metrics)
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {
                'var': 0.0,
                'cvar': 0.0,
                'adjusted_var': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0
            }
            
    def apply_position_sizing(self, trades, portfolio_metrics=None):
        """
        Apply position sizing to trades
        
        Args:
            trades: List of trade dictionaries
            portfolio_metrics: Dictionary with portfolio metrics (default: None)
            
        Returns:
            List of trades with updated position sizes
        """
        try:
            if not trades:
                return []
                
            if portfolio_metrics is None:
                portfolio_metrics = self.calculate_portfolio_metrics(trades)
                
            for trade in trades:
                risk_factor = 1.0
                
                if 'adjusted_var' in portfolio_metrics and portfolio_metrics['adjusted_var'] != 0:
                    risk_factor = min(1.0, 0.02 / abs(portfolio_metrics['adjusted_var']))
                    
                position_size = min(self.max_position_size, risk_factor)
                
                trade['position_size'] = position_size
                trade['risk_factor'] = risk_factor
                
            return trades
        except Exception as e:
            self.logger.error(f"Error applying position sizing: {str(e)}")
            return trades
            
    def apply_stop_loss_take_profit(self, trades, stop_loss_pct=0.02, take_profit_pct=0.05):
        """
        Apply stop-loss and take-profit to trades
        
        Args:
            trades: List of trade dictionaries
            stop_loss_pct: Stop-loss percentage (default: 0.02)
            take_profit_pct: Take-profit percentage (default: 0.05)
            
        Returns:
            List of trades with updated stop-loss and take-profit levels
        """
        try:
            if not trades:
                return []
                
            for trade in trades:
                entry_price = trade.get('entry_price', 0)
                
                if entry_price <= 0:
                    continue
                    
                direction = trade.get('direction', 'long')
                
                if direction == 'long':
                    stop_loss = entry_price * (1 - stop_loss_pct)
                    take_profit = entry_price * (1 + take_profit_pct)
                else:
                    stop_loss = entry_price * (1 + stop_loss_pct)
                    take_profit = entry_price * (1 - take_profit_pct)
                    
                trade['stop_loss'] = stop_loss
                trade['take_profit'] = take_profit
                trade['stop_loss_pct'] = stop_loss_pct
                trade['take_profit_pct'] = take_profit_pct
                
            return trades
        except Exception as e:
            self.logger.error(f"Error applying stop-loss and take-profit: {str(e)}")
            return trades
            
    def calculate_risk_adjusted_returns(self, returns):
        """
        Calculate risk-adjusted returns
        
        Args:
            returns: Array of returns
            
        Returns:
            Dictionary with risk-adjusted return metrics
        """
        try:
            if len(returns) < 2:
                return {
                    'sharpe_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'calmar_ratio': 0.0,
                    'omega_ratio': 0.0
                }
                
            equity_curve = np.cumprod(1 + returns)
            
            risk_free_rate = 0.0
            excess_returns = returns - risk_free_rate
            
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0.0
            
            downside_returns = excess_returns[excess_returns < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0.0001
            sortino_ratio = np.mean(excess_returns) * np.sqrt(252) / downside_deviation if downside_deviation > 0 else 0.0
            
            max_drawdown = abs(self.calculate_max_drawdown(equity_curve))
            calmar_ratio = np.mean(excess_returns) * 252 / max_drawdown if max_drawdown > 0 else 0.0
            
            threshold = 0.0
            omega_ratio = np.sum(returns[returns > threshold] - threshold) / abs(np.sum(returns[returns < threshold] - threshold)) if np.sum(returns[returns < threshold] - threshold) != 0 else float('inf')
            
            return {
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'calmar_ratio': float(calmar_ratio),
                'omega_ratio': float(omega_ratio),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted returns: {str(e)}")
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'omega_ratio': 0.0
            }
